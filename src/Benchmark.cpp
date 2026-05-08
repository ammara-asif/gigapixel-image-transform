#include "Benchmark.h"
#include "BoundedTileQueue.h"   // used only in runSingle — kept out of Benchmark.h

#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <thread>
#include <atomic>
#include <algorithm>
#include <memory>
#include <cstdlib>
#include <chrono>

// ===========================================================
// Helpers
// ===========================================================

std::string BenchmarkSuite::modeName(ProcessingMode m) const
{
    switch (m)
    {
        case ProcessingMode::CPU_ONLY:      return "CPU-Only";
        case ProcessingMode::GPU_ONLY:      return "GPU-Only";
        case ProcessingMode::HETEROGENEOUS: return "Heterogeneous";
    }
    return "Unknown";
}

void BenchmarkSuite::printSeparator(int width) const
{
    std::cout << std::string(width, '-') << "\n";
}

// ===========================================================
// createSyntheticTile
// Allocates a pinned-memory tile (same path as TileReader)
// and fills it with random RGB data so workers have real
// pixel values to operate on.
// ===========================================================
Tile BenchmarkSuite::createSyntheticTile(int tileSize, int overlap, TransformOperation op)
{
    Tile tile;
    tile.width     = tileSize + 2 * overlap;
    tile.height    = tileSize + 2 * overlap;
    tile.overlap   = overlap;
    tile.channels  = CHANNELS;
    tile.operation = op;
    tile.x         = 0;
    tile.y         = 0;
    tile.out_x     = 0;
    tile.out_y     = 0;

    size_t bytes = static_cast<size_t>(tile.width) * tile.height * CHANNELS;
    tile.allocate(bytes, CHANNELS);

    // Fill with random pixel values
    uint8_t* ptr = tile.getRawPtr();
    for (size_t i = 0; i < bytes; i += 3)
    {
        ptr[i]     = static_cast<uint8_t>(rand() % 256);
        ptr[i + 1] = static_cast<uint8_t>(rand() % 256);
        ptr[i + 2] = static_cast<uint8_t>(rand() % 256);
    }
    return tile;
}

// ===========================================================
// runSingle
// Spins up the same reader → queue → workers pipeline that
// main.cpp uses, but with synthetic tiles and no disk writer.
// Measures wall-clock throughput and per-tile latency.
// ===========================================================
BenchmarkResult BenchmarkSuite::runSingle(
    TransformOperation  op,
    const std::string&  opName,
    const std::string&  imageSize,
    int                 numTiles,
    int                 tileSize,
    int                 overlap,
    int                 pipelineDepth,
    ProcessingMode      mode)
{
    unsigned int numCores   = std::thread::hardware_concurrency();
    unsigned int numWorkers = std::max(1u, numCores - 1u);

    // Queue capacity = pipelineDepth * numWorkers  (mirrors main.cpp approach)
    size_t queueCapacity = static_cast<size_t>(pipelineDepth) * numWorkers;

    // VRAM pool  — always created; CPU mode never calls acquireBuffer()
    size_t maxTileBytes = static_cast<size_t>(tileSize + 2 * overlap) *
                          static_cast<size_t>(tileSize + 2 * overlap) *
                          (CHANNELS + 1); // +1 for gray output staging
    VramManager vramPool(static_cast<int>(numWorkers) + 2, maxTileBytes);

    // ----------------------------------------------------------
    // Build per-mode workers
    // Each CPU worker thread gets its own OptimizedCPUWorker to
    // avoid data races on the internal perfStats struct.
    // GPU and Scheduler objects are thread-safe (mutex-guarded).
    // ----------------------------------------------------------
    std::vector<std::unique_ptr<OptimizedCPUWorker>> cpuWorkers;
    std::unique_ptr<AsyncGPUWorker>                  gpuWorker;
    std::unique_ptr<Scheduler>                       scheduler;

    if (mode == ProcessingMode::CPU_ONLY)
    {
        for (unsigned int i = 0; i < numWorkers; ++i)
            cpuWorkers.push_back(std::make_unique<OptimizedCPUWorker>());
    }
    else if (mode == ProcessingMode::GPU_ONLY)
    {
        gpuWorker = std::make_unique<AsyncGPUWorker>();
    }
    else
    {
        scheduler = std::make_unique<Scheduler>();
    }

    // ----------------------------------------------------------
    // Warm-up pass (5 tiles) — not timed
    // Ensures CUDA context is alive and caches are warm before
    // the actual measurement begins.
    // ----------------------------------------------------------
    {
        OptimizedCPUWorker warmCpu;
        for (int w = 0; w < 5; ++w)
        {
            Tile t = createSyntheticTile(tileSize, overlap, op);
            if (mode == ProcessingMode::CPU_ONLY)
            {
                warmCpu.execute(t);
            }
            else if (mode == ProcessingMode::GPU_ONLY)
            {
                uint8_t* d = vramPool.acquireBuffer();
                gpuWorker->execute(t, d);
                vramPool.releaseBuffer(d);
            }
            else
            {
                uint8_t* d = vramPool.acquireBuffer();
                scheduler->dispatch(t, d);
                vramPool.releaseBuffer(d);
            }
        }
    }

    // ----------------------------------------------------------
    // Timed measurement pass
    // ----------------------------------------------------------
    BoundedTileQueue          inputQueue(queueCapacity);
    std::atomic<long long>    totalLatencyUs{0};
    std::atomic<int>          tilesCompleted{0};

    // Start global wall-clock
    auto globalStart = std::chrono::high_resolution_clock::now();

    // --- Worker threads ---
    std::vector<std::thread> workers;
    for (unsigned int i = 0; i < numWorkers; ++i)
    {
        workers.emplace_back([&, i]()
        {
            Tile tile;
            while (inputQueue.pop(tile))
            {
                auto t0 = std::chrono::high_resolution_clock::now();

                if (mode == ProcessingMode::CPU_ONLY)
                {
                    cpuWorkers[i]->execute(tile);
                }
                else if (mode == ProcessingMode::GPU_ONLY)
                {
                    uint8_t* d = vramPool.acquireBuffer();
                    gpuWorker->execute(tile, d);
                    vramPool.releaseBuffer(d);
                }
                else
                {
                    uint8_t* d = vramPool.acquireBuffer();
                    scheduler->dispatch(tile, d);
                    vramPool.releaseBuffer(d);
                }

                auto t1  = std::chrono::high_resolution_clock::now();
                long long us = std::chrono::duration_cast<
                    std::chrono::microseconds>(t1 - t0).count();

                totalLatencyUs.fetch_add(us);
                tilesCompleted.fetch_add(1);
            }
        });
    }

    // --- Reader thread: push synthetic tiles ---
    std::thread readerThread([&]()
    {
        for (int k = 0; k < numTiles; ++k)
        {
            Tile t = createSyntheticTile(tileSize, overlap, op);
            inputQueue.push(std::move(t));
        }
        inputQueue.setFinished();
    });

    readerThread.join();
    for (auto& w : workers)
        w.join();

    auto globalEnd = std::chrono::high_resolution_clock::now();

    // ----------------------------------------------------------
    // Compute metrics
    // ----------------------------------------------------------
    double elapsedSec = std::chrono::duration<double>(globalEnd - globalStart).count();

    // Throughput is measured on logical (non-overlapped) pixels
    long long totalPixels     = static_cast<long long>(tileSize) * tileSize * tilesCompleted;
    double    throughputMpps  = (totalPixels / 1.0e6) / elapsedSec;
    double    avgLatencyMs    = (tilesCompleted > 0)
                                ? (totalLatencyUs.load() / 1000.0 / tilesCompleted)
                                : 0.0;

    BenchmarkResult result;
    result.operation      = opName;
    result.mode           = modeName(mode);
    result.imageSize      = imageSize;
    result.tileSize       = tileSize;
    result.overlap        = overlap;
    result.pipelineDepth  = pipelineDepth;
    result.numTiles       = tilesCompleted;
    result.throughputMpps = throughputMpps;
    result.latencyMs      = avgLatencyMs;
    result.speedupVsCPU   = 1.0; // filled by computeSpeedups()
    return result;
}

// ===========================================================
// computeSpeedups
// For every GPU-Only / Heterogeneous result, find the matching
// CPU-Only result and compute the speedup ratio.
// ===========================================================
void BenchmarkSuite::computeSpeedups()
{
    for (auto& r : results_)
    {
        if (r.mode == "CPU-Only") { r.speedupVsCPU = 1.0; continue; }

        for (const auto& cpu : results_)
        {
            if (cpu.mode         == "CPU-Only"   &&
                cpu.operation    == r.operation   &&
                cpu.imageSize    == r.imageSize   &&
                cpu.tileSize     == r.tileSize    &&
                cpu.overlap      == r.overlap     &&
                cpu.pipelineDepth == r.pipelineDepth)
            {
                r.speedupVsCPU = (cpu.throughputMpps > 0.0)
                                 ? r.throughputMpps / cpu.throughputMpps
                                 : 1.0;
                break;
            }
        }
    }
}

// ===========================================================
// runAll — main entry point
// ===========================================================
void BenchmarkSuite::runAll(bool quickMode)
{
    // --- Operations ---
    struct OpCfg { TransformOperation op; std::string name; };
    const std::vector<OpCfg> ops = {
        { TransformOperation::GRAYSCALE,    "Grayscale"  },
        { TransformOperation::ROTATE_90_CW, "Rotate90CW" }
    };

    // --- Image sizes → tile counts
    // We process enough tiles to produce stable throughput numbers.
    // Larger "image size" labels use more tiles so pipeline-depth
    // effects become visible in the measurements.
    struct SizeCfg { std::string label; int tiles; };
    std::vector<SizeCfg> sizes;
    if (quickMode)
        sizes = { {"1GP", 60}, {"10GP", 150} };
    else
        sizes = { {"1GP", 100}, {"10GP", 300}, {"50GP", 600} };

    // --- Tile sizes (pixels) ---
    std::vector<int> tileSizes = quickMode
        ? std::vector<int>{256, 512}
        : std::vector<int>{256, 512, 1024};

    // --- Overlap widths (pixels) ---
    std::vector<int> overlaps = quickMode
        ? std::vector<int>{0, 32}
        : std::vector<int>{0, 16, 32, 64};

    // --- Pipeline depths (queue = depth × numWorkers) ---
    std::vector<int> depths = quickMode
        ? std::vector<int>{4, 8}
        : std::vector<int>{4, 8, 16};

    // --- Processing modes ---
    const std::vector<ProcessingMode> modes = {
        ProcessingMode::CPU_ONLY,
        ProcessingMode::GPU_ONLY,
        ProcessingMode::HETEROGENEOUS
    };

    int total = static_cast<int>(
        ops.size() * sizes.size() * tileSizes.size() *
        overlaps.size() * depths.size() * modes.size());

    int done = 0;
    std::cout << "\n[Benchmark] Starting — " << total << " configurations\n\n";

    for (const auto& op  : ops)
    for (const auto& sz  : sizes)
    for (int  ts         : tileSizes)
    for (int  ov         : overlaps)
    for (int  pd         : depths)
    for (auto modeVal    : modes)
    {
        ++done;
        std::cout << "[" << std::setw(3) << done << "/" << total << "] "
                  << std::left
                  << std::setw(12) << op.name
                  << std::setw(6)  << sz.label
                  << "tile=" << std::setw(5) << ts
                  << "ovlp=" << std::setw(4) << ov
                  << "depth=" << std::setw(3) << pd
                  << std::setw(14) << modeName(modeVal)
                  << "... " << std::flush;

        try
        {
            BenchmarkResult r = runSingle(
                op.op, op.name, sz.label, sz.tiles,
                ts, ov, pd, modeVal);

            results_.push_back(r);

            std::cout << std::fixed << std::setprecision(1)
                      << r.throughputMpps << " MP/s  "
                      << std::setprecision(2) << r.latencyMs << " ms/tile\n";
        }
        catch (const std::exception& e)
        {
            std::cout << "FAILED: " << e.what() << "\n";
        }
    }

    computeSpeedups();
    std::cout << "\n[Benchmark] Complete — " << results_.size() << " results collected.\n";
}

// ===========================================================
// printThroughputTable
// ===========================================================
void BenchmarkSuite::printThroughputTable() const
{
    const int W = 95;
    std::cout << "\n";
    printSeparator(W);
    std::cout << " THROUGHPUT COMPARISON  (Megapixels / second)\n";
    printSeparator(W);
    std::cout << std::left
              << std::setw(8)  << "Size"
              << std::setw(13) << "Operation"
              << std::setw(9)  << "TileSz"
              << std::setw(9)  << "Overlap"
              << std::setw(8)  << "Depth"
              << std::setw(14) << "CPU-Only"
              << std::setw(14) << "GPU-Only"
              << std::setw(14) << "Heterogeneous"
              << "\n";
    printSeparator(W);

    // Collect unique (imageSize, op, tileSize, overlap, depth) combinations
    using Key = std::tuple<std::string,std::string,int,int,int>;
    std::vector<Key> combos;
    for (const auto& r : results_)
    {
        Key k{r.imageSize, r.operation, r.tileSize, r.overlap, r.pipelineDepth};
        if (std::find(combos.begin(), combos.end(), k) == combos.end())
            combos.push_back(k);
    }

    for (const auto& [imgSz, opName, ts, ov, pd] : combos)
    {
        double cpu = 0, gpu = 0, het = 0;
        for (const auto& r : results_)
        {
            if (r.imageSize == imgSz && r.operation == opName &&
                r.tileSize  == ts    && r.overlap   == ov    &&
                r.pipelineDepth == pd)
            {
                if (r.mode == "CPU-Only")      cpu = r.throughputMpps;
                if (r.mode == "GPU-Only")      gpu = r.throughputMpps;
                if (r.mode == "Heterogeneous") het = r.throughputMpps;
            }
        }
        std::cout << std::fixed << std::setprecision(1) << std::left
                  << std::setw(8)  << imgSz
                  << std::setw(13) << opName
                  << std::setw(9)  << ts
                  << std::setw(9)  << ov
                  << std::setw(8)  << pd
                  << std::setw(14) << cpu
                  << std::setw(14) << gpu
                  << std::setw(14) << het
                  << "\n";
    }
    printSeparator(W);
}

// ===========================================================
// printLatencyTable
// ===========================================================
void BenchmarkSuite::printLatencyTable() const
{
    const int W = 95;
    std::cout << "\n";
    printSeparator(W);
    std::cout << " LATENCY COMPARISON  (milliseconds per tile)\n";
    printSeparator(W);
    std::cout << std::left
              << std::setw(8)  << "Size"
              << std::setw(13) << "Operation"
              << std::setw(9)  << "TileSz"
              << std::setw(9)  << "Overlap"
              << std::setw(8)  << "Depth"
              << std::setw(14) << "CPU-Only"
              << std::setw(14) << "GPU-Only"
              << std::setw(14) << "Heterogeneous"
              << "\n";
    printSeparator(W);

    using Key = std::tuple<std::string,std::string,int,int,int>;
    std::vector<Key> combos;
    for (const auto& r : results_)
    {
        Key k{r.imageSize, r.operation, r.tileSize, r.overlap, r.pipelineDepth};
        if (std::find(combos.begin(), combos.end(), k) == combos.end())
            combos.push_back(k);
    }

    for (const auto& [imgSz, opName, ts, ov, pd] : combos)
    {
        double cpu = 0, gpu = 0, het = 0;
        for (const auto& r : results_)
        {
            if (r.imageSize == imgSz && r.operation == opName &&
                r.tileSize  == ts    && r.overlap   == ov    &&
                r.pipelineDepth == pd)
            {
                if (r.mode == "CPU-Only")      cpu = r.latencyMs;
                if (r.mode == "GPU-Only")      gpu = r.latencyMs;
                if (r.mode == "Heterogeneous") het = r.latencyMs;
            }
        }
        std::cout << std::fixed << std::setprecision(2) << std::left
                  << std::setw(8)  << imgSz
                  << std::setw(13) << opName
                  << std::setw(9)  << ts
                  << std::setw(9)  << ov
                  << std::setw(8)  << pd
                  << std::setw(14) << cpu
                  << std::setw(14) << gpu
                  << std::setw(14) << het
                  << "\n";
    }
    printSeparator(W);
}

// ===========================================================
// printSpeedupTable
// ===========================================================
void BenchmarkSuite::printSpeedupTable() const
{
    const int W = 85;
    std::cout << "\n";
    printSeparator(W);
    std::cout << " SPEEDUP vs CPU-Only  (higher is better)\n";
    printSeparator(W);
    std::cout << std::left
              << std::setw(8)  << "Size"
              << std::setw(13) << "Operation"
              << std::setw(9)  << "TileSz"
              << std::setw(9)  << "Overlap"
              << std::setw(8)  << "Depth"
              << std::setw(16) << "GPU Speedup"
              << std::setw(16) << "Hetero Speedup"
              << "\n";
    printSeparator(W);

    using Key = std::tuple<std::string,std::string,int,int,int>;
    std::vector<Key> combos;
    for (const auto& r : results_)
    {
        Key k{r.imageSize, r.operation, r.tileSize, r.overlap, r.pipelineDepth};
        if (std::find(combos.begin(), combos.end(), k) == combos.end())
            combos.push_back(k);
    }

    for (const auto& [imgSz, opName, ts, ov, pd] : combos)
    {
        double gpuSpeedup = 1.0, hetSpeedup = 1.0;
        for (const auto& r : results_)
        {
            if (r.imageSize == imgSz && r.operation == opName &&
                r.tileSize  == ts    && r.overlap   == ov    &&
                r.pipelineDepth == pd)
            {
                if (r.mode == "GPU-Only")      gpuSpeedup = r.speedupVsCPU;
                if (r.mode == "Heterogeneous") hetSpeedup = r.speedupVsCPU;
            }
        }

        // Format speedup as e.g. "2.35x"
        std::ostringstream gpuStr, hetStr;
        gpuStr << std::fixed << std::setprecision(2) << gpuSpeedup << "x";
        hetStr << std::fixed << std::setprecision(2) << hetSpeedup << "x";

        std::cout << std::left
                  << std::setw(8)  << imgSz
                  << std::setw(13) << opName
                  << std::setw(9)  << ts
                  << std::setw(9)  << ov
                  << std::setw(8)  << pd
                  << std::setw(16) << gpuStr.str()
                  << std::setw(16) << hetStr.str()
                  << "\n";
    }
    printSeparator(W);
}

// ===========================================================
// exportCSV
// ===========================================================
// ===========================================================
// runWithImage
// Reads real tiles from a user-provided TIFF image and runs
// them through CPU-Only, GPU-Only, and Heterogeneous modes.
// Reports throughput and latency for each mode.
// ===========================================================
void BenchmarkSuite::runWithImage(const std::string& imagePath,
                                   int tileSize,
                                   int overlap,
                                   int pipelineDepth)
{
    std::cout << "\n";
    std::cout << "=======================================================\n";
    std::cout << " REAL IMAGE BENCHMARK\n";
    std::cout << " Image      : " << imagePath << "\n";
    std::cout << " Tile size  : " << tileSize << "x" << tileSize << "\n";
    std::cout << " Overlap    : " << overlap << " px\n";
    std::cout << " Pipeline   : depth=" << pipelineDepth << "\n";
    std::cout << "=======================================================\n";

    // --- Open image ---
    TileReader reader(imagePath);
    int imgW     = reader.getImageWidth();
    int imgH     = reader.getImageHeight();
    int channels = reader.getChannels();
    long long totalPixels = (long long)imgW * imgH;

    std::cout << " Resolution : " << imgW << " x " << imgH
              << "  (" << std::fixed << std::setprecision(2)
              << totalPixels / 1.0e6 << " MP)\n";
    std::cout << " Channels   : " << channels << "\n";

    // --- Read all tiles from disk upfront ---
    // Tiles are stored so the same data is used for all three modes
    int nTilesX   = (imgW + tileSize - 1) / tileSize;
    int nTilesY   = (imgH + tileSize - 1) / tileSize;
    int totalTiles = nTilesX * nTilesY;

    std::cout << " Tiles      : " << nTilesX << " x " << nTilesY
              << " = " << totalTiles << " tiles\n\n";

    // Load tiles into a vector (pinned memory via Tile::allocate)
    std::vector<Tile> sourceTiles;
    sourceTiles.reserve(totalTiles);

    std::cout << "[Image] Reading tiles from disk...\n";
    for (int ty = 0; ty < imgH; ty += tileSize)
    {
        for (int tx = 0; tx < imgW; tx += tileSize)
        {
            int tw = std::min(tileSize, imgW - tx);
            int th = std::min(tileSize, imgH - ty);
            Tile t = reader.getTile(tx, ty, tw, th, overlap);
            t.operation = TransformOperation::GRAYSCALE;
            sourceTiles.push_back(std::move(t));
        }
    }
    std::cout << "[Image] " << sourceTiles.size() << " tiles loaded.\n\n";

    // --- Run each operation x mode combination ---
    struct OpCfg { TransformOperation op; std::string name; };
    const std::vector<OpCfg> ops = {
        { TransformOperation::GRAYSCALE,    "Grayscale"  },
        { TransformOperation::ROTATE_90_CW, "Rotate90CW" }
    };

    const std::vector<ProcessingMode> modes = {
        ProcessingMode::CPU_ONLY,
        ProcessingMode::GPU_ONLY,
        ProcessingMode::HETEROGENEOUS
    };

    unsigned int numCores   = std::thread::hardware_concurrency();
    unsigned int numWorkers = std::max(1u, numCores - 1u);
    size_t queueCapacity    = (size_t)pipelineDepth * numWorkers;

    size_t maxTileBytes = (size_t)(tileSize + 2 * overlap) *
                          (size_t)(tileSize + 2 * overlap) *
                          (channels + 1);
    VramManager vramPool((int)numWorkers + 2, maxTileBytes);

    std::string imageLabel = std::to_string((int)(totalPixels / 1.0e6)) + "MP";

    for (const auto& op : ops)
    {
        // Update operation on all tiles
        for (auto& t : sourceTiles)
            t.operation = op.op;

        for (auto modeVal : modes)
        {
            std::cout << "[Running] " << std::left
                      << std::setw(12) << op.name
                      << std::setw(15) << modeName(modeVal)
                      << "... " << std::flush;

            // Build workers
            std::vector<std::unique_ptr<OptimizedCPUWorker>> cpuWorkers;
            std::unique_ptr<AsyncGPUWorker> gpuWorker;
            std::unique_ptr<Scheduler>      scheduler;

            if (modeVal == ProcessingMode::CPU_ONLY)
                for (unsigned int i = 0; i < numWorkers; ++i)
                    cpuWorkers.push_back(std::make_unique<OptimizedCPUWorker>());
            else if (modeVal == ProcessingMode::GPU_ONLY)
                gpuWorker = std::make_unique<AsyncGPUWorker>();
            else
                scheduler = std::make_unique<Scheduler>();

            BoundedTileQueue          queue(queueCapacity);
            std::atomic<long long>    totalLatUs{0};
            std::atomic<int>          done{0};

            auto globalStart = std::chrono::high_resolution_clock::now();

            // Worker threads
            std::vector<std::thread> workers;
            for (unsigned int i = 0; i < numWorkers; ++i)
            {
                workers.emplace_back([&, i]()
                {
                    Tile tile;
                    while (queue.pop(tile))
                    {
                        auto t0 = std::chrono::high_resolution_clock::now();

                        if (modeVal == ProcessingMode::CPU_ONLY)
                            cpuWorkers[i]->execute(tile);
                        else if (modeVal == ProcessingMode::GPU_ONLY)
                        {
                            uint8_t* d = vramPool.acquireBuffer();
                            gpuWorker->execute(tile, d);
                            vramPool.releaseBuffer(d);
                        }
                        else
                        {
                            uint8_t* d = vramPool.acquireBuffer();
                            scheduler->dispatch(tile, d);
                            vramPool.releaseBuffer(d);
                        }

                        auto t1 = std::chrono::high_resolution_clock::now();
                        totalLatUs += std::chrono::duration_cast<
                            std::chrono::microseconds>(t1 - t0).count();
                        done++;
                    }
                });
            }

            // Reader thread — copies from sourceTiles so original data is preserved
            std::thread reader([&]()
            {
                for (const auto& src : sourceTiles)
                {
                    // shallow copy is fine — shared_ptr keeps data alive
                    queue.push(src);
                }
                queue.setFinished();
            });

            reader.join();
            for (auto& w : workers) w.join();

            auto globalEnd = std::chrono::high_resolution_clock::now();
            double elapsedSec = std::chrono::duration<double>(
                                    globalEnd - globalStart).count();

            long long processedPixels = (long long)tileSize * tileSize * done.load();
            double throughput = (processedPixels / 1.0e6) / elapsedSec;
            double latency    = done > 0
                                ? totalLatUs.load() / 1000.0 / done.load()
                                : 0.0;

            std::cout << std::fixed << std::setprecision(1)
                      << throughput << " MP/s  "
                      << std::setprecision(2) << latency << " ms/tile\n";

            // Store result
            BenchmarkResult r;
            r.operation      = op.name;
            r.mode           = modeName(modeVal);
            r.imageSize      = imageLabel;
            r.tileSize       = tileSize;
            r.overlap        = overlap;
            r.pipelineDepth  = pipelineDepth;
            r.numTiles       = done.load();
            r.throughputMpps = throughput;
            r.latencyMs      = latency;
            r.speedupVsCPU   = 1.0;
            results_.push_back(r);
        }
    }

    computeSpeedups();

    // Print summary tables for this image
    printThroughputTable();
    printLatencyTable();
    printSpeedupTable();
}

void BenchmarkSuite::exportCSV(const std::string& filename) const
{
    std::ofstream f(filename);
    if (!f.is_open())
    {
        std::cerr << "[Benchmark] ERROR: Cannot open CSV file: " << filename << "\n";
        return;
    }

    f << "Operation,Mode,ImageSize,TileSize,Overlap,"
      << "PipelineDepth,NumTiles,ThroughputMpps,LatencyMs,SpeedupVsCPU\n";

    for (const auto& r : results_)
    {
        f << r.operation     << ","
          << r.mode          << ","
          << r.imageSize     << ","
          << r.tileSize      << ","
          << r.overlap       << ","
          << r.pipelineDepth << ","
          << r.numTiles      << ","
          << std::fixed << std::setprecision(3)
          << r.throughputMpps << ","
          << r.latencyMs      << ","
          << r.speedupVsCPU   << "\n";
    }

    std::cout << "[Benchmark] CSV saved → " << filename << "\n";
}
