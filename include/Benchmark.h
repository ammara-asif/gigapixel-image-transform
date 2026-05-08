#pragma once

#include <string>
#include <vector>

#include "Tile.h"
#include "TileReader.h"
#include "OptimizedCPUWorker.h"
#include "AsyncGPUWorker.h"
#include "VramManager.h"
#include "Scheduler.h"

// -------------------------------------------------------
// ProcessingMode: controls which execution path is used
// -------------------------------------------------------
enum class ProcessingMode
{
    CPU_ONLY,       // Only OptimizedCPUWorker
    GPU_ONLY,       // Only AsyncGPUWorker  (falls back to CPU for unsupported ops)
    HETEROGENEOUS   // Scheduler decides per tile
};

// -------------------------------------------------------
// BenchmarkResult: one row of benchmark data
// -------------------------------------------------------
struct BenchmarkResult
{
    std::string operation;    // "Grayscale" or "Rotate90CW"
    std::string mode;         // "CPU-Only", "GPU-Only", "Heterogeneous"
    std::string imageSize;    // "1GP", "10GP", "50GP"
    int         tileSize;     // logical tile width/height (pixels)
    int         overlap;      // overlap border added to each tile edge (pixels)
    int         pipelineDepth;// BoundedTileQueue depth multiplier
    int         numTiles;     // actual tiles processed in the timed run
    double      throughputMpps; // megapixels per second (end-to-end)
    double      latencyMs;      // average per-tile processing time (ms)
    double      speedupVsCPU;   // throughput ratio vs CPU-Only baseline
};

// -------------------------------------------------------
// BenchmarkSuite: runs the full parameter sweep and
// reports throughput, latency, and speedup tables
// -------------------------------------------------------
class BenchmarkSuite
{
public:
    BenchmarkSuite() = default;

    // Run all configurations.
    // quickMode=true  -> smaller tile counts, fewer params  (good for quick Colab test)
    // quickMode=false -> full sweep with 1GP / 10GP / 50GP simulation
    void runAll(bool quickMode = false);

    // Print formatted throughput table (MP/s)
    void printThroughputTable() const;

    // Print formatted latency table (ms per tile)
    void printLatencyTable() const;

    // Print speedup vs CPU-Only table
    void printSpeedupTable() const;

    // Run benchmark on a real user-provided image
    // Sweeps CPU-Only / GPU-Only / Heterogeneous on the actual image tiles
    void runWithImage(const std::string& imagePath,
                      int tileSize     = 512,
                      int overlap      = 0,
                      int pipelineDepth = 8);

    // Write all results to a CSV file
    void exportCSV(const std::string& filename) const;

private:
    std::vector<BenchmarkResult> results_;

    static constexpr int CHANNELS = 3; // RGB image

    // ---- helpers ----

    // Allocate a pinned-memory tile filled with random RGB pixels
    Tile createSyntheticTile(int tileSize, int overlap, TransformOperation op);

    // Run one benchmark configuration and return the result
    BenchmarkResult runSingle(
        TransformOperation  op,
        const std::string&  opName,
        const std::string&  imageSize,
        int                 numTiles,
        int                 tileSize,
        int                 overlap,
        int                 pipelineDepth,
        ProcessingMode      mode
    );

    // Fill speedupVsCPU for every non-CPU result
    void computeSpeedups();

    std::string modeName(ProcessingMode m) const;
    void        printSeparator(int width) const;
};
