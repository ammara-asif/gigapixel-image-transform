#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <string>
#include <mutex>
#include <memory>

#include "TileReader.h"
#include "Tile.h"
#include "BoundedTileQueue.h"
#include "OutputWriter.h"
#include "Transform.h"
#include "Scheduler.h"
#include "TileOptimizer.h"
#include "VramManager.h"
#include "PipelineComposition.h"
// --- Thread-Safe Console Logging ---
std::mutex printMutex;
void logMessage(const std::string &msg)
{
    std::lock_guard<std::mutex> lock(printMutex);
    std::cout << msg << std::endl;
}

/**
 * Display pipeline menu and return user's choice
 */
std::shared_ptr<TransformPipeline> showPipelineMenu()
{
    std::cout << "\n";
    std::cout << "======================================\n";
    std::cout << " Milestone 3: Pipeline Composition\n";
    std::cout << "======================================\n";
    std::cout << "1. Single Grayscale (Baseline)\n";
    std::cout << "2. Single Rotate 90° CW (Baseline)\n";
    std::cout << "3. Rotate → Resize → Grayscale (Pre-built)\n";
    std::cout << "4. Rotate → Gaussian → Color Correct (Pre-built)\n";
    std::cout << "5. Gaussian → Sobel (Edge Detection)\n";
    std::cout << "6. Median → Gaussian (Noise Reduction)\n";
    std::cout << "7. Custom Pipeline Builder\n";
    std::cout << "Select option (1-7): ";

    int choice = 0;
    while (!(std::cin >> choice) || (choice < 1 || choice > 7))
    {
        std::cout << "Invalid input. Please enter 1-7: ";
        std::cin.clear();
        std::cin.ignore(10000, '\n');
    }

    std::cin.ignore(10000, '\n'); // Clear input buffer

    auto pipeline = std::make_shared<TransformPipeline>();

    switch (choice)
    {
    case 1:
        pipeline = std::make_shared<TransformPipeline>("grayscale");
        pipeline->add(TransformOperation::GRAYSCALE);
        std::cout << "\n[Pipeline] Selected: Single Grayscale\n";
        break;

    case 2:
        pipeline = std::make_shared<TransformPipeline>("rotate_cw");
        pipeline->add(TransformOperation::ROTATE_90_CW);
        std::cout << "\n[Pipeline] Selected: Single Rotate 90° CW\n";
        break;

    case 3:
        pipeline = PipelineComposer::rotateResizeGrayscale(0.5f, 0.5f);
        std::cout << "\n[Pipeline] Selected: Rotate → Resize (0.5x) → Grayscale\n";
        break;

    case 4:
        pipeline = PipelineComposer::rotateFilterColorCorrect(5, 1.0f);
        std::cout << "\n[Pipeline] Selected: Rotate → Gaussian (5x5) → Color Correct\n";
        break;

    case 5:
        pipeline = PipelineComposer::edgeDetectionPipeline();
        std::cout << "\n[Pipeline] Selected: Gaussian (3x3) → Sobel Edge Detection\n";
        break;

    case 6:
        pipeline = PipelineComposer::noiseReductionPipeline();
        std::cout << "\n[Pipeline] Selected: Median (5x5) → Gaussian (3x3) Noise Reduction\n";
        break;

    case 7:
    {
        std::cout << "\n[Pipeline Builder] Custom Pipeline Construction\n";
        std::cout << "Available Transforms:\n";
        std::cout << "  0. IDENTITY\n";
        std::cout << "  1. GRAYSCALE\n";
        std::cout << "  2. ROTATE_90_CW\n";
        std::cout << "  3. RESIZE\n";
        std::cout << "  4. COLOR_CORRECTION\n";
        std::cout << "  5. FILTER_GAUSSIAN\n";
        std::cout << "  6. FILTER_MEDIAN\n";
        std::cout << "  7. FILTER_SOBEL\n";

        pipeline = std::make_shared<TransformPipeline>("custom");

        int numStages = 0;
        std::cout << "\nHow many stages? (1-5): ";
        std::cin >> numStages;
        numStages = std::max(1, std::min(numStages, 5));

        for (int i = 0; i < numStages; ++i)
        {
            std::cout << "Stage " << (i + 1) << " - Transform (0-7): ";
            int op = 0;
            std::cin >> op;
            op = std::max(0, std::min(op, 7));

            TransformOperation opType = static_cast<TransformOperation>(op);

            // Get parameters if needed
            TransformParams params;
            if (op == 3) // RESIZE
            {
                std::cout << "  Scale X (0.1-2.0): ";
                std::cin >> params.scaleX;
                std::cout << "  Scale Y (0.1-2.0): ";
                std::cin >> params.scaleY;
                params.scaleX = std::max(0.1f, std::min(params.scaleX, 2.0f));
                params.scaleY = std::max(0.1f, std::min(params.scaleY, 2.0f));
            }
            else if (op == 4) // COLOR_CORRECTION
            {
                std::cout << "  Brightness (-100 to 100): ";
                std::cin >> params.brightness;
                std::cout << "  Contrast (0.5 to 2.0): ";
                std::cin >> params.contrast;
            }
            else if (op == 5 || op == 6) // GAUSSIAN, MEDIAN
            {
                std::cout << "  Kernel Size (3, 5, 7, 9): ";
                int ks = 3;
                std::cin >> ks;
                params.kernelSize = (ks % 2 == 1) ? ks : ks + 1;
            }

            pipeline->add(opType, params);
        }

        std::cout << "\n[Pipeline Builder] Custom pipeline created: " << pipeline->describe() << "\n";
        break;
    }

    default:
        pipeline = std::make_shared<TransformPipeline>("grayscale");
        pipeline->add(TransformOperation::GRAYSCALE);
        break;
    }

    std::cout << "[Pipeline] Pipeline stages: " << pipeline->getStageCount() << "\n";
    std::cout << "[Pipeline] Description: " << pipeline->describe() << "\n";
    std::cout << "[Pipeline] Point operation only: " << (pipeline->isSimplePointOperation() ? "Yes" : "No") << "\n";
    std::cout << "[Pipeline] Requires overlap: " << (pipeline->requiresOverlapRegion() ? "Yes" : "No") << "\n";

    return pipeline;
}

int main()
{
    try
    {
        logMessage("[Main] Starting Gigapixel Processing Pipeline (Milestone 3 - Pipeline Composition)...");

        // =========================================================
        // MILESTONE 3: Pipeline Selection
        // =========================================================
        auto pipeline = showPipelineMenu();

        // =========================================================
        // Tile Reader Config
        // =========================================================

        logMessage("[Main] Initializing TileReader for 'input.tiff'...");
        TileReader reader("images/input.tiff");
        int overlap = pipeline->computeRequiredOverlap();

        logMessage("[Main] Computing optimal tile sizes based on device capabilities...");

        // Get device-specific tile sizes
        int cpuTileSize = Scheduler::getOptimalTileSize(DeviceType::CPU);
        int gpuTileSize = Scheduler::getOptimalTileSize(DeviceType::GPU);

        logMessage("[Main] Optimal tile sizes: CPU=" + std::to_string(cpuTileSize) +
                   "x" + std::to_string(cpuTileSize) + ", GPU=" + std::to_string(gpuTileSize) +
                   "x" + std::to_string(gpuTileSize));

        // Use CPU tile size for uniform processing (scheduler will adapt as needed)
        int tileSize = cpuTileSize;
        tileSize = std::min(tileSize, 512); // TODO: remove when using BIGTIFF

        int input_w = reader.getImageWidth();
        int input_h = reader.getImageHeight();
        int imgChannels = reader.getChannels();

        // =========================================================
        // PIPELINE CONFIGURATION: Calculate Output Dimensions
        // =========================================================

        int output_w = input_w;
        int output_h = input_h;

        // Calculate output dimensions based on pipeline
        for (size_t i = 0; i < pipeline->getStageCount(); ++i)
        {
            const auto& stage = pipeline->getStage(i);
            if (stage.operation == TransformOperation::ROTATE_90_CW)
            {
                std::swap(output_w, output_h);
            }
            else if (stage.operation == TransformOperation::RESIZE)
            {
                output_w = static_cast<int>(output_w * stage.params.scaleX);
                output_h = static_cast<int>(output_h * stage.params.scaleY);
            }
        }

        logMessage("[Main] Output dimensions: " + std::to_string(output_w) + "x" + std::to_string(output_h));

        // ---------------------------------------------------------
        // INITIALIZE YOUR WRITER
        // ---------------------------------------------------------
        logMessage("[Main] Initializing TiledOutputWriter...");
        TiledOutputWriter writer(
            "output.tiff",
            output_w,
            output_h,
            imgChannels,
            0,
            tileSize,
            false, true); // Use BigTIFF if set to True

        // Hardware concurrency setup
        unsigned int numCores = std::thread::hardware_concurrency();
        unsigned int numWorkers = std::max(1u, numCores - 2);

        logMessage("[Main] Hardware cores detected: " + std::to_string(numCores));
        logMessage("[Main] Reserving 1 core for Reader, 1 for Writer. Spawning " + std::to_string(numWorkers) + " Worker threads.");

        // =========================================================
        // MILESTONE 3: Memory Manager (VRAM Pool Initialization)
        // =========================================================
        logMessage("[Main] Milestone 3: Initializing VRAM Manager Pool...");

        // Calculate the maximum possible size for a tile in bytes
        // Using the larger GPU tile size to ensure buffers are big enough
        size_t maxTileBytes = gpuTileSize * gpuTileSize * imgChannels * sizeof(uint8_t);

        // Allocate 1 buffer per worker, plus 2 extra for smooth overlapping
        VramManager vramPool(numWorkers + 2, maxTileBytes);

        logMessage("[Main] Pre-allocated " + std::to_string(numWorkers + 2) + " GPU buffers.");

        // =========================================================
        // MILESTONE 2/4: Asynchronous GPU Transfers & Tile Optimization
        // =========================================================
        logMessage("[Main] Milestone 2 & 4: Async GPU transfers and tile optimization ENABLED");

        BoundedTileQueue inputQueue(static_cast<size_t>(numWorkers * 4));
        Scheduler scheduler;

        // ---------------------------------------------------------
        // Start the Writer Thread
        // ---------------------------------------------------------
        logMessage("[Main] Starting Dedicated Writer Thread...");
        std::thread writerThread([&writer]()
                                 { writer.run(); });

        // ---------------------------------------------------------
        // Start the Worker Threads
        // ---------------------------------------------------------
        std::vector<std::thread> workers;
        for (unsigned int i = 0; i < numWorkers; ++i)
        {
            // ADDED: Capture pipeline by value for worker threads
            workers.emplace_back([&inputQueue, &writer, &scheduler, &vramPool, pipeline, i]()
                                 {
                logMessage("[Worker " + std::to_string(i) + "] Started and waiting for tiles...");
                Tile tile;
                
                while (inputQueue.pop(tile)) {
                    logMessage("[Worker " + std::to_string(i) + "] Popped tile (" + std::to_string(tile.x) + "," + std::to_string(tile.y) + "). Requesting VRAM...");
                    
                    // Set pipeline for this tile
                    tile.setPipeline(pipeline);
                    
                    // 1. Acquire a pre-allocated GPU buffer from the pool (Blocks if none available)
                    uint8_t* d_buffer = vramPool.acquireBuffer();
                    
                    // 2. Dispatch to the Scheduler (We pass the VRAM buffer so the Scheduler doesn't have to allocate it)
                    scheduler.dispatch(tile, d_buffer); 
                    
                    // 3. Return the buffer to the pool immediately so another thread can use it
                    vramPool.releaseBuffer(d_buffer);
                    
                    logMessage("[Worker " + std::to_string(i) + "] Finished tile (" + std::to_string(tile.x) + "," + std::to_string(tile.y) + "). Submitting to Writer...");
                    writer.submit_tile(tile); 
                } 
                
                logMessage("[Worker " + std::to_string(i) + "] Queue empty & finished signal received. Shutting down."); });
        }

        
        // ---- Reader thread (prefetch) ----
        // Runs concurrently with workers — pushes tiles into the bounded queue
        // and blocks on not_full when workers are slow.  Calling setFinished()
        // at the end causes pop() to return false once the queue is drained.
        logMessage("[Main] Starting reader (prefetch) thread...");

        int nTilesX = (output_w + tileSize - 1) / tileSize;
        int nTilesY = (output_h + tileSize - 1) / tileSize;
        int totalTiles = nTilesX * nTilesY;
        int tilesRead = 0;

        // launch as a thread
        std::thread readerThread([&]() {
            for (int out_y = 0; out_y < output_h; out_y += tileSize){
                for (int out_x = 0; out_x < output_w; out_x += tileSize){
                    int tile_w = std::min(tileSize, output_w - out_x);
                    int tile_h = std::min(tileSize, output_h - out_y);

                    int in_x, in_y, read_w, read_h;

                // For pipeline: handle coordinate mapping based on first stage only (simplified)
                // For production: would need more sophisticated mapping
                if (pipeline->getStageCount() > 0 && 
                    pipeline->getStage(0).operation == TransformOperation::ROTATE_90_CW)
                {
                    in_x = out_y;
                    in_y = input_h - out_x - tile_w;
                    read_w = tile_h;
                    read_h = tile_w;
                }
                else
                {
                    in_x = out_x;
                    in_y = out_y;
                    read_w = tile_w;
                    read_h = tile_h;
                }

                    logMessage("[Reader] Tile " + std::to_string(tilesRead + 1) + "/" + std::to_string(totalTiles) +
                            " | Read (" + std::to_string(in_x) + "," + std::to_string(in_y) + ") -> Write (" +
                            std::to_string(out_x) + "," + std::to_string(out_y) + ")");

                    Tile loadedTile = reader.getTile(in_x, in_y, read_w, read_h, overlap);

                loadedTile.out_x = out_x;
                loadedTile.out_y = out_y;
                loadedTile.x = out_x;
                loadedTile.y = out_y;

                    inputQueue.push(std::move(loadedTile));
                    tilesRead++;
                }
            }

            // --- Graceful Shutdown Sequence ---
            logMessage("[Reader] All tiles read from disk. Sending 'Finished' signal to Input Queue.");
            inputQueue.setFinished();
        });

        //Joining: 
        //   1. readerThread  — wait for all tiles to be pushed
        //   2. workers       — wait for all tiles to be processed
        //   3. writer        — finalise + join after workers done

        readerThread.join();

        logMessage("[Main] Reader finished.");
        for (auto &worker : workers)
        {
            worker.join();
        }
        logMessage("[Main] All Worker threads have successfully joined (shutdown).");

        writer.finalize();
        logMessage("[Main] Joining Writer thread...");
        writerThread.join();

        logMessage("[Main] Pipeline composition complete. Image processed and saved to output.tiff");
        std::cout << "\n[Success] Processing completed!\n";
    }
    catch (const std::exception &e)
    {
        logMessage(std::string("[FATAL ERROR] ") + e.what());
        return 1;
    }

    return 0;
}