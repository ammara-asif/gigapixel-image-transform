#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <string>
#include <mutex>

#include "TileReader.h"
#include "Tile.h"
#include "BoundedTileQueue.h"
#include "OutputWriter.h"
#include "Transform.h"
#include "Scheduler.h"
#include "TileOptimizer.h"
#include "VramManager.h"

// --- Thread-Safe Console Logging ---
std::mutex printMutex;
void logMessage(const std::string &msg)
{
    std::lock_guard<std::mutex> lock(printMutex);
    std::cout << msg << std::endl;
}

int main()
{
    try
    {
        logMessage("[Main] Starting Gigapixel Processing Pipeline...");

        // =========================================================
        // USER INPUT: Select the Transformation
        // =========================================================

        int userChoice = 0;
        std::cout << "======================================\n";
        std::cout << " Gigapixel Image Transform Pipeline\n";
        std::cout << "======================================\n";
        std::cout << "1. Grayscale (Point Operation)\n";
        std::cout << "2. Rotate 90 CW (Geometric Inverse Map)\n";
        std::cout << "Select an operation (1 or 2): ";

        while (!(std::cin >> userChoice) || (userChoice < 1 || userChoice > 2))
        {
            std::cout << "Invalid input. Please enter 1 or 2: ";
            std::cin.clear();             // clear error flags
            std::cin.ignore(10000, '\n'); // discard bad input
        }

        TransformOperation currentOp;
        if (userChoice == 2)
        {
            currentOp = TransformOperation::ROTATE_90_CW;
            logMessage("\n[Main] User selected: ROTATE 90 CW");
        }
        else
        {
            currentOp = TransformOperation::GRAYSCALE;
            logMessage("\n[Main] User selected: GRAYSCALE");
        }

        // =========================================================
        // Tile Reader Config
        // =========================================================

        logMessage("[Main] Initializing TileReader for 'input.tiff'...");
        TileReader reader("input.tiff");
        int overlap = 0;

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
        // PIPELINE CONFIGURATION: Change this flag to swap algorithms!
        // =========================================================

        // Calculate Output Dimensions dynamically
        int output_w = input_w;
        int output_h = input_h;

        if (currentOp == TransformOperation::ROTATE_90_CW)
        {
            output_w = input_h; // Swap dimensions for 90 CW
            output_h = input_w;
        }

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
        // Memory Manager (VRAM Pool Initialization)
        // =========================================================
        logMessage("[Main] Initializing VRAM Manager Pool...");

        // Calculate the maximum possible size for a tile in bytes
        // Using the larger GPU tile size to ensure buffers are big enough
        size_t maxTileBytes = gpuTileSize * gpuTileSize * (imgChannels+1) * sizeof(uint8_t);

        // Allocate 1 buffer per worker, plus 2 extra for smooth overlapping
        VramManager vramPool(numWorkers + 2, maxTileBytes);

        logMessage("[Main] Pre-allocated " + std::to_string(numWorkers + 2) + " GPU buffers.");

        // =========================================================
        // Asynchronous GPU Transfers & Tile Optimization
        // =========================================================
        logMessage("[Main] Async GPU transfers and tile optimization ENABLED");

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
            // ADDED: Capture vramPool by reference
            workers.emplace_back([&inputQueue, &writer, &scheduler, &vramPool, i]()
                                 {
                logMessage("[Worker " + std::to_string(i) + "] Started and waiting for tiles...");
                Tile tile;
                
                while (inputQueue.pop(tile)) {
                    logMessage("[Worker " + std::to_string(i) + "] Popped tile (" + std::to_string(tile.x) + "," + std::to_string(tile.y) + "). Requesting VRAM...");
                    
                    // 1. Acquire a pre-allocated GPU buffer from the pool (Blocks if none available)
                    uint8_t* d_buffer = vramPool.acquireBuffer();
                    
                    // 2. Dispatch to the Scheduler (We pass the VRAM buffer so the Scheduler doesn't have to allocate it)
                    // NOTE to teammate doing Milestone 4: Update `scheduler.dispatch` to accept (Tile&, uint8_t*)
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

                    if (currentOp == TransformOperation::ROTATE_90_CW)
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
                    loadedTile.operation = currentOp;
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

        logMessage("[Main] Pipeline complete. Output saved to output.tiff");
    }
    catch (const std::exception &e)
    {
        logMessage(std::string("[FATAL ERROR] ") + e.what());
        return 1;
    }

    return 0;
}