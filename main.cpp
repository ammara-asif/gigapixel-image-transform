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
        TileReader reader("images/input.tiff");
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

        // std::vector<TileIndex> grid = reader.getTileGrid(tileSize);
        // logMessage("[Main] Grid calculated. Total tiles to process: " + std::to_string(grid.size()));

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
            // TODO: set to true for bigtiff
            false); // Use BigTIFF if set to True

        // Hardware concurrency setup
        unsigned int numCores = std::thread::hardware_concurrency();
        unsigned int numWorkers = std::max(1u, numCores - 2);

        logMessage("[Main] Hardware cores detected: " + std::to_string(numCores));
        logMessage("[Main] Reserving 1 core for Reader, 1 for Writer. Spawning " + std::to_string(numWorkers) + " Worker threads.");

        // =========================================================
        // MILESTONE 2: Asynchronous GPU Transfers & Tile Optimization
        // =========================================================
        // The Scheduler now handles:
        // 1. Asynchronous CUDA stream management for overlapping H2D, compute, D2H
        // 2. Device-specific tile size optimization:
        //    - CPU: Cache-optimized (512x512) for L3 cache efficiency
        //    - GPU: Occupancy-optimized (1024x1024) for warp utilization
        // 3. Intelligent device selection based on:
        //    - Tile size and operation type
        //    - Current device queue depths
        //    - GPU availability and CUDA stream capacity
        logMessage("[Main] Milestone 2: Async GPU transfers and tile optimization ENABLED");

        // We only need ONE queue now (Disk -> Workers)
        BoundedTileQueue inputQueue(static_cast<size_t>(numWorkers * 2));
        
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
            workers.emplace_back([&inputQueue, &writer, &scheduler, i]()
                                 {
                logMessage("[Worker " + std::to_string(i) + "] Started and waiting for tiles...");
                Tile tile;
                
                while (inputQueue.pop(tile)) {
                    logMessage("[Worker " + std::to_string(i) + "] Popped tile (" + std::to_string(tile.x) + "," + std::to_string(tile.y) + "). Processing...");
                    
                    //processTile(tile); // Do the heavy math 
                    
                    scheduler.dispatch(tile);
                    
                    logMessage("[Worker " + std::to_string(i) + "] Finished tile (" + std::to_string(tile.x) + "," + std::to_string(tile.y) + "). Submitting to Writer...");
                    writer.submit_tile(tile); 
                } 
                
                logMessage("[Worker " + std::to_string(i) + "] Queue empty & finished signal received. Shutting down."); });
        }

        // ---------------------------------------------------------
        // The Reader (Main Thread)
        // ---------------------------------------------------------
        logMessage("[Main/Reader] Pipeline fully active. Beginning disk reads...");

        // Calculate total tiles so the logger knows the progress
        int nTilesX = (output_w + tileSize - 1) / tileSize;
        int nTilesY = (output_h + tileSize - 1) / tileSize;
        int totalTiles = nTilesX * nTilesY;

        int tilesRead = 0;

        // Loop over the OUTPUT dimensions
        for (int out_y = 0; out_y < output_h; out_y += tileSize)
        {
            for (int out_x = 0; out_x < output_w; out_x += tileSize)
            {

                // Calculate actual size of this output tile (handling edge clamps)
                int tile_w = std::min(tileSize, output_w - out_x);
                int tile_h = std::min(tileSize, output_h - out_y);

                // --- DYNAMIC MAPPING LOGIC ---
                int in_x, in_y, read_w, read_h;

                if (currentOp == TransformOperation::ROTATE_90_CW)
                {
                    // Inverse Mapping for Rotation
                    in_x = out_y;
                    in_y = input_h - out_x - tile_w;
                    read_w = tile_h;
                    read_h = tile_w;
                }
                else
                {
                    // Forward Mapping for Grayscale / Point Operations
                    in_x = out_x;
                    in_y = out_y;
                    read_w = tile_w;
                    read_h = tile_h;
                }

                logMessage("[Main/Reader] Tile " + std::to_string(tilesRead + 1) + "/" + std::to_string(totalTiles) +
                           " | Read (" + std::to_string(in_x) + "," + std::to_string(in_y) + ") -> Write (" +
                           std::to_string(out_x) + "," + std::to_string(out_y) + ")");

                Tile loadedTile = reader.getTile(in_x, in_y, read_w, read_h, overlap);

                loadedTile.out_x = out_x;
                loadedTile.out_y = out_y;
                loadedTile.operation = currentOp; // Pass the flag to the worker

                loadedTile.x = out_x;
                loadedTile.y = out_y;

                inputQueue.push(std::move(loadedTile));
                tilesRead++;
            }
        }

        // --- Graceful Shutdown Sequence ---
        logMessage("[Main] All tiles read from disk. Sending 'Finished' signal to Input Queue.");
        inputQueue.setFinished(); // 1. Tell workers no more input is coming

        logMessage("[Main] Waiting for Worker threads to clear the remaining queue...");
        for (auto &worker : workers)
        {
            worker.join(); // 2. Wait for workers to finish all processing
        }
        logMessage("[Main] All Worker threads have successfully joined (shutdown).");

        logMessage("[Main] Finalizing Output Writer (pushing sentinel and waiting)...");
        // 3. Tell your writer to push the nullptr sentinel and spin-wait for done_
        writer.finalize();

        logMessage("[Main] Joining Writer thread...");
        // 4. Join the writer thread to ensure clean OS cleanup
        writerThread.join();

        logMessage("[Main] Pipeline integration complete. Image processed and saved.");
    }
    catch (const std::exception &e)
    {
        logMessage(std::string("[FATAL ERROR] ") + e.what());
        return 1;
    }

    return 0;
}