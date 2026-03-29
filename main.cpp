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

        logMessage("[Main] Initializing TileReader for 'input.tiff'...");
        TileReader reader("input.tiff");
        int overlap = 0;

        logMessage("[Main] Computing optimal tile size based on memory budget...");
        int tileSize = reader.computeTileSize(TILE_MEMORY_BUDGET, overlap);
        tileSize = std::min(tileSize, 512);

        std::vector<TileIndex> grid = reader.getTileGrid(tileSize);
        logMessage("[Main] Grid calculated. Total tiles to process: " + std::to_string(grid.size()));

        // ---------------------------------------------------------
        // INITIALIZE YOUR WRITER
        // ---------------------------------------------------------
        int imgChannels = reader.getChannels();

        logMessage("[Main] Initializing TiledOutputWriter...");
        TiledOutputWriter writer("output.tiff",
                                 reader.getImageWidth(),
                                 reader.getImageHeight(),
                                 imgChannels,
                                 grid.size(),
                                 tileSize,
                                 // TODO: set to true for bigtiff
                                 false); // Use BigTIFF if set to True

        // Hardware concurrency setup
        unsigned int numCores = std::thread::hardware_concurrency();
        unsigned int numWorkers = std::max(1u, numCores - 2);

        logMessage("[Main] Hardware cores detected: " + std::to_string(numCores));
        logMessage("[Main] Reserving 1 core for Reader, 1 for Writer. Spawning " + std::to_string(numWorkers) + " Worker threads.");

        // We only need ONE queue now (Disk -> Workers)
        BoundedTileQueue inputQueue(numWorkers * 2);

        // ---------------------------------------------------------
        // STAGE 3: Start the Writer Thread
        // ---------------------------------------------------------
        logMessage("[Main] Starting Dedicated Writer Thread...");
        std::thread writerThread([&writer]()
                                 { writer.run(); });

        // ---------------------------------------------------------
        // STAGE 2: Start the Worker Threads
        // ---------------------------------------------------------
        std::vector<std::thread> workers;
        for (unsigned int i = 0; i < numWorkers; ++i)
        {
            workers.emplace_back([&inputQueue, &writer, i]()
                                 {
                logMessage("[Worker " + std::to_string(i) + "] Started and waiting for tiles...");
                Tile tile;
                
                while (inputQueue.pop(tile)) {
                    logMessage("[Worker " + std::to_string(i) + "] Popped tile (" + std::to_string(tile.x) + "," + std::to_string(tile.y) + "). Processing...");
                    
                    processTile(tile); // Do the heavy math
                    
                    logMessage("[Worker " + std::to_string(i) + "] Finished tile (" + std::to_string(tile.x) + "," + std::to_string(tile.y) + "). Submitting to Writer...");
                    writer.submit_tile(tile); 
                } 
                
                logMessage("[Worker " + std::to_string(i) + "] Queue empty & finished signal received. Shutting down."); });
        }

        // ---------------------------------------------------------
        // STAGE 1: The Reader (Main Thread)
        // ---------------------------------------------------------
        logMessage("[Main/Reader] Pipeline fully active. Beginning disk reads...");
        int tilesRead = 0;
        for (const auto &idx : grid)
        {
            logMessage("[Main/Reader] Reading tile " + std::to_string(tilesRead + 1) + "/" + std::to_string(grid.size()) + " at (" + std::to_string(idx.x) + "," + std::to_string(idx.y) + ")...");

            Tile loadedTile = reader.getTile(idx.x, idx.y, idx.width, idx.height, overlap);
            inputQueue.push(std::move(loadedTile));

            tilesRead++;
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