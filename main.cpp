#include <iostream>
#include "TileReader.h"
#include "OutputWriter.h"

int main(){
    try {
        TileReader reader("Philips-3.tiff");
        int overlap=2;
        
        int tile_size= reader.computeTileSize(TILE_MEMORY_BUDGET, overlap);
        auto grid= reader.getTileGrid(tile_size);
        std::cout << "Image:     " << reader.getImageWidth() << " x " << reader.getImageHeight() << "\n";
        std::cout << "Tile size: " << tile_size << "px logical (" << tile_size + 2*overlap << "px buffered)\n";
        std::cout << "Tiles:     " << grid.size() << " total\n\n";

        TiledOutputWriter writer("output.tiff",
                         reader.getImageWidth(),
                         reader.getImageHeight(),
                         channels,
                         (int)grid.size(),
                         /*big_tiff=*/true);
 
        #pragma omp parallel num_threads(2){
            if (omp_get_thread_num() == 0) {
                for (auto& idx : grid) {
                    Tile t = reader.getTile(idx.x, idx.y, idx.width, idx.height, overlap);
                    //...
                    writer.submit_tile(t);
                    std::cout << "  tile (" << idx.col << "," << idx.row << ")"
                              << "  origin=(" << idx.x << "," << idx.y << ")"
                              << "  buffer=" << t.width << "x" << t.height << "\n";
                }
                writer.finalize();   // signals writer thread to stop
            } else {
                writer.run();        // writer thread — drains queue, writes to TIFF
            }
        }
    }
    catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}