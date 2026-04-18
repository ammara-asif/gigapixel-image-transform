#pragma once 
#include <iostream>
#include <string>
#include "Tile.h"
#include "IWorker.h"
#include "Transform.h"

class GPUWorker : public IWorker {
public:
    void execute(Tile& tile) override {

       std::cout<<"[GPUWorker] Would run GPU kernel on tile (" <<
        std::to_string(tile.x) << "," << std::to_string(tile.y) <<
         ") size=" << std::to_string(tile.width * tile.height)<<std::endl;
        
        // Temporarily fall back to CPU for testing
        processTile(tile);
    }
};