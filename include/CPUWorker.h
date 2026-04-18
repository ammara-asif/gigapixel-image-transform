#pragma once
#include <iostream>
#include "Tile.h"
#include "IWorker.h"
#include "Transform.h"

class CPUWorker : public IWorker {
public:
    void execute(Tile& tile) override {
        std::cout<<"[CPUWorker] Would run CPU on tile (" <<
        std::to_string(tile.x) << "," << std::to_string(tile.y) <<
         ") size=" << std::to_string(tile.width * tile.height)<<std::endl;
        
        processTile(tile);
    }
};