#pragma once 
#include <iostream>
#include <string>
#include "Tile.h"
#include "IWorker.h"
#include "Transform.h"
#include "CUDAKernels.h"

class GPUWorker : public IWorker {
public:
    void execute(Tile& tile) override {
       std::cout << "[GPUWorker] Attempting GPU kernel on tile ("
                 << std::to_string(tile.x) << "," << std::to_string(tile.y)
                 << ") size=" << std::to_string(tile.width * tile.height) << std::endl;

       const bool gpuDone = processTileCuda(tile);
       if (!gpuDone) {
           std::cout << "[GPUWorker] CUDA unavailable/unsupported for this tile, using CPU fallback." << std::endl;
           processTile(tile);
       }
    }
};