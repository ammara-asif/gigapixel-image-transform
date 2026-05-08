#include "FusionWorker.h"
#include <iostream>

FusionWorker::FusionWorker() : fusionEnabled(true)
{
    optimizer = std::make_unique<FusionOptimizer>();
}

void FusionWorker::execute(Tile &tile)
{
    if (!tile.getRawPtr() || tile.dataSizeBytes == 0)
    {
        throw std::runtime_error("FusionWorker: tile has no data");
    }

    std::cout << "[FusionWorker] Executing tile (" << tile.x << "," << tile.y
              << ") with fusion optimization" << std::endl;

    optimizer->executeFusedTransforms(tile);
}

FusionOptimizer::PerformanceMetrics FusionWorker::getMetrics() const
{
    return optimizer->getLastMetrics();
}

bool FusionWorker::shouldUseFusion(const Tile &tile) const
{
    size_t tilePixels = static_cast<size_t>(tile.width) * tile.height;
    return fusionEnabled && tilePixels > 4096;
}
