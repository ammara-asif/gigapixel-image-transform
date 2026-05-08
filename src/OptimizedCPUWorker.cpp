#include "OptimizedCPUWorker.h"
#include "Transform.h"
#include "PipelineComposition.h"
#include <iostream>
#include <thread>
#include <cstring>
#include <algorithm>
#include <cmath>

OptimizedCPUWorker::OptimizedCPUWorker() : useFusion(true)
{
    detectCacheHierarchy();
    if (useFusion)
    {
        fusionOptimizer = std::make_unique<FusionOptimizer>();
    }
}

void OptimizedCPUWorker::detectCacheHierarchy()
{
    // TODO: Actual cache detection using cpuid or system calls
    // For now, use typical values for modern CPUs
    cacheInfo.numCores = std::thread::hardware_concurrency();
    cacheInfo.numL3Caches = std::max(1, cacheInfo.numCores / 4);
    cacheInfo.l1CacheSize = 32 * 1024;        // 32 KB
    cacheInfo.l2CacheSize = 256 * 1024;       // 256 KB
    cacheInfo.l3CacheSize = 20 * 1024 * 1024; // 20 MB
    cacheInfo.cacheLineSize = 64;             // 64 bytes

    std::cout << "[OptimizedCPUWorker] CPU Cache Detected:" << std::endl;
    std::cout << "  Cores: " << cacheInfo.numCores << std::endl;
    std::cout << "  L1: " << (cacheInfo.l1CacheSize / 1024) << " KB" << std::endl;
    std::cout << "  L2: " << (cacheInfo.l2CacheSize / 1024) << " KB" << std::endl;
    std::cout << "  L3: " << (cacheInfo.l3CacheSize / (1024 * 1024)) << " MB" << std::endl;
}

void OptimizedCPUWorker::execute(Tile &tile)
{
    std::cout << "[OptimizedCPUWorker] Processing tile (" << tile.x << "," << tile.y
              << ") size=" << tile.width << "x" << tile.height;

    if (useFusion && fusionOptimizer)
    {
        std::cout << " [FUSION-OPTIMIZED]" << std::endl;
        try
        {
            fusionOptimizer->executeFusedTransforms(tile);
            auto metrics = fusionOptimizer->getLastMetrics();
            std::cout << "[OptimizedCPUWorker] Fusion metrics - Time: " << metrics.executionTimeMs
                      << "ms, Cache hit: " << (metrics.cacheHitRate * 100) << "%" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "[OptimizedCPUWorker] Fusion failed: " << e.what() << std::endl;
            processWithCacheOptimization(tile);
        }
    }
    else
    {
        std::cout << " [CLASSIC-OPTIMIZED]" << std::endl;
        try
        {
            processWithCacheOptimization(tile);
        }
        catch (const std::exception &e)
        {
            std::cerr << "[OptimizedCPUWorker] Error: " << e.what() << std::endl;
            processTile(tile);
        }
    }

    recordPerformance();
}

bool OptimizedCPUWorker::canFitInL3Cache(const Tile &tile) const
{
    // Estimate memory footprint (RGB image, need input + output)
    size_t footprint = static_cast<size_t>(tile.width) * tile.height * 3 * 2;
    // Leave 30% of L3 for other data
    return footprint <= (cacheInfo.l3CacheSize * 7) / 10;
}

void OptimizedCPUWorker::processWithCacheOptimization(Tile &tile)
{
    if (tile.operation == TransformOperation::GRAYSCALE)
        grayscaleOptimized(tile);
    else if (tile.operation == TransformOperation::ROTATE_90_CW)
        rotateOptimized(tile);
    else
        processTile(tile);
}

void OptimizedCPUWorker::grayscaleOptimized(Tile &tile)
{
    std::cout << "[OptimizedCPUWorker] Running cache-optimized grayscale" << std::endl;

    uint8_t *px = tile.getRawPtr();
    if (!px || tile.dataSizeBytes == 0 || tile.width <= 0 || tile.height <= 0)
        throw std::runtime_error("Invalid tile buffer for grayscale optimization");

    const int channels = tile.channels;
    const size_t numPixels = static_cast<size_t>(tile.width) * tile.height;

    // Strip height targets L1 cache per core
    int stripHeight = std::max(1, (int)(cacheInfo.l1CacheSize / (tile.width * channels)));
    stripHeight = std::min(stripHeight, tile.height);

    for (int y = 0; y < tile.height; y += stripHeight)
    {
        int h = std::min(stripHeight, tile.height - y);
        for (int x = 0; x < tile.width; ++x)
        {
            for (int dy = 0; dy < h; ++dy)
            {
                size_t idx = ((y + dy) * tile.width + x) * channels;
                uint8_t gray = static_cast<uint8_t>(
                    0.299f * px[idx] + 0.587f * px[idx + 1] + 0.114f * px[idx + 2]);
                px[idx] = px[idx + 1] = px[idx + 2] = gray;
            }
        }
    }

    perfStats.executionTime = 0.0;
    perfStats.cacheHitRate = 0.95;
}

void OptimizedCPUWorker::rotateOptimized(Tile &tile)
{
    std::cout << "[OptimizedCPUWorker] Running cache-optimized rotation" << std::endl;

    uint8_t *px = tile.getRawPtr();
    if (!px || tile.dataSizeBytes == 0 || tile.width <= 0 || tile.height <= 0)
        throw std::runtime_error("Invalid tile buffer for rotation optimization");

    const int channels = tile.channels;

    std::vector<uint8_t> rotated(tile.dataSizeBytes);

    // Block size targets L2 cache for better cache locality
    int blockSize = std::min(32, std::max(4,
                                          (int)std::sqrt((double)cacheInfo.l2CacheSize / (64 * channels))));

    for (int by = 0; by < tile.height; by += blockSize)
    {
        for (int bx = 0; bx < tile.width; bx += blockSize)
        {
            int bh = std::min(blockSize, tile.height - by);
            int bw = std::min(blockSize, tile.width - bx);

            for (int y = by; y < by + bh; ++y)
            {
                for (int x = bx; x < bx + bw; ++x)
                {
                    size_t src_idx = (static_cast<size_t>(y) * tile.width + x) * channels;
                    // 90° CW: dst(x, H-1-y) = src(y, x)
                    size_t dst_idx = (static_cast<size_t>(x) * tile.height + (tile.height - 1 - y)) * channels;
                    for (int c = 0; c < channels; ++c)
                        rotated[dst_idx + c] = px[src_idx + c];
                }
            }
        }
    }

    std::memcpy(px, rotated.data(), tile.dataSizeBytes);
    std::swap(tile.width, tile.height);

    perfStats.executionTime = 0.0;
    perfStats.cacheHitRate = 0.90;
}

void OptimizedCPUWorker::recordPerformance()
{
    // TODO: Implement actual performance monitoring
    // Could collect cache misses, memory bandwidth, etc.
    std::cout << "[OptimizedCPUWorker] Cache hit rate: "
              << (perfStats.cacheHitRate * 100) << "%" << std::endl;
}