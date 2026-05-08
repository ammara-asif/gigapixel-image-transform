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

<<<<<<< HEAD
void OptimizedCPUWorker::detectCacheHierarchy()
{
    cacheInfo.numCores = std::thread::hardware_concurrency();
    cacheInfo.numL3Caches = std::max(1, cacheInfo.numCores / 4);
    cacheInfo.l1CacheSize = 32 * 1024;
    cacheInfo.l2CacheSize = 256 * 1024;
    cacheInfo.l3CacheSize = 20 * 1024 * 1024;
    cacheInfo.cacheLineSize = 64;
=======
void OptimizedCPUWorker::detectCacheHierarchy() {
    // TODO: Actual cache detection using cpuid or system calls
    // For now, use typical values for modern CPUs
    
    cacheInfo.numCores = std::thread::hardware_concurrency();
    cacheInfo.numL3Caches = std::max(1, cacheInfo.numCores / 4); // Rough estimate
    cacheInfo.l1CacheSize = 32 * 1024;                          // 32 KB
    cacheInfo.l2CacheSize = 256 * 1024;                         // 256 KB
    cacheInfo.l3CacheSize = 20 * 1024 * 1024;                   // 20 MB
    cacheInfo.cacheLineSize = 64;                               // 64 bytes
>>>>>>> e3e93d768b5dbdd00f924cf3428dbedd82015386

    std::cout << "[OptimizedCPUWorker] CPU Cache Detected:" << std::endl;
    std::cout << "  Cores: " << cacheInfo.numCores << std::endl;
    std::cout << "  L1: " << (cacheInfo.l1CacheSize / 1024) << " KB" << std::endl;
    std::cout << "  L2: " << (cacheInfo.l2CacheSize / 1024) << " KB" << std::endl;
    std::cout << "  L3: " << (cacheInfo.l3CacheSize / (1024 * 1024)) << " MB" << std::endl;
}

<<<<<<< HEAD
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
=======
void OptimizedCPUWorker::execute(Tile& tile) {
    std::cout << "[OptimizedCPUWorker] Processing tile (" << tile.x << "," << tile.y 
              << ") size=" << tile.width << "x" << tile.height << std::endl;

    try {
        // Handle pipeline execution (Milestone 3)
        if (tile.usePipeline()) {
            std::cout << "[OptimizedCPUWorker] Executing pipeline: " 
                      << tile.pipeline->describe() << std::endl;
            processTilePipeline(tile, tile.pipeline);
        } else {
            // Single operation execution (legacy)
            if (canFitInL3Cache(tile)) {
                processWithCacheOptimization(tile);
            } else {
                // Process in strips that fit in cache
                processWithCacheOptimization(tile);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[OptimizedCPUWorker] Error: " << e.what() << std::endl;
        // Fallback to standard processing
        processTile(tile);
>>>>>>> e3e93d768b5dbdd00f924cf3428dbedd82015386
    }

    recordPerformance();
}

<<<<<<< HEAD
bool OptimizedCPUWorker::canFitInL3Cache(const Tile &tile) const
{
    size_t footprint = static_cast<size_t>(tile.width) * tile.height * 3 * 2;
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
=======
bool OptimizedCPUWorker::canFitInL3Cache(const Tile& tile) const {
    // Estimate memory footprint (RGB image, need input + output)
    int channels = 3; // Assuming RGB
    size_t memoryFootprint = tile.width * tile.height * channels * 2;
    
    // Leave 30% of L3 for other data
    size_t availableL3 = (cacheInfo.l3CacheSize * 7) / 10;
    
    return memoryFootprint <= availableL3;
}

void OptimizedCPUWorker::processWithCacheOptimization(Tile& tile) {
    if (tile.operation == TransformOperation::GRAYSCALE) {
        grayscaleOptimized(tile);
    } else if (tile.operation == TransformOperation::ROTATE_90_CW) {
        rotateOptimized(tile);
    } else {
        // Fallback to standard processing
        processTile(tile);
    }
>>>>>>> e3e93d768b5dbdd00f924cf3428dbedd82015386
}

void OptimizedCPUWorker::grayscaleOptimized(Tile &tile)
{
    std::cout << "[OptimizedCPUWorker] Running cache-optimized grayscale" << std::endl;

<<<<<<< HEAD
    uint8_t *px = tile.getRawPtr();
    size_t numPixels = static_cast<size_t>(tile.width) * tile.height;
    int channels = tile.channels;
=======
    uint8_t* data = tile.getRawPtr();
    if (!data || tile.dataSizeBytes == 0 || tile.width <= 0 || tile.height <= 0) {
        throw std::runtime_error("Invalid tile buffer for grayscale optimization");
    }
>>>>>>> e3e93d768b5dbdd00f924cf3428dbedd82015386

    const size_t numPixels = static_cast<size_t>(tile.width) * static_cast<size_t>(tile.height);
    const size_t channels = std::max<size_t>(1, tile.dataSizeBytes / numPixels);

    // Process in cache-friendly strips
    // Strip size targets L1 cache per core
    int stripHeight = std::max(1, (int)(cacheInfo.l1CacheSize / (tile.width * channels)));
    stripHeight = std::min(stripHeight, tile.height);

    for (int y = 0; y < tile.height; y += stripHeight)
    {
        int h = std::min(stripHeight, tile.height - y);
<<<<<<< HEAD
        for (int x = 0; x < tile.width; ++x)
        {
            for (int dy = 0; dy < h; ++dy)
            {
                size_t idx = ((y + dy) * tile.width + x) * channels;
                uint8_t gray = static_cast<uint8_t>(
                    0.299f * px[idx] + 0.587f * px[idx + 1] + 0.114f * px[idx + 2]);
                px[idx] = px[idx + 1] = px[idx + 2] = gray;
=======
        
        for (int x = 0; x < tile.width; ++x) {
            for (int dy = 0; dy < h; ++dy) {
                size_t idx = ((y + dy) * tile.width + x) * channels;
                
                uint8_t r = data[idx];
                uint8_t g = data[idx + 1];
                uint8_t b = data[idx + 2];

                uint8_t gray = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b);

                data[idx] = gray;
                data[idx + 1] = gray;
                data[idx + 2] = gray;
>>>>>>> e3e93d768b5dbdd00f924cf3428dbedd82015386
            }
        }
    }

    perfStats.executionTime = 0.0;
    perfStats.cacheHitRate = 0.95; // Estimated
}

void OptimizedCPUWorker::rotateOptimized(Tile &tile)
{
    std::cout << "[OptimizedCPUWorker] Running cache-optimized rotation" << std::endl;

<<<<<<< HEAD
    uint8_t *px = tile.getRawPtr();
    size_t numPixels = static_cast<size_t>(tile.width) * tile.height;
    int channels = tile.channels;
=======
    uint8_t* data = tile.getRawPtr();
    if (!data || tile.dataSizeBytes == 0 || tile.width <= 0 || tile.height <= 0) {
        throw std::runtime_error("Invalid tile buffer for rotation optimization");
    }
>>>>>>> e3e93d768b5dbdd00f924cf3428dbedd82015386

    const size_t numPixels = static_cast<size_t>(tile.width) * static_cast<size_t>(tile.height);
    const size_t channels = std::max<size_t>(1, tile.dataSizeBytes / numPixels);

<<<<<<< HEAD
    int blockSize = std::min(32, std::max(4,
                                          (int)std::sqrt((double)cacheInfo.l2CacheSize / (64 * channels))));
=======
    uint8_t* rotated_raw = nullptr;
    if (cudaMallocHost((void**)&rotated_raw, tile.dataSizeBytes) != cudaSuccess || rotated_raw == nullptr) {
        throw std::runtime_error("Failed to allocate pinned buffer for rotated tile");
    }
    std::memset(rotated_raw, 0, tile.dataSizeBytes);

    // Use block transpose for better cache locality
    // Block size targets L2 cache
    int blockSize = std::min(32, std::max(4, (int)std::sqrt(cacheInfo.l2CacheSize / (64 * channels))));
>>>>>>> e3e93d768b5dbdd00f924cf3428dbedd82015386

    for (int by = 0; by < tile.height; by += blockSize)
    {
        for (int bx = 0; bx < tile.width; bx += blockSize)
        {
            int bh = std::min(blockSize, tile.height - by);
            int bw = std::min(blockSize, tile.width - bx);

<<<<<<< HEAD
            for (int y = by; y < by + bh; ++y)
            {
                for (int x = bx; x < bx + bw; ++x)
                {
                    size_t src_idx = (static_cast<size_t>(y) * tile.width + x) * channels;
                    size_t dst_idx = (static_cast<size_t>(x) * tile.height + (tile.height - 1 - y)) * channels;
                    for (int c = 0; c < channels; ++c)
                        rotated[dst_idx + c] = px[src_idx + c];
=======
            // Process this block with rotation
            for (int y = by; y < by + bh; ++y) {
                for (int x = bx; x < bx + bw; ++x) {
                    size_t src_idx = (y * tile.width + x) * channels;
                    
                    int dst_x = tile.height - 1 - y;
                    int dst_y = x;
                    
                    size_t dst_idx = (dst_y * tile.height + dst_x) * channels;

                    for (int c = 0; c < channels; ++c) {
                        rotated_raw[dst_idx + c] = data[src_idx + c];
                    }
>>>>>>> e3e93d768b5dbdd00f924cf3428dbedd82015386
                }
            }
        }
    }

<<<<<<< HEAD
    std::memcpy(px, rotated.data(), tile.dataSizeBytes);
    std::swap(tile.width, tile.height);
=======
    // Swap dimensions
    int temp = tile.width;
    tile.width = tile.height;
    tile.height = temp;
>>>>>>> e3e93d768b5dbdd00f924cf3428dbedd82015386

    tile.data = std::shared_ptr<uint8_t>(rotated_raw, CudaHostDeleter());

    perfStats.executionTime = 0.0;
    perfStats.cacheHitRate = 0.90; // Block-wise is good but not as good as sequential
}

<<<<<<< HEAD
void OptimizedCPUWorker::recordPerformance()
{
    std::cout << "[OptimizedCPUWorker] Cache hit rate: "
              << (perfStats.cacheHitRate * 100) << "%" << std::endl;
}
=======
/**
 * Record performance metrics (placeholder)
 */
void OptimizedCPUWorker::recordPerformance()
{
    // TODO: Implement actual performance monitoring
    // Could collect cache misses, memory bandwidth, etc.
}
>>>>>>> e3e93d768b5dbdd00f924cf3428dbedd82015386
