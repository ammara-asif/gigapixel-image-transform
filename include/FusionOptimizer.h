#pragma once

#include "Tile.h"
#include <cstddef>
#include <memory>
#include <array>

class FusionOptimizer
{
public:
    FusionOptimizer();
    ~FusionOptimizer() = default;

    /**
     * Execute fused operations with cache-optimized blocking
     * Combines multiple transforms in a single pass to reduce memory traffic
     */
    void executeFusedTransforms(Tile &tile);

    struct BlockingStrategy
    {
        int blockWidth;
        int blockHeight;
        int stripHeight;
        size_t targetCacheSize;
    };

    /**
     * Compute optimal blocking parameters based on cache hierarchy and tile properties
     */
    BlockingStrategy computeBlockingStrategy(int tileWidth, int tileHeight,
                                             int channels, size_t cacheSize) const;

    /**
     * SIMD-friendly grayscale with blocking
     * Uses row-major blocking with proper alignment for vectorization
     */
    void grayscaleFused(Tile &tile);

    /**
     * SIMD-friendly rotation with blocking
     * Processes in cache-friendly blocks to minimize memory traffic
     */
    void rotateFused(Tile &tile);

    /**
     * Fused grayscale + rotation in single pass
     * Applies both operations without intermediate memory writes
     */
    void grayscaleRotateFused(Tile &tile);

    struct PerformanceMetrics
    {
        double executionTimeMs;
        double cacheHitRate;
        double memoryBandwidthMBps;
        int l3CacheMisses;
        int memoryPasses;
    };

    PerformanceMetrics getLastMetrics() const { return lastMetrics; }

private:
    struct CPUCacheInfo
    {
        int numCores;
        size_t l1CacheSize;
        size_t l2CacheSize;
        size_t l3CacheSize;
        int cacheLineSize;
    };

    CPUCacheInfo cacheInfo;
    PerformanceMetrics lastMetrics;

    void detectCacheHierarchy();

    /**
     * SIMD-aligned memory buffer for intermediate results
     * Allocated on 64-byte boundaries for cache-line efficiency
     */
    struct AlignedBuffer
    {
        std::unique_ptr<uint8_t[], decltype(&free)> data;
        size_t size;
        size_t alignment;

        AlignedBuffer(size_t sz, size_t align = 64);
        AlignedBuffer(const AlignedBuffer &) = delete;
        AlignedBuffer &operator=(const AlignedBuffer &) = delete;
    };

    /**
     * Helper: Process grayscale in cache-friendly row strips
     */
    void processGrayscaleStrips(uint8_t *px, int width, int height,
                                int channels, int stripHeight);

    /**
     * Helper: Process rotation in blocking regions (L-shaped transpose blocks)
     */
    void processRotationBlocks(uint8_t *px, uint8_t *dst, int width, int height,
                               int channels, int blockSize);

    /**
     * Helper: Fused kernel combining grayscale + rotation
     */
    void processFusedGrayscaleRotation(uint8_t *px, uint8_t *dst, int width, int height,
                                       int channels, int blockSize);

    /**
     * Helper: Calculate optimal block size for rotation based on cache
     */
    int calculateOptimalBlockSize(int width, int height, int channels) const;

    /**
     * Helper: Calculate optimal strip height for point operations
     */
    int calculateOptimalStripHeight(int width, int channels) const;
};
