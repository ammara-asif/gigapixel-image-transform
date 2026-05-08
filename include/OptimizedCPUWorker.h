#pragma once

#include <thread>
#include <memory>
#include "Tile.h"
#include "IWorker.h"
#include "TileOptimizer.h"
#include "FusionOptimizer.h"

/**
 * OptimizedCPUWorker: CPU worker optimized for cache efficiency
 *
 * Features:
 * - Cache-line aware processing
 * - Tile size optimization for L3 cache
 * - SIMD vectorization hints
 * - Fusion-based optimization with blocking and SIMD-friendly layouts
 * - NUMA awareness (if available)
 */
class OptimizedCPUWorker : public IWorker
{
public:
    OptimizedCPUWorker();
    ~OptimizedCPUWorker() = default;

    /**
     * Execute tile on CPU with cache optimization
     */
    void execute(Tile &tile) override;

    /**
     * Get CPU cache information
     */
    struct CPUCacheInfo
    {
        int numCores;
        int numL3Caches;
        size_t l1CacheSize;
        size_t l2CacheSize;
        size_t l3CacheSize;
        int cacheLineSize;
    };

    CPUCacheInfo getCacheInfo() const { return cacheInfo; }

    /**
     * Enable/disable fusion-based optimization
     */
    void setUseFusion(bool enable) { useFusion = enable; }

    /**
     * Get fusion optimization status
     */
    bool isUsingFusion() const { return useFusion; }

private:
    CPUCacheInfo cacheInfo;
    std::unique_ptr<FusionOptimizer> fusionOptimizer;
    bool useFusion;

    /**
     * Initialize and detect CPU cache characteristics
     */
    void detectCacheHierarchy();

    /**
     * Process tile with cache-aware algorithms
     */
    void processWithCacheOptimization(Tile &tile);

    /**
     * Cache-optimized grayscale conversion
     * Processes in strips to fit L3 cache
     */
    void grayscaleOptimized(Tile &tile);

    /**
     * Cache-optimized rotation
     * Uses block-wise transpose for better cache utilization
     */
    void rotateOptimized(Tile &tile);

    /**
     * Analyze tile access pattern for cache efficiency
     */
    bool canFitInL3Cache(const Tile &tile) const;

    // Performance monitoring
    struct PerfStats
    {
        double executionTime;
        double cacheHitRate;
        int l3CacheMisses;
    } perfStats;

    /**
     * Record performance metrics for this execution
     */
    void recordPerformance();
};
