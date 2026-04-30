#pragma once

#include <cstddef>
#include "IWorker.h"

/**
 * TileOptimizer: Computes device-specific optimal tile sizes
 * 
 * CPU: Cache-optimized (typically 256x256-512x512)
 *      - Targets L3 cache size (~20-40 MB per core)
 *      - Balances register pressure and memory bandwidth
 * 
 * GPU: Occupancy-optimized (typically 512x512-1024x1024)
 *      - Targets warp occupancy and thread block efficiency
 *      - Larger tiles reduce kernel launch overhead
 */
class TileOptimizer {
public:
    struct TileConfig {
        int cpuTileSize;    // Optimal tile size for CPU processing
        int gpuTileSize;    // Optimal tile size for GPU processing
        int maxGPUMemory;   // Maximum GPU memory per tile (MB)
        int maxCPUMemory;   // Maximum CPU memory per tile (MB)
    };

    /**
     * Compute optimal tile configuration based on hardware
     */
    static TileConfig computeOptimalConfig();

    /**
     * Get tile size for a specific device
     */
    static int getTileSizeForDevice(DeviceType device);

    /**
     * Get memory budget for a device
     */
    static size_t getMemoryBudgetForDevice(DeviceType device);

private:
    // Default configurations (tunable)
    static constexpr int DEFAULT_CPU_TILE_SIZE = 512;      // 512x512 for CPU
    static constexpr int DEFAULT_GPU_TILE_SIZE = 1024;     // 1024x1024 for GPU
    static constexpr size_t DEFAULT_GPU_MEMORY_BUDGET = 256 * 1024 * 1024;  // 256 MB
    static constexpr size_t DEFAULT_CPU_MEMORY_BUDGET = 64 * 1024 * 1024;   // 64 MB

    // L3 cache sizes for common architectures
    static constexpr size_t L3_CACHE_SIZE = 20 * 1024 * 1024; // 20 MB conservative estimate
};
