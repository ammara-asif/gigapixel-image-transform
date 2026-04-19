#include "TileOptimizer.h"
#include <algorithm>
#include <thread>
#include <cmath>

TileOptimizer::TileConfig TileOptimizer::computeOptimalConfig() {
    TileConfig config;

    // Query hardware capabilities
    int numCores = std::thread::hardware_concurrency();

    // CPU: Target L3 cache efficiency
    // Typical L3: 20-40 MB shared among cores
    // Per-tile budget: L3_CACHE_SIZE / numCores * 0.7 (70% utilization target)
    size_t cpuCachePerCore = (L3_CACHE_SIZE / std::max(1, numCores)) * 7 / 10;
    
    // For RGB image: 3 bytes/pixel, need ~2x memory (input + output)
    // Tile size = sqrt(cpuCachePerCore / (3 * 2))
    int cpuPixelsPerTile = cpuCachePerCore / (3 * 2);
    config.cpuTileSize = std::max(256, std::min(512, (int)std::sqrt(cpuPixelsPerTile)));
    
    // Align to power of 2
    if (config.cpuTileSize <= 256) config.cpuTileSize = 256;
    else if (config.cpuTileSize <= 512) config.cpuTileSize = 512;
    else config.cpuTileSize = 1024;

    // GPU: Optimize for warp occupancy
    // Modern GPUs benefit from larger tiles (512x512 to 1024x1024)
    // Reduces kernel launch overhead and improves occupancy
    config.gpuTileSize = DEFAULT_GPU_TILE_SIZE; // 1024x1024 default

    // Memory budgets
    config.maxCPUMemory = 64;  // 64 MB per CPU tile
    config.maxGPUMemory = 256; // 256 MB per GPU tile

    return config;
}

int TileOptimizer::getTileSizeForDevice(DeviceType device) {
    TileConfig config = computeOptimalConfig();
    
    if (device == DeviceType::CPU) {
        return config.cpuTileSize;
    } else {
        return config.gpuTileSize;
    }
}

size_t TileOptimizer::getMemoryBudgetForDevice(DeviceType device) {
    if (device == DeviceType::CPU) {
        return DEFAULT_CPU_MEMORY_BUDGET;
    } else {
        return DEFAULT_GPU_MEMORY_BUDGET;
    }
}
