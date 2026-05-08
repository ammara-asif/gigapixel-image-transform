#pragma once
#include <atomic>
#include <memory>
#include <iostream> // Added for std::cerr
#include "OptimizedCPUWorker.h"
#include "AsyncGPUWorker.h"
#include "TileOptimizer.h"
#include "PipelineComposition.h" // For pipeline-aware scheduling

/**
 * Scheduler: Intelligent task distribution with device-specific optimizations
 * * Features:
 * - Device selection based on tile characteristics
 * - GPU queue depth monitoring with async transfers
 * - Tile size optimization per device
 * - Performance-aware load balancing
 */
class Scheduler
{
private:
    std::unique_ptr<OptimizedCPUWorker> cpu;
    std::unique_ptr<AsyncGPUWorker> gpu;

    std::atomic<int> gpuQueueDepth{0};
    std::atomic<int> cpuQueueDepth{0};

    static constexpr int GPU_OVERLOAD_THRESHOLD = 8;
    static constexpr int CPU_OVERLOAD_THRESHOLD = 4;

    DeviceType decide_device(const Tile &tile)
    {
        int pixel_count = tile.width * tile.height;

        // Rule 1: Small tiles always go to CPU (GPU launch overhead not worth it)
        if (pixel_count < 256 * 256)
            return DeviceType::CPU;

        // Rule 1.5: Pipeline complexity considerations (Milestone 3)
        if (tile.usePipeline())
        {
            auto pipeline = tile.pipeline;
            // Complex pipelines (many stages, filters) → CPU for now
            if (pipeline->getStageCount() > 2 || pipeline->requiresOverlapRegion())
                return DeviceType::CPU;

            // Simple point-operation pipelines can go to GPU
            if (pipeline->isSimplePointOperation())
            {
                int gpuLoad = gpuQueueDepth.load();
                int cpuLoad = cpuQueueDepth.load();
                if (gpuLoad < GPU_OVERLOAD_THRESHOLD && gpuLoad <= cpuLoad)
                    return DeviceType::GPU;
            }
            return DeviceType::CPU;
        }

        // Rule 2: Irregular access patterns (rotation) → CPU
        // (Non-sequential memory access, poor GPU coalescing)
        if (tile.operation == TransformOperation::ROTATE_90_CW)
            return DeviceType::CPU;

        // Rule 3: Dense arithmetic on large tiles → GPU if not overloaded
        if (tile.operation == TransformOperation::GRAYSCALE)
        {
            int gpuLoad = gpuQueueDepth.load();
            int cpuLoad = cpuQueueDepth.load();

            // Route to GPU if it's less loaded and not overloaded
            if (gpuLoad < GPU_OVERLOAD_THRESHOLD && gpuLoad <= cpuLoad)
            {
                return DeviceType::GPU;
            }
        }

        // Fallback to CPU
        return DeviceType::CPU;
    }

public:
    Scheduler() : cpu(std::make_unique<OptimizedCPUWorker>()),
                  gpu(std::make_unique<AsyncGPUWorker>())
    {
    }

    /**
     * Get optimal tile size for a device
     */
    static int getOptimalTileSize(DeviceType device)
    {
        return TileOptimizer::getTileSizeForDevice(device);
    }

    /**
     * Get memory budget for a device
     */
    static size_t getMemoryBudget(DeviceType device)
    {
        return TileOptimizer::getMemoryBudgetForDevice(device);
    }

    /**
     * Dispatch task to appropriate worker with async transfers
     * UPDATED: Now accepts the pre-allocated VRAM buffer
     */
    void dispatch(Tile &task, uint8_t *d_buffer)
    {
        DeviceType device = decide_device(task);

        if (device == DeviceType::GPU)
        {
            gpuQueueDepth.fetch_add(1);
            try
            {
                // Async GPU execution with CUDA streams
                // Pass the pre-allocated VRAM buffer to the GPU worker
                gpu->execute(task, d_buffer);
            }
            catch (const std::exception &e)
            {
                // Fallback to CPU on GPU error
                std::cerr << "GPU execution failed, falling back to CPU. Error: " << e.what() << std::endl;
                cpuQueueDepth.fetch_add(1);
                cpu->execute(task);
                cpuQueueDepth.fetch_sub(1);
            }
            gpuQueueDepth.fetch_sub(1);
        }
        else
        {
            cpuQueueDepth.fetch_add(1);
            // Cache-optimized CPU execution
            // CPU doesn't need the d_buffer, it just operates on task.getRawPtr()
            cpu->execute(task);
            cpuQueueDepth.fetch_sub(1);
        }
    }

    /**
     * Get current queue depths
     */
    int getGPUQueueDepth() const { return gpuQueueDepth.load(); }
    int getCPUQueueDepth() const { return cpuQueueDepth.load(); }

    /**
     * Wait for all pending operations
     */
    void waitForCompletion()
    {
        if (gpu)
            gpu->waitForCompletion();
    }
};