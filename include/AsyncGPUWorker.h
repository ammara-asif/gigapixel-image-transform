#pragma once

#include <memory>
#include <mutex>
#include <queue>
#include <cstdint>
#include "Tile.h"
#include "IWorker.h"
#include "CUDAStreamManager.h"

/**
 * AsyncGPUWorker: GPU worker with asynchronous data transfer
 * * Features:
 * - CUDA streams for overlapping H2D, computation, and D2H
 * - Centralized VRAM management (Receives pre-allocated buffers)
 * - Automatic stream rotation
 * - Performance monitoring
 */
class AsyncGPUWorker : public IWorker
{
public:
    AsyncGPUWorker();
    ~AsyncGPUWorker();

    /**
     * Execute tile on GPU with async transfers
     * Overlaps:
     * H2D: Send tile data to device (using pre-allocated d_buffer)
     * Compute: Run GPU kernel on tile
     * D2H: Retrieve processed tile back to host
     * * NOTE: Signature updated to accept the pre-allocated VRAM buffer.
     * Ensure IWorker.h is also updated to match this signature if using polymorphism.
     */
    void execute(Tile &tile, uint8_t *d_buffer);

    // Keep the old signature for CPU fallback/interface compatibility if needed,
    // but it should throw an error or route to the new one if called directly on GPU.
    void execute(Tile &tile) override
    {
        throw std::runtime_error("AsyncGPUWorker requires a pre-allocated VRAM buffer. Call execute(tile, d_buffer) instead.");
    }

    /**
     * Get GPU device info
     */
    struct GPUInfo
    {
        int deviceId;
        int computeCapability;
        size_t totalMemory;
        size_t availableMemory;
        int maxThreadsPerBlock;
        int warpSize;
    };

    GPUInfo getGPUInfo() const { return gpuInfo; }

    /**
     * Set GPU device to use
     */
    void setGPUDevice(int deviceId);

    /**
     * Wait for all pending operations
     */
    void waitForCompletion();

private:
    std::unique_ptr<CUDAStreamManager> streamManager;
    GPUInfo gpuInfo;
    int currentGPU;

    // Pending operations tracking
    struct PendingOp
    {
        int streamId;
        uint8_t *deviceData;
        size_t dataSize;
    };
    std::queue<PendingOp> pendingOps;
    std::mutex operationsMutex;

    /**
     * Initialize GPU device and query capabilities
     */
    void initializeGPU();

    /**
     * Execute GPU kernel on device buffer
     */
    void executeGPUKernel(Tile &tile, uint8_t *deviceData, int streamId);

    /**
     * Handle async transfer completion
     */
    void onTransferComplete(PendingOp &op);

    // Performance monitoring
    struct PerfStats
    {
        double h2dTime;
        double computeTime;
        double d2hTime;
        double totalTime;
    } perfStats;

    void recordPerformance();
};