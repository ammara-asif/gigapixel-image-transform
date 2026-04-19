#pragma once

#include <memory>
#include <mutex>
#include <queue>
#include "Tile.h"
#include "IWorker.h"
#include "CUDAStreamManager.h"

/**
 * AsyncGPUWorker: GPU worker with asynchronous data transfer
 * 
 * Features:
 * - CUDA streams for overlapping H2D, computation, and D2H
 * - Triple buffering for maximum pipeline efficiency
 * - Automatic device memory management
 * - Performance monitoring
 */
class AsyncGPUWorker : public IWorker {
public:
    AsyncGPUWorker();
    ~AsyncGPUWorker();

    /**
     * Execute tile on GPU with async transfers
     * Overlaps:
     *   H2D: Send tile data to device
     *   Compute: Run GPU kernel on tile
     *   D2H: Retrieve processed tile back to host
     */
    void execute(Tile& tile) override;

    /**
     * Get GPU device info
     */
    struct GPUInfo {
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
    
    // Device memory buffers (triple buffering)
    struct DeviceBuffer {
        void* data;
        size_t size;
        bool allocated;
    };
    std::vector<DeviceBuffer> deviceBuffers;

    // Pending operations tracking
    struct PendingOp {
        int streamId;
        void* deviceData;
        size_t dataSize;
    };
    std::queue<PendingOp> pendingOps;
    std::mutex operationsMutex;

    /**
     * Initialize GPU device and query capabilities
     */
    void initializeGPU();

    /**
     * Allocate device memory buffer
     */
    void allocateDeviceBuffer(DeviceBuffer& buffer, size_t sizeBytes);

    /**
     * Free device memory buffer
     */
    void freeDeviceBuffer(DeviceBuffer& buffer);

    /**
     * Execute GPU kernel on device buffer
     */
    void executeGPUKernel(Tile& tile, void* deviceData, int streamId);

    /**
     * Handle async transfer completion
     */
    void onTransferComplete(PendingOp& op);

    // Performance monitoring
    struct PerfStats {
        double h2dTime;
        double computeTime;
        double d2hTime;
        double totalTime;
    } perfStats;

    void recordPerformance();
};
