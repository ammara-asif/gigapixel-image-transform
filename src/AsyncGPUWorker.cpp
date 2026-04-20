#include "AsyncGPUWorker.h"
#include "Transform.h"
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <cuda_runtime.h> // Added for CUDA API

AsyncGPUWorker::AsyncGPUWorker() : currentGPU(0)
{
    try
    {
        // Initialize stream manager (assume it creates a pool of streams)
        streamManager = std::make_unique<CUDAStreamManager>(3);

        // Initialize GPU capabilities
        initializeGPU();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[AsyncGPUWorker] Initialization failed: " << e.what() << std::endl;
        throw;
    }
}

AsyncGPUWorker::~AsyncGPUWorker()
{
    try
    {
        waitForCompletion();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[AsyncGPUWorker] Destruction error: " << e.what() << std::endl;
    }
}

void AsyncGPUWorker::initializeGPU()
{
    // In a real scenario, use cudaGetDeviceProperties
    // cudaDeviceProp props;
    // cudaGetDeviceProperties(&props, currentGPU);

    // Placeholder values
    gpuInfo.deviceId = 0;
    gpuInfo.computeCapability = 75;                  // Turing or similar
    gpuInfo.totalMemory = 8ULL * 1024 * 1024 * 1024; // 8 GB
    gpuInfo.availableMemory = gpuInfo.totalMemory;
    gpuInfo.maxThreadsPerBlock = 1024;
    gpuInfo.warpSize = 32;

    std::cout << "[AsyncGPUWorker] GPU Initialized: CC " << gpuInfo.computeCapability
              << ", Memory: " << (gpuInfo.totalMemory / (1024 * 1024)) << " MB" << std::endl;
}

void AsyncGPUWorker::setGPUDevice(int deviceId)
{
    currentGPU = deviceId;
    // cudaSetDevice(deviceId);
    initializeGPU();
}

// UPDATED: Now accepts the pre-allocated VRAM buffer (d_buffer)
void AsyncGPUWorker::execute(Tile &tile, uint8_t *d_buffer)
{
    // Use the exact allocated size from our pinned memory Tile update
    size_t tileDataSize = tile.dataSizeBytes;

    try
    {
        // Phase 1: Async H2D Transfer
        std::cout << "[AsyncGPUWorker] Async H2D transfer for tile (" << tile.x << ","
                  << tile.y << "), size: " << (tileDataSize / 1024) << " KB" << std::endl;

        // Note: Using tile.getRawPtr() to access the pinned host memory
        int streamId = streamManager->acquireStream();

        // Phase 2: GPU Computation
        std::cout << "[AsyncGPUWorker] Launching GPU kernel on tile (" << tile.x << ","
                  << tile.y << ")" << std::endl;
        executeGPUKernel(tile, d_buffer, streamId);

        // Phase 3: Async D2H Transfer
        std::cout << "[AsyncGPUWorker] Async D2H transfer for tile (" << tile.x << ","
                  << tile.y << ")" << std::endl;

        // The D2DTransfer method name in your old code looks like a typo (launchD2DTransfer),
        // assuming it acts as a Device-To-Host transfer here.
        streamManager->launchD2HTransfer(streamId, tile.getRawPtr(), d_buffer, tileDataSize);

        // Phase 4: Stream Synchronization
        // CRITICAL: We must block this specific worker thread until the GPU finishes this tile.
        // If we don't, main.cpp will return d_buffer to the pool while the GPU is still using it!
        // Because other worker threads are doing this simultaneously on OTHER streams,
        // the GPU remains fully saturated and overlaps transfers/compute.

        // Assuming your CUDAStreamManager has a method to sync a specific stream.
        // If not, you will need to add it, or use standard CUDA: cudaStreamSynchronize(stream);
        // streamManager->synchronizeStream(streamId);
        streamManager->synchronizeStream(streamId);

        recordPerformance();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[AsyncGPUWorker] Execution error: " << e.what() << std::endl;

        // Fallback to CPU processing
        std::cout << "[AsyncGPUWorker] Falling back to CPU processing" << std::endl;
        // processTile(tile); // Uncomment/implement your CPU fallback logic here
    }
}

void AsyncGPUWorker::executeGPUKernel(Tile &tile, uint8_t *deviceData, int streamId)
{
    // TODO: Actual CUDA kernel launch
    // Example: myKernel<<<blocks, threads, 0, actualCudaStream>>>(deviceData, tile.width, tile.height);

    std::cout << "[AsyncGPUWorker] GPU kernel execution on tile (" << tile.x << ","
              << tile.y << ") using stream " << streamId << std::endl;
}

void AsyncGPUWorker::waitForCompletion()
{
    if (streamManager)
    {
        streamManager->synchronizeAllStreams();
    }
}

void AsyncGPUWorker::recordPerformance()
{
    // TODO: Actual performance measurements via CUDA Events
    perfStats.h2dTime = 0.0;
    perfStats.computeTime = 0.0;
    perfStats.d2hTime = 0.0;
    perfStats.totalTime = 0.0;
}