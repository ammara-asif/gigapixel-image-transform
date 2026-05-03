#include "AsyncGPUWorker.h"
#include "Transform.h"
#include "CUDAKernels.h"  
#include <iostream>
#include <cstring>
#include <stdexcept>
#include <cuda_runtime.h> // Addaed for CUDA API

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


void AsyncGPUWorker::execute(Tile& tile, uint8_t* d_buffer)
{
    const size_t tileDataSize = tile.dataSizeBytes;

    std::cout << "[AsyncGPUWorker] Processing tile (" << tile.x << "," << tile.y
              << ")  " << tile.width << "x" << tile.height
              << "  size=" << (tileDataSize / 1024) << " KB" << std::endl;

    int streamId = -1;
    try
    {
        streamId = streamManager->acquireStream();
        cudaStream_t rawStream = streamManager->getRawCudaStream(streamId);

        // Full pipeline: H2D → kernel → D2H → sync — all inside processTileCuda.
        // It uses d_buffer for device memory and rawStream for async ordering.
        bool gpuOk = processTileCuda(tile, d_buffer, rawStream);

        if (!gpuOk)
        {
            // Operation not supported on GPU (e.g. ROTATE_90_CW) or CUDA error.
            // Fall back to CPU.  The stream is still released below.
            std::cout << "[AsyncGPUWorker] GPU path unavailable, falling back to CPU" << std::endl;
            processTile(tile);   // CPU fallback from Transform.h
        }

        // Release the stream back to the pool.
        // cudaStreamSynchronize inside processTileCuda already guaranteed the
        // device is idle, so this call returns immediately on the CUDA side.
        streamManager->synchronizeStream(streamId);

        recordPerformance();
    }
    catch (const std::exception& e)
    {
        std::cerr << "[AsyncGPUWorker] Execution error: " << e.what() << std::endl;

        // Always release the stream so the pool is not permanently exhausted.
        if (streamId >= 0)
            streamManager->synchronizeStream(streamId);

        // CPU fallback
        std::cout << "[AsyncGPUWorker] Falling back to CPU processing" << std::endl;
        processTile(tile);
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