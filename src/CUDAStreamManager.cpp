#include "CUDAStreamManager.h"
#include <iostream>
#include <stdexcept>
#include <string>

CUDAStreamManager::CUDAStreamManager(int numStreams)
    : numStreams(numStreams), profilingEnabled(false)
{

    streams.resize(numStreams);
    streamAvailable.resize(numStreams, true);

    try
    {
        // Initialize actual CUDA streams
        for (int i = 0; i < numStreams; i++)
        {
            // cudaStreamNonBlocking allows this stream to run independently of the default stream
            cudaError_t status = cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
            checkCudaError(status, "cudaStreamCreateWithFlags");
        }

        std::cout << "[CUDAStreamManager] Initialized pool with " << numStreams << " generic CUDA streams." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[CUDAStreamManager] Initialization failed: " << e.what() << std::endl;
        throw;
    }
}

CUDAStreamManager::~CUDAStreamManager()
{
    for (int i = 0; i < numStreams; i++)
    {
        if (streams[i] != nullptr)
        {
            cudaStreamDestroy(streams[i]);
        }
    }
}

int CUDAStreamManager::acquireStream()
{
    std::lock_guard<std::mutex> lock(poolMutex);

    for (int i = 0; i < numStreams; i++)
    {
        if (streamAvailable[i])
        {
            streamAvailable[i] = false;
            return i;
        }
    }

    // If no stream is available, fallback to 0.
    // (If your VRAM pool size matches your Stream pool size, this will never hit).
    std::cerr << "[CUDAStreamManager] Warning: Stream pool exhausted. Falling back to Stream 0." << std::endl;
    return 0;
}

void CUDAStreamManager::launchH2DTransfer(int streamId, void *dst, const void *src, size_t numBytes)
{
    if (profilingEnabled)
    {
        std::cout << "[CUDAStreamManager] Launching H2D transfer: " << numBytes << " bytes on stream " << streamId << std::endl;
    }

    // Actual async H2D transfer
    cudaError_t status = cudaMemcpyAsync(dst, src, numBytes, cudaMemcpyHostToDevice, streams[streamId]);
    checkCudaError(status, "cudaMemcpyAsync (H2D)");
}

void CUDAStreamManager::launchD2HTransfer(int streamId, void *dst, const void *src, size_t numBytes)
{
    if (profilingEnabled)
    {
        std::cout << "[CUDAStreamManager] Launching D2H transfer: " << numBytes << " bytes on stream " << streamId << std::endl;
    }

    // Actual async D2H transfer
    cudaError_t status = cudaMemcpyAsync(dst, src, numBytes, cudaMemcpyDeviceToHost, streams[streamId]);
    checkCudaError(status, "cudaMemcpyAsync (D2H)");
}

void CUDAStreamManager::synchronizeStream(int streamId)
{
    if (streamId < 0 || streamId >= numStreams)
        return;

    // Block the CPU thread until this specific stream is completely done
    cudaError_t status = cudaStreamSynchronize(streams[streamId]);
    checkCudaError(status, "cudaStreamSynchronize");

    // Safely return the stream to the pool
    {
        std::lock_guard<std::mutex> lock(poolMutex);
        streamAvailable[streamId] = true;
    }
}

void CUDAStreamManager::synchronizeAllStreams()
{
    cudaDeviceSynchronize();

    std::lock_guard<std::mutex> lock(poolMutex);
    std::fill(streamAvailable.begin(), streamAvailable.end(), true);
}

cudaStream_t CUDAStreamManager::getRawCudaStream(int streamId) const
{
    return streams[streamId];
}

int CUDAStreamManager::getNumStreams() const
{
    return numStreams;
}

void CUDAStreamManager::enableProfiling(bool enable)
{
    profilingEnabled = enable;
}

void CUDAStreamManager::checkCudaError(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        std::string errorMsg = std::string(operation) + " failed: " + cudaGetErrorString(status);
        throw std::runtime_error(errorMsg);
    }
}