#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <cuda_runtime.h> // Added actual CUDA runtime

/**
 * CUDAStreamManager: Manages a pool of CUDA streams for async data transfer.
 * * Architecture:
 * - A Worker Thread acquires a single stream for the lifespan of a tile.
 * - H2D, Compute, and D2H are queued sequentially onto that same stream.
 * - Concurrency is achieved because multiple threads use different streams simultaneously.
 */
class CUDAStreamManager
{
public:
    /**
     * Initialize CUDA streams
     * @param numStreams: Should ideally match your Worker Thread count
     */
    explicit CUDAStreamManager(int numStreams = 4);
    ~CUDAStreamManager();

    /**
     * Blocks (or searches) for an available stream and returns its ID.
     */
    int acquireStream();

    /**
     * Launch async H2D transfer on a specific stream
     */
    void launchH2DTransfer(int streamId, void *dst, const void *src, size_t numBytes);

    /**
     * Launch async D2H transfer on a specific stream (Renamed from D2D)
     */
    void launchD2HTransfer(int streamId, void *dst, const void *src, size_t numBytes);

    /**
     * Block the calling CPU thread until all operations on this stream finish.
     * Automatically releases the stream back to the pool.
     */
    void synchronizeStream(int streamId);

    /**
     * Synchronize all streams globally
     */
    void synchronizeAllStreams();

    /**
     * Fetch the actual raw cudaStream_t for passing into Kernel launches
     */
    cudaStream_t getRawCudaStream(int streamId) const;

    int getNumStreams() const;
    void enableProfiling(bool enable);

private:
    int numStreams;
    std::vector<cudaStream_t> streams; // Actual CUDA stream handles
    std::vector<bool> streamAvailable; // Availability status
    std::mutex poolMutex;              // Thread safety for concurrent workers
    bool profilingEnabled;

    void checkCudaError(cudaError_t status, const char *operation);
};