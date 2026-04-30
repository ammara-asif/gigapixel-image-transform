#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

class VramManager
{
public:
    /**
     * Initializes the VRAM pool.
     * @param numBuffers How many concurrent GPU buffers to allocate (usually matches Worker Thread count + a few extra).
     * @param bufferSizeBytes The maximum size of a single tile in bytes.
     */
    VramManager(int numBuffers, size_t bufferSizeBytes);

    ~VramManager();

    // Disable copy/move to prevent accidental double-frees
    VramManager(const VramManager &) = delete;
    VramManager &operator=(const VramManager &) = delete;

    /**
     * Blocks until a VRAM buffer becomes available, then returns a pointer to it.
     */
    uint8_t *acquireBuffer();

    /**
     * Returns a VRAM buffer back to the pool for another thread to use.
     */
    void releaseBuffer(uint8_t *devicePtr);

private:
    std::queue<uint8_t *> freeBuffers;
    std::vector<uint8_t *> allAllocations; // Keep track for cleanup

    std::mutex mtx;
    std::condition_variable cv;

    size_t bufferSize;
};