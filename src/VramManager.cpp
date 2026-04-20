#include "VramManager.h"

VramManager::VramManager(int numBuffers, size_t bufferSizeBytes)
    : bufferSize(bufferSizeBytes)
{
    for (int i = 0; i < numBuffers; ++i)
    {
        uint8_t *d_ptr = nullptr;
        cudaError_t err = cudaMalloc((void **)&d_ptr, bufferSize);

        if (err != cudaSuccess)
        {
            // Clean up whatever we managed to allocate before failing
            this->~VramManager();
            throw std::runtime_error("Memory Manager Error: Failed to pre-allocate VRAM! " +
                                     std::string(cudaGetErrorString(err)));
        }

        freeBuffers.push(d_ptr);
        allAllocations.push_back(d_ptr);
    }
}

VramManager::~VramManager()
{
    // We don't need the mutex here because destruction happens at the end of the program
    for (uint8_t *ptr : allAllocations)
    {
        if (ptr)
        {
            cudaFree(ptr);
        }
    }
    allAllocations.clear();
    while (!freeBuffers.empty())
        freeBuffers.pop();
}

uint8_t *VramManager::acquireBuffer()
{
    std::unique_lock<std::mutex> lock(mtx);

    // Wait until at least one buffer is available in the queue
    cv.wait(lock, [this]()
            { return !freeBuffers.empty(); });

    // Pop the available buffer and return it
    uint8_t *d_ptr = freeBuffers.front();
    freeBuffers.pop();

    return d_ptr;
}

void VramManager::releaseBuffer(uint8_t *devicePtr)
{
    if (!devicePtr)
        return;

    {
        std::lock_guard<std::mutex> lock(mtx);
        freeBuffers.push(devicePtr);
    }

    // Notify ONE waiting worker thread that a buffer is now available
    cv.notify_one();
}