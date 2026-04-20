#pragma once
#include <vector>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h> // Required for CUDA memory APIs

// Memory budget per tile. Total usage ≈ TILE_MEMORY_BUDGET * num_threads * 2 (double buffering).
static constexpr size_t TILE_MEMORY_BUDGET = 32 * 1024 * 1024; // 32 MB

enum class TransformOperation
{
    GRAYSCALE,
    ROTATE_90_CW
};

// --- CUDA Custom Deleter ---
// Ensures pinned memory is safely freed when the Tile is destroyed
struct CudaHostDeleter
{
    void operator()(uint8_t *ptr) const
    {
        if (ptr)
        {
            cudaFreeHost(ptr);
        }
    }
};

// A rectangular region of image data loaded from disk.
struct Tile
{
    int x = 0, y = 0;          // Coordinate in the INPUT file (input location)
    int out_x = 0, out_y = 0;  // Coordinate in the OUTPUT file (write location)
    int width = 0, height = 0; // buffer dimensions (includes overlap)
    int overlap = 0;

    TransformOperation operation;

    // --- Pinned Memory Buffer ---
    // Replaced std::vector with a shared_ptr managing page-locked memory
    std::shared_ptr<uint8_t> data;

    // Track allocated size for easy reference during memcopies
    size_t dataSizeBytes = 0;

    // Helper to allocate pinned memory
    void allocate(size_t numBytes)
    {
        uint8_t *raw_ptr = nullptr;
        cudaError_t err = cudaMallocHost((void **)&raw_ptr, numBytes);

        if (err != cudaSuccess)
        {
            throw std::runtime_error("CUDA Error: Failed to allocate Pinned Memory! " +
                                     std::string(cudaGetErrorString(err)));
        }

        data = std::shared_ptr<uint8_t>(raw_ptr, CudaHostDeleter());
        dataSizeBytes = numBytes;
    }

    // Helper to get raw pointer for LibTIFF and CUDA functions
    uint8_t *getRawPtr() const
    {
        return data.get();
    }
};

// Represents one cell in the tile grid
struct TileIndex
{
    int col, row;      // grid position
    int x, y;          // pixel origin in image
    int width, height; // actual size (may be smaller at image edges)
};