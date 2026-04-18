#pragma once
#include <vector>
#include <cstdint>

// Memory budget per tile. Total usage ≈ TILE_MEMORY_BUDGET * num_threads * 2 (double buffering).
static constexpr size_t TILE_MEMORY_BUDGET = 32 * 1024 * 1024; // 32 MB

enum class TransformOperation
{
    GRAYSCALE,
    ROTATE_90_CW
};

// A rectangular region of image data loaded from disk.
struct Tile
{
    int x, y;          // Coordinate in the INPUT file (input location)
    int out_x, out_y;  // Coordinate in the OUTPUT file (write location)
    int width, height; // buffer dimensions (includes overlap)
    int overlap;

    TransformOperation operation;

    std::vector<uint8_t> data;
};

// Represents one cell in the tile grid
struct TileIndex
{
    int col, row;      // grid position
    int x, y;          // pixel origin in image
    int width, height; // actual size (may be smaller at image edges)
};