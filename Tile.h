#pragma once
#include <vector>
#include <cstdint>

// Memory budget per tile. Total usage ≈ TILE_MEMORY_BUDGET * num_threads * 2 (double buffering).
static constexpr size_t TILE_MEMORY_BUDGET = 32 * 1024 * 1024; // 32 MB


//A rectangular region of image data loaded from disk.
struct Tile {
    int x, y;           // logical tile origin (without overlap)
    int width, height;  // buffer dimensions (includes overlap)
    int overlap;
    std::vector<uint8_t> data;
};

// Represents one cell in the tile grid
struct TileIndex {
    int col, row;       // grid position
    int x, y;           // pixel origin in image
    int width, height;  // actual size (may be smaller at image edges)
};