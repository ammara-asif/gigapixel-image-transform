#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <tiffio.h>
#include "Tile.h"
 
/**
 * Reads rectangular regions (tiles) from a TIFF or BigTIFF image without
 * loading the entire image into memory. Supports scanline and tiled TIFFs.
 */
class TileReader {
public:
    /**
     * Opens a TIFF or BigTIFF file for reading.
     * Tries BigTIFF mode ("r8") first, falls back to classic TIFF ("r").
    */
    explicit TileReader(const std::string& filename);
 
    ~TileReader();
 
    /** Returns image width in pixels. */
    int getImageWidth()  const { return imgWidth;  }
 
    /** Returns image height in pixels. */
    int getImageHeight() const { return imgHeight; }
 
    /**
     * Reads a rectangular region of the image into a Tile, including overlap border.
     * x, y: Top-left pixel coordinate of the logical tile (without overlap)
     * width, height: Requested size of the logical tile (without overlap)
     * overlap: Extra pixels to read on each side for filter kernels
     */
    Tile getTile(int x, int y, int width, int height, int overlap = 0);
 
    /**
     * Computes an appropriate square tile size based on a memory budget.
     * targetMemoryBytes  Max bytes per tile (default 32MB)
     * overlap            Overlap pixels subtracted from each side
     * returns            Tile side length in pixels, aligned to native TIFF tile if tiled
     */
    int computeTileSize(size_t targetMemoryBytes = TILE_MEMORY_BUDGET, int overlap = 0);
 
    /**
     * Returns a flat grid of TileIndex descriptors covering the entire image.
     * Edge tiles are clamped to the image boundary (may be smaller than tileSize).
     * tileSize  Nominal tile side length from computeTileSize()
    */
    std::vector<TileIndex> getTileGrid(int tileSize);
 
private:
    std::string  filename;
    TIFF*        tif;
    uint32_t     imgWidth, imgHeight;
    int          channels;
    std::vector<uint8_t> nativeTileBuf; //holds TIFF tiles (temp buffer)

    // Reads a region from a scanline-organised TIFF into tile.data
    void readScanline(int x, int y, Tile& tile);
 
    // Reads a region from a tiled TIFF into tile.data
    void readTiled(int x, int y, Tile& tile);
};