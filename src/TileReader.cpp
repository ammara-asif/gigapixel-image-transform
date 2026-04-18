#include "TileReader.h"
#include <iostream>
#include <cmath>
#include <tiffio.h>

TileReader::TileReader(const std::string &filename) : filename(filename), tif(nullptr)
{
    tif = TIFFOpen(filename.c_str(), "r8");
    if (!tif)
    {
        tif = TIFFOpen(filename.c_str(), "r");
    }
    if (!tif)
        throw std::runtime_error("Failed to open TIFF");
    std::cout << "Reading file " << filename << std::endl;

    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &imgWidth);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imgHeight);

    uint16_t samplesPerPixel;
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
    channels = samplesPerPixel;

    uint16_t bitsPerSample;
    TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitsPerSample);
    if (bitsPerSample != 8)
    {
        TIFFClose(tif);
        tif = nullptr;
        throw std::runtime_error(
            "Unsupported bit depth: " + std::to_string(bitsPerSample) +
            " bps. Only 8-bit TIFFs are currently supported.");
    }
}

TileReader::~TileReader()
{
    if (tif)
        TIFFClose(tif);
}

void TileReader::readScanline(int x, int y, Tile &tile)
{
    std::vector<uint8_t> rowBuffer(imgWidth * channels);

    // map pixels from row buffer to tile buffer
    for (int row = 0; row < tile.height; row++)
    {
        int imgRow = y + row;
        int result = TIFFReadScanline(tif, rowBuffer.data(), imgRow);
        if (result < 0)
        {
            throw std::runtime_error(
                "TIFFReadScanline failed at row " + std::to_string(imgRow));
        }

        for (int col = 0; col < tile.width; col++)
        {
            int imgCol = x + col;

            for (int c = 0; c < channels; c++)
            {
                tile.data[(row * tile.width + col) * channels + c] =
                    rowBuffer[(imgCol * channels) + c];
            }
        }
    }
}

void TileReader::readTiled(int x, int y, Tile &tile)
{
    // Get the native tile dimensions from the file
    uint32_t nativeTileW, nativeTileH;
    TIFFGetField(tif, TIFFTAG_TILEWIDTH, &nativeTileW);
    TIFFGetField(tif, TIFFTAG_TILELENGTH, &nativeTileH);
    // std::cout<<"Native tile: "<<nativeTileH<<"x"<<nativeTileW<<std::endl;

    // Temporary buffer for one native TIFF tile
    nativeTileBuf.resize(nativeTileW * nativeTileH * channels);

    // Find which native tiles overlap requested region [x, x+tile.width) x [y, y+tile.height)
    int tileStartCol = x / nativeTileW;
    int tileStartRow = y / nativeTileH;
    int tileEndCol = (x + tile.width - 1) / nativeTileW;
    int tileEndRow = (y + tile.height - 1) / nativeTileH;

    for (int tr = tileStartRow; tr <= tileEndRow; tr++)
    {
        for (int tc = tileStartCol; tc <= tileEndCol; tc++)
        {

            // Pixel origin of this native tile in image coordinates
            int nativeOriginX = tc * nativeTileW;
            int nativeOriginY = tr * nativeTileH;

            // read full TIFF tile
            tsize_t result = TIFFReadTile(tif, nativeTileBuf.data(), nativeOriginX, nativeOriginY, 0, 0);
            if (result < 0)
                throw std::runtime_error(
                    "TIFFReadTile failed at (" + std::to_string(nativeOriginX) +
                    ", " + std::to_string(nativeOriginY) + ")");

            // Compute the intersection of [nativeOrigin, nativeOrigin+nativeSize)
            // with our requested region [x, x+tile.width) x [y, y+tile.height)
            int srcX0 = std::max(x, nativeOriginX);
            int srcY0 = std::max(y, nativeOriginY);
            int srcX1 = std::min(x + tile.width, (int)(nativeOriginX + nativeTileW));
            int srcY1 = std::min(y + tile.height, (int)(nativeOriginY + nativeTileH));

            // Copy the intersection pixels into the output buffer
            for (int py = srcY0; py < srcY1; py++)
            {
                for (int px = srcX0; px < srcX1; px++)
                {

                    // Position in native tile buffer
                    int nativeCol = px - nativeOriginX;
                    int nativeRow = py - nativeOriginY;
                    int nativeIdx = (nativeRow * nativeTileW + nativeCol) * channels;

                    // Position in output tile buffer
                    int outCol = px - x;
                    int outRow = py - y;
                    int outIdx = (outRow * tile.width + outCol) * channels;

                    for (int c = 0; c < channels; c++)
                        tile.data[outIdx + c] = nativeTileBuf[nativeIdx + c];
                }
            }
        }
    }
}

Tile TileReader::getTile(int x, int y, int width, int height, int overlap)
{
    if (x < 0 || y < 0 || width <= 0 || height <= 0)
    {
        throw std::runtime_error("Tile coordinates or size are invalid");
    }

    if (x >= (int)imgWidth || y >= (int)imgHeight)
    {
        throw std::runtime_error("Tile coordinates outside image bounds");
    }

    // Compute overlapped region
    int x0 = std::max(0, x - overlap);
    int y0 = std::max(0, y - overlap);

    int x1 = std::min((int)imgWidth, x + width + overlap);
    int y1 = std::min((int)imgHeight, y + height + overlap);

    int readWidth = x1 - x0;
    int readHeight = y1 - y0;

    Tile tile;
    tile.x = x;
    tile.y = y;
    tile.width = readWidth;
    tile.height = readHeight;
    tile.overlap = overlap;
    tile.data.resize(readWidth * readHeight * channels);

    // read data
    if (TIFFIsTiled(tif))
    {
        readTiled(x0, y0, tile);
    }
    else
    {
        readScanline(x0, y0, tile);
    }

    return tile;
}

int TileReader::computeTileSize(size_t targetMemoryBytes, int overlap)
{
    size_t tileArea = targetMemoryBytes / channels;
    int tileSize = static_cast<int>(sqrt((double)tileArea)) - 2 * overlap;

    // Clamp minimum size(avod tiny tiles)
    tileSize = std::max(tileSize, 128);

    // Align to TIFF tile if exists
    if (TIFFIsTiled(tif))
    {
        uint32_t tileWidth, tileHeight;
        TIFFGetField(tif, TIFFTAG_TILEWIDTH, &tileWidth);
        TIFFGetField(tif, TIFFTAG_TILELENGTH, &tileHeight);

        int alignSize = std::min(tileWidth, tileHeight);
        int aligned = (tileSize / alignSize) * alignSize;
        tileSize = (aligned > 0) ? aligned : alignSize;
    }

    return tileSize;
}

std::vector<TileIndex> TileReader::getTileGrid(int tileSize)
{
    std::vector<TileIndex> grid;

    int nTilesX = (imgWidth + tileSize - 1) / tileSize; // ceiling division
    int nTilesY = (imgHeight + tileSize - 1) / tileSize;

    for (int row = 0; row < nTilesY; row++)
    {
        for (int col = 0; col < nTilesX; col++)
        {
            TileIndex idx;
            idx.col = col;
            idx.row = row;
            idx.x = col * tileSize;
            idx.y = row * tileSize;
            // clamp edge tiles to image boundary
            idx.width = std::min(tileSize, (int)imgWidth - idx.x);
            idx.height = std::min(tileSize, (int)imgHeight - idx.y);
            grid.push_back(idx);
        }
    }
    return grid;
}
