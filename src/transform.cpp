#include "Transform.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>

void processTile(Tile &tile)
{
    if (!tile.getRawPtr() || tile.dataSizeBytes == 0)
        throw std::runtime_error("processTile: tile has no data");

    size_t numPixels = static_cast<size_t>(tile.width) * static_cast<size_t>(tile.height);
    if (numPixels == 0)
        return;

    // dataSizeBytes = width * height * channels, so:
    int channels = static_cast<int>(tile.dataSizeBytes / numPixels);
    uint8_t* px = tile.getRawPtr();

    // --- ALGORITHM 1: POINT OPERATION (GRAYSCALE) ---
    if (tile.operation == TransformOperation::GRAYSCALE)
    {
        // In-place: read R/G/B, write gray back to all three channels
        for (size_t i = 0; i < numPixels; ++i)
        {
            size_t idx = i * channels;
            uint8_t r = px[idx];
            uint8_t g = px[idx + 1];
            uint8_t b = px[idx + 2];

            uint8_t gray = static_cast<uint8_t>(0.299f * r + 0.587f * g + 0.114f * b);

            px[idx]     = gray;
            px[idx + 1] = gray;
            px[idx + 2] = gray;
        }
    }
    // --- ALGORITHM 2: GEOMETRIC OPERATION (ROTATE 90 CW) ---
    else if (tile.operation == TransformOperation::ROTATE_90_CW)
    {
        // Out-of-place: allocate a plain vector as the rotation buffer,
        // then copy the result back into the tile's pinned memory.
        std::vector<uint8_t> rotated(tile.dataSizeBytes, 0);

        for (int y = 0; y < tile.height; ++y)
        {
            for (int x = 0; x < tile.width; ++x)
            {
                size_t src_idx = (static_cast<size_t>(y) * tile.width + x) * channels;

                // 90 CW mapping: dst(x, H-1-y) = src(x, y)
                int dst_x = tile.height - 1 - y;
                int dst_y = x;
                size_t dst_idx = (static_cast<size_t>(dst_y) * tile.height + dst_x) * channels;

                for (int c = 0; c < channels; ++c)
                    rotated[dst_idx + c] = px[src_idx + c];
            }
        }

        // Copy rotated pixels back into the pinned buffer.
        // We cannot move a std::vector into a shared_ptr<uint8_t>,
        // so we memcpy instead — the pinned allocation is already the right size.
        std::memcpy(px, rotated.data(), tile.dataSizeBytes);

        // Swap dimensions to reflect the new geometry
        std::swap(tile.width, tile.height);
    }
}