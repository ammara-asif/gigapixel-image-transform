#include "Transform.h"
#include <vector>
#include <cstdint>

void processTile(Tile &tile)
{
    int channels = tile.data.size() / (tile.width * tile.height);
    size_t numPixels = tile.width * tile.height;

    // --- ALGORITHM 1: POINT OPERATION (GRAYSCALE) ---
    if (tile.operation == TransformOperation::GRAYSCALE)
    {
        // Safe to do In-Place
        for (size_t i = 0; i < numPixels; ++i)
        {
            size_t idx = i * channels;
            uint8_t r = tile.data[idx];
            uint8_t g = tile.data[idx + 1];
            uint8_t b = tile.data[idx + 2];

            uint8_t gray = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b);

            tile.data[idx] = gray;
            tile.data[idx + 1] = gray;
            tile.data[idx + 2] = gray;
        }
    }
    // --- ALGORITHM 2: GEOMETRIC OPERATION (90 CW) ---
    else if (tile.operation == TransformOperation::ROTATE_90_CW)
    {
        // Must do Out-of-Place
        std::vector<uint8_t> rotated_buffer(tile.data.size(), 0);

        for (int y = 0; y < tile.height; ++y)
        {
            for (int x = 0; x < tile.width; ++x)
            {

                size_t src_idx = (y * tile.width + x) * channels;

                int dst_x = tile.height - 1 - y;
                int dst_y = x;

                size_t dst_idx = (dst_y * tile.height + dst_x) * channels;

                for (int c = 0; c < channels; ++c)
                {
                    rotated_buffer[dst_idx + c] = tile.data[src_idx + c];
                }
            }
        }

        // Swap dimensions
        int temp = tile.width;
        tile.width = tile.height;
        tile.height = temp;

        tile.data = std::move(rotated_buffer);
    }
}