#include "Transform.h"
#include "PipelineComposition.h"
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstring>

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Allocate a new buffer (mimics Tile's pinned memory allocation)
 */
static std::shared_ptr<uint8_t> allocateBuffer(size_t numBytes)
{
    uint8_t* raw_ptr = nullptr;
    cudaError_t err = cudaMallocHost((void**)&raw_ptr, numBytes);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to allocate pinned memory for buffer");
    }
    // Use same deleter as Tile
    struct CudaHostDeleter {
        void operator()(uint8_t* ptr) const {
            if (ptr) cudaFreeHost(ptr);
        }
    };
    return std::shared_ptr<uint8_t>(raw_ptr, CudaHostDeleter());
}

/**
 * Clamp value to [0, 255]
 */
static inline uint8_t clamp255(float val)
{
    if (val < 0.0f) return 0;
    if (val > 255.0f) return 255;
    return static_cast<uint8_t>(val);
}

/**
 * Bilinear interpolation for resampling
 */
static inline uint8_t bilinearInterpolate(
    const uint8_t* src, int srcWidth, int srcHeight,
    float x, float y, int channel)
{
    // Clamp coordinates
    x = std::max(0.0f, std::min(static_cast<float>(srcWidth - 1), x));
    y = std::max(0.0f, std::min(static_cast<float>(srcHeight - 1), y));

    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int x1 = std::min(x0 + 1, srcWidth - 1);
    int y1 = std::min(y0 + 1, srcHeight - 1);

    float dx = x - x0;
    float dy = y - y0;

    uint8_t v00 = src[(y0 * srcWidth + x0) * 3 + channel];
    uint8_t v10 = src[(y0 * srcWidth + x1) * 3 + channel];
    uint8_t v01 = src[(y1 * srcWidth + x0) * 3 + channel];
    uint8_t v11 = src[(y1 * srcWidth + x1) * 3 + channel];

    float v0 = v00 * (1 - dx) + v10 * dx;
    float v1 = v01 * (1 - dx) + v11 * dx;
    float result = v0 * (1 - dy) + v1 * dy;

    return clamp255(result);
}

/**
 * Get pixel value with boundary handling (zero-pad)
 */
static inline uint8_t getPixelSafe(
    const uint8_t* data, int width, int height, 
    int x, int y, int channel)
{
    if (x < 0 || x >= width || y < 0 || y >= height)
        return 0;
    return data[(y * width + x) * 3 + channel];
}

// ============================================================================
// TRANSFORM IMPLEMENTATIONS
// ============================================================================

/**
 * GRAYSCALE: Point operation - RGB to grayscale
 */
static void transform_grayscale(Tile& tile, const TransformParams& params = TransformParams())
{
    (void)params;  // Unused for grayscale
    size_t numPixels = tile.width * tile.height;

    for (size_t i = 0; i < numPixels; ++i)
    {
        size_t idx = i * 3;
        uint8_t r = tile.data.get()[idx];
        uint8_t g = tile.data.get()[idx + 1];
        uint8_t b = tile.data.get()[idx + 2];

        uint8_t gray = static_cast<uint8_t>(0.299 * r + 0.587 * g + 0.114 * b);

        tile.data.get()[idx] = gray;
        tile.data.get()[idx + 1] = gray;
        tile.data.get()[idx + 2] = gray;
    }
}

/**
 * ROTATE_90_CW: Geometric operation - 90 degree clockwise rotation
 * Requires output buffer and swaps dimensions
 */
static std::shared_ptr<uint8_t> transform_rotate_90cw(Tile& tile, const TransformParams& params = TransformParams())
{
    (void)params;  // Unused for rotation
    size_t bufferSize = tile.width * tile.height * 3;
    auto newData = allocateBuffer(bufferSize);

    uint8_t* src = tile.data.get();
    uint8_t* dst = newData.get();

    for (int y = 0; y < tile.height; ++y)
    {
        for (int x = 0; x < tile.width; ++x)
        {
            size_t srcIdx = (y * tile.width + x) * 3;

            int dstX = tile.height - 1 - y;
            int dstY = x;
            size_t dstIdx = (dstY * tile.height + dstX) * 3;

            dst[dstIdx] = src[srcIdx];
            dst[dstIdx + 1] = src[srcIdx + 1];
            dst[dstIdx + 2] = src[srcIdx + 2];
        }
    }

    // Swap dimensions
    std::swap(tile.width, tile.height);
    return newData;
}

/**
 * RESIZE: Geometric operation - bilinear resampling
 * Requires output buffer and changes dimensions
 */
static std::shared_ptr<uint8_t> transform_resize(Tile& tile, const TransformParams& params)
{
    float scaleX = params.scaleX;
    float scaleY = params.scaleY;

    int newWidth = static_cast<int>(tile.width * scaleX);
    int newHeight = static_cast<int>(tile.height * scaleY);
    
    // Ensure at least 1 pixel
    newWidth = std::max(1, newWidth);
    newHeight = std::max(1, newHeight);

    size_t newSize = newWidth * newHeight * 3;
    auto newData = allocateBuffer(newSize);

    uint8_t* src = tile.data.get();
    uint8_t* dst = newData.get();

    for (int y = 0; y < newHeight; ++y)
    {
        for (int x = 0; x < newWidth; ++x)
        {
            // Map destination pixel to source coordinates
            float srcX = x / scaleX;
            float srcY = y / scaleY;

            for (int c = 0; c < 3; ++c)
            {
                uint8_t val = bilinearInterpolate(src, tile.width, tile.height, srcX, srcY, c);
                dst[(y * newWidth + x) * 3 + c] = val;
            }
        }
    }

    tile.width = newWidth;
    tile.height = newHeight;
    return newData;
}

/**
 * COLOR_CORRECTION: Point operation - brightness and contrast adjustment
 */
static void transform_color_correction(Tile& tile, const TransformParams& params)
{
    float brightness = params.brightness;  // -100 to +100
    float contrast = params.contrast;      // 0.5 to 2.0

    size_t numPixels = tile.width * tile.height;
    uint8_t* data = tile.data.get();

    for (size_t i = 0; i < numPixels; ++i)
    {
        for (int c = 0; c < 3; ++c)
        {
            size_t idx = i * 3 + c;
            float val = data[idx];

            // Apply contrast: (val - 128) * contrast + 128
            val = (val - 128.0f) * contrast + 128.0f;

            // Apply brightness
            val += brightness;

            data[idx] = clamp255(val);
        }
    }
}

/**
 * FILTER_GAUSSIAN: Spatial filter - Gaussian blur via separable 1D kernels
 */
static std::shared_ptr<uint8_t> transform_gaussian_filter(Tile& tile, const TransformParams& params)
{
    int kernelSize = params.kernelSize;
    if (kernelSize < 1) kernelSize = 1;
    if (kernelSize % 2 == 0) kernelSize++;  // Ensure odd

    float sigma = params.sigma;
    int radius = kernelSize / 2;

    // Generate 1D Gaussian kernel
    std::vector<float> kernel(kernelSize);
    float sum = 0.0f;
    for (int i = 0; i < kernelSize; ++i)
    {
        float x = i - radius;
        kernel[i] = std::exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }
    for (int i = 0; i < kernelSize; ++i)
        kernel[i] /= sum;

    size_t bufferSize = tile.width * tile.height * 3;
    auto tempData = allocateBuffer(bufferSize);
    auto newData = allocateBuffer(bufferSize);

    uint8_t* src = tile.data.get();
    uint8_t* temp = tempData.get();
    uint8_t* dst = newData.get();

    // Horizontal pass
    for (int y = 0; y < tile.height; ++y)
    {
        for (int x = 0; x < tile.width; ++x)
        {
            for (int c = 0; c < 3; ++c)
            {
                float sum = 0.0f;
                for (int k = 0; k < kernelSize; ++k)
                {
                    int kx = x + k - radius;
                    uint8_t val = getPixelSafe(src, tile.width, tile.height, kx, y, c);
                    sum += val * kernel[k];
                }
                temp[(y * tile.width + x) * 3 + c] = clamp255(sum);
            }
        }
    }

    // Vertical pass
    for (int y = 0; y < tile.height; ++y)
    {
        for (int x = 0; x < tile.width; ++x)
        {
            for (int c = 0; c < 3; ++c)
            {
                float sum = 0.0f;
                for (int k = 0; k < kernelSize; ++k)
                {
                    int ky = y + k - radius;
                    uint8_t val = getPixelSafe(temp, tile.width, tile.height, x, ky, c);
                    sum += val * kernel[k];
                }
                dst[(y * tile.width + x) * 3 + c] = clamp255(sum);
            }
        }
    }

    return newData;
}

/**
 * FILTER_MEDIAN: Spatial filter - median blur for noise reduction
 */
static std::shared_ptr<uint8_t> transform_median_filter(Tile& tile, const TransformParams& params)
{
    int kernelSize = params.kernelSize;
    if (kernelSize < 1) kernelSize = 1;
    if (kernelSize % 2 == 0) kernelSize++;  // Ensure odd

    int radius = kernelSize / 2;

    size_t bufferSize = tile.width * tile.height * 3;
    auto newData = allocateBuffer(bufferSize);

    uint8_t* src = tile.data.get();
    uint8_t* dst = newData.get();

    std::vector<uint8_t> window;
    window.reserve(kernelSize * kernelSize);

    for (int y = 0; y < tile.height; ++y)
    {
        for (int x = 0; x < tile.width; ++x)
        {
            for (int c = 0; c < 3; ++c)
            {
                window.clear();

                // Collect neighborhood
                for (int ky = -radius; ky <= radius; ++ky)
                {
                    for (int kx = -radius; kx <= radius; ++kx)
                    {
                        uint8_t val = getPixelSafe(src, tile.width, tile.height, x + kx, y + ky, c);
                        window.push_back(val);
                    }
                }

                // Find median
                std::nth_element(window.begin(), window.begin() + window.size() / 2, window.end());
                uint8_t median = window[window.size() / 2];

                dst[(y * tile.width + x) * 3 + c] = median;
            }
        }
    }

    return newData;
}

/**
 * FILTER_SOBEL: Edge detection filter
 */
static std::shared_ptr<uint8_t> transform_sobel_filter(Tile& tile, const TransformParams& params)
{
    bool sobelX = params.sobelX;
    bool sobelY = params.sobelY;

    size_t bufferSize = tile.width * tile.height * 3;
    auto newData = allocateBuffer(bufferSize);

    uint8_t* src = tile.data.get();
    uint8_t* dst = newData.get();

    // Sobel kernels
    int sobelKernelX[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    int sobelKernelY[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };

    for (int y = 0; y < tile.height; ++y)
    {
        for (int x = 0; x < tile.width; ++x)
        {
            for (int c = 0; c < 3; ++c)
            {
                float gx = 0.0f, gy = 0.0f;

                for (int ky = -1; ky <= 1; ++ky)
                {
                    for (int kx = -1; kx <= 1; ++kx)
                    {
                        uint8_t val = getPixelSafe(src, tile.width, tile.height, x + kx, y + ky, c);
                        int kidx = (ky + 1) * 3 + (kx + 1);

                        if (sobelX) gx += val * sobelKernelX[kidx];
                        if (sobelY) gy += val * sobelKernelY[kidx];
                    }
                }

                float magnitude = std::sqrt(gx * gx + gy * gy);
                dst[(y * tile.width + x) * 3 + c] = clamp255(magnitude);
            }
        }
    }

    return newData;
}

/**
 * IDENTITY: No-op transform (for testing)
 */
static void transform_identity(Tile& tile, const TransformParams& params = TransformParams())
{
    (void)tile;    // No-op
    (void)params;
}

// ============================================================================
// PUBLIC API - SINGLE TRANSFORM EXECUTION
// ============================================================================

/**
 * Process a single transform stage on a tile
 * Returns new buffer if output dimensions differ, otherwise modifies in-place
 */
std::shared_ptr<uint8_t> processTransformStage(
    Tile& tile,
    const PipelineStage& stage,
    bool needsOutputBuffer)
{
    std::shared_ptr<uint8_t> newBuffer = nullptr;

    switch (stage.operation)
    {
    case TransformOperation::GRAYSCALE:
        transform_grayscale(tile, stage.params);
        break;

    case TransformOperation::ROTATE_90_CW:
        newBuffer = transform_rotate_90cw(tile, stage.params);
        break;

    case TransformOperation::RESIZE:
        newBuffer = transform_resize(tile, stage.params);
        break;

    case TransformOperation::COLOR_CORRECTION:
        transform_color_correction(tile, stage.params);
        break;

    case TransformOperation::FILTER_GAUSSIAN:
        newBuffer = transform_gaussian_filter(tile, stage.params);
        break;

    case TransformOperation::FILTER_MEDIAN:
        newBuffer = transform_median_filter(tile, stage.params);
        break;

    case TransformOperation::FILTER_SOBEL:
        newBuffer = transform_sobel_filter(tile, stage.params);
        break;

    case TransformOperation::IDENTITY:
        transform_identity(tile, stage.params);
        break;

    default:
        break;
    }

    return newBuffer;
}

/**
 * Legacy API: Process single operation on tile
 */
void processTile(Tile &tile)
{
    if (tile.usePipeline())
    {
        processTilePipeline(tile, tile.pipeline);
    }
    else
    {
        PipelineStage stage(tile.operation);
        processTransformStage(tile, stage);
    }
}

/**
 * Execute complete pipeline on a tile
 * Handles intermediate buffer management
 */
void processTilePipeline(Tile& tile, const std::shared_ptr<TransformPipeline>& pipeline)
{
    if (!pipeline || pipeline->isEmpty())
        return;

    for (size_t i = 0; i < pipeline->getStageCount(); ++i)
    {
        const auto& stage = pipeline->getStage(i);

        std::shared_ptr<uint8_t> newBuffer = processTransformStage(tile, stage);

        // If stage returned new buffer (geometric transform), update tile
        if (newBuffer)
        {
            tile.data = newBuffer;
            tile.dataSizeBytes = tile.width * tile.height * 3;
        }
    }
}