#include "CUDAKernels.h"

#ifdef USE_CUDA

#include "CUDAKernels.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                          \
    do {                                                                                          \
        cudaError_t err__ = (call);                                                               \
        if (err__ != cudaSuccess) {                                                               \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,                 \
                         cudaGetErrorString(err__));                                              \
            return false;                                                                         \
        }                                                                                         \
    } while (0)

namespace {

__global__ void rgbToGrayKernel(const uchar3* rgb, uint8_t* gray, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int idx = y * width + x;
    uchar3 p = rgb[idx];
    float g = 0.299f * static_cast<float>(p.x) + 0.587f * static_cast<float>(p.y) +
              0.114f * static_cast<float>(p.z);
    g = fminf(255.0f, fmaxf(0.0f, g));
    gray[idx] = static_cast<uint8_t>(g);
}

__global__ void convolve2DKernel(
    const float* in, float* out, int width, int height, const float* kernel, int ksize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int r = ksize / 2;
    float acc = 0.0f;

    for (int ky = -r; ky <= r; ++ky) {
        for (int kx = -r; kx <= r; ++kx) {
            int sx = min(max(x + kx, 0), width - 1);
            int sy = min(max(y + ky, 0), height - 1);
            float pix = in[sy * width + sx];
            float kw = kernel[(ky + r) * ksize + (kx + r)];
            acc += pix * kw;
        }
    }

    out[y * width + x] = acc;
}

__device__ uint8_t bilinearSample(const uint8_t* img, int w, int h, float x, float y)
{
    if (x < 0.0f || y < 0.0f || x > static_cast<float>(w - 1) || y > static_cast<float>(h - 1)) {
        return 0;
    }

    int x0 = static_cast<int>(floorf(x));
    int y0 = static_cast<int>(floorf(y));
    int x1 = min(x0 + 1, w - 1);
    int y1 = min(y0 + 1, h - 1);

    float dx = x - static_cast<float>(x0);
    float dy = y - static_cast<float>(y0);

    float p00 = static_cast<float>(img[y0 * w + x0]);
    float p10 = static_cast<float>(img[y0 * w + x1]);
    float p01 = static_cast<float>(img[y1 * w + x0]);
    float p11 = static_cast<float>(img[y1 * w + x1]);

    float top = p00 + dx * (p10 - p00);
    float bot = p01 + dx * (p11 - p01);
    float val = top + dy * (bot - top);

    val = fminf(255.0f, fmaxf(0.0f, val));
    return static_cast<uint8_t>(val);
}

__global__ void warpAffineGrayKernel(
    const uint8_t* in, uint8_t* out, int inW, int inH, int outW, int outH, const float* M)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= outW || y >= outH)
        return;

    float sx = M[0] * static_cast<float>(x) + M[1] * static_cast<float>(y) + M[2];
    float sy = M[3] * static_cast<float>(x) + M[4] * static_cast<float>(y) + M[5];

    out[y * outW + x] = bilinearSample(in, inW, inH, sx, sy);
}

} // namespace

namespace cuda_kernels {

void launchRgbToGray(
    const uchar3* d_rgb, uint8_t* d_gray, int width, int height, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    rgbToGrayKernel<<<grid, block, 0, stream>>>(d_rgb, d_gray, width, height);
}

void launchConvolve2D(
    const float* d_in,
    float* d_out,
    int width,
    int height,
    const float* d_kernel,
    int ksize,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    convolve2DKernel<<<grid, block, 0, stream>>>(d_in, d_out, width, height, d_kernel, ksize);
}

void launchWarpAffineGray(
    const uint8_t* d_in,
    uint8_t* d_out,
    int inW,
    int inH,
    int outW,
    int outH,
    const float M[6],
    cudaStream_t stream)
{
    float* d_M = nullptr;
    cudaMalloc(&d_M, 6 * sizeof(float));
    cudaMemcpyAsync(d_M, M, 6 * sizeof(float), cudaMemcpyHostToDevice, stream);

    dim3 block(16, 16);
    dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y);
    warpAffineGrayKernel<<<grid, block, 0, stream>>>(d_in, d_out, inW, inH, outW, outH, d_M);

    cudaFree(d_M);
}

} // namespace cuda_kernels

bool gpuSupportsOperation(TransformOperation operation)
{
    return operation == TransformOperation::GRAYSCALE;
}

bool processTileCuda(Tile& tile)
{
    if (!gpuSupportsOperation(tile.operation)) {
        return false;
    }

    if (tile.data.empty() || tile.width <= 0 || tile.height <= 0) {
        return false;
    }

    const size_t numPixels = static_cast<size_t>(tile.width) * static_cast<size_t>(tile.height);
    const size_t channels = tile.data.size() / numPixels;
    if (channels != 3) {
        return false;
    }

    uchar3* d_rgb = nullptr;
    uint8_t* d_gray = nullptr;

    CUDA_CHECK(cudaMalloc(&d_rgb, numPixels * sizeof(uchar3)));
    CUDA_CHECK(cudaMalloc(&d_gray, numPixels * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_rgb, tile.data.data(), numPixels * sizeof(uchar3), cudaMemcpyHostToDevice));

    cuda_kernels::launchRgbToGray(d_rgb, d_gray, tile.width, tile.height, nullptr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint8_t> gray(numPixels);
    CUDA_CHECK(cudaMemcpy(gray.data(), d_gray, numPixels * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < numPixels; ++i) {
        uint8_t g = gray[i];
        tile.data[i * 3 + 0] = g;
        tile.data[i * 3 + 1] = g;
        tile.data[i * 3 + 2] = g;
    }

    cudaFree(d_gray);
    cudaFree(d_rgb);
    return true;
}

#endif
