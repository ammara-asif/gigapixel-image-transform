#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace cuda_kernels {

void launchRgbToGray(
    const uchar3* d_rgb,
    uint8_t* d_gray,
    int width,
    int height,
    cudaStream_t stream = 0);

void launchConvolve2D(
    const float* d_in,
    float* d_out,
    int width,
    int height,
    const float* d_kernel,
    int ksize,
    cudaStream_t stream = 0);

// Affine matrix is row-major: [m0 m1 m2; m3 m4 m5].
void launchWarpAffineGray(
    const uint8_t* d_in,
    uint8_t* d_out,
    int inW,
    int inH,
    int outW,
    int outH,
    const float M[6],
    cudaStream_t stream = 0);

}
