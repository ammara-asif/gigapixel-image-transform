#ifndef USE_CUDA
#include "CUDAKernels.h"

bool gpuSupportsOperation(TransformOperation) { return false; }
bool processTileCuda(Tile&, uint8_t*, cudaStream_t) { return false; }
#endif