#pragma once

#include "Tile.h"

#ifdef USE_CUDA

// Returns true when a tile operation has a CUDA implementation available.
bool gpuSupportsOperation(TransformOperation operation);

// Executes a tile on CUDA and writes results back into tile.data.
// Returns false when CUDA path is unavailable or execution fails.
bool processTileCuda(Tile& tile);

#else

inline bool gpuSupportsOperation(TransformOperation operation)
{
	return operation == TransformOperation::GRAYSCALE;
}

inline bool processTileCuda(Tile&)
{
	return false;
}

#endif
