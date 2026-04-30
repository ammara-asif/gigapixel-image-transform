# Gigapixel Image Transform
Parallel Image Transforms for Gigapixel Images on Heterogeneous Compute

## Overview
This project implements a high-performance tiled image processing pipeline designed for gigapixel images.  
It supports out-of-core processing, parallel CPU execution, and a heterogeneous CPU–GPU scheduling framework.

The system processes images in tiles to avoid loading the full image into memory, enabling scalable processing of extremely large datasets.

---

## Features

- Tiled image reading (memory-efficient processing)
- Parallel CPU-based tile processing
- GPU-aware scheduling (heterogeneous design)
- CUDA kernel implementation for:
  - Color conversion (RGB -> Grayscale)
  - 2D convolution (single-channel float)
  - Geometric affine warp (bilinear sampling)
- Support for image transformations:
  - Grayscale conversion
  - 90° rotation (inverse mapping)
- Overlap handling for correct filtering
- Streaming output writer (no full-image memory usage)

---

## Project Structure

```
gigapixel-pipeline/
├── images/
│   └── input.tiff
├── include/
│   ├── Tile.h
│   ├── IWorker.h
│   ├── CPUWorker.h
│   ├── GPUWorker.h
│   ├── Scheduler.h
│   ├── TileReader.h
│   ├── BoundedTileQueue.h
│   ├── OutputWriter.h
│   ├── Transform.h
│   ├── CUDAKernels.h
│   └── CUDAKernels.cuh
├── src/
│   ├── TileReader.cpp
│   ├── OutputWriter.cpp
│   ├── transform.cpp
│   ├── CUDAKernels.cu
│   └── CUDAKernelsStub.cpp
└── main.cpp
```

---

## Build Instructions

### Requirements

| Dependency | Purpose |
|---|---|
| `g++` (GCC / MinGW) | Compiler |
| `libtiff` | TIFF image I/O |
| `OpenMP` | Parallel processing |
| `pthread` | Thread management |

### Compile (Linux / WSL / MinGW)

```bash
g++ -Iinclude main.cpp src/TileReader.cpp src/OutputWriter.cpp src/transform.cpp \
    -o main -ltiff -fopenmp -pthread
```

### Optional CUDA Build (Milestone 2)

Compile CUDA kernels with `nvcc` and enable `USE_CUDA`:

```bash
nvcc -std=c++17 -DUSE_CUDA -Iinclude -c src/CUDAKernels.cu -o CUDAKernels.o
g++ -DUSE_CUDA -Iinclude main.cpp src/TileReader.cpp src/OutputWriter.cpp src/transform.cpp \
  CUDAKernels.o -o main -ltiff -fopenmp -pthread -lcudart
```

### Run

```bash
./main
```
