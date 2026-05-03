# Gigapixel Image Transform
Parallel Image Transforms for Gigapixel Images on Heterogeneous Compute

## Overview
This project implements a high-performance tiled image processing pipeline designed for gigapixel images.  
It supports out-of-core processing, parallel CPU execution, and a heterogeneous CPU–GPU scheduling framework.

The system processes images in tiles to avoid loading the full image into memory, enabling scalable processing of extremely large datasets.

---

## Features

- Tiled image reading (memory-efficient, out-of-core processing)
- Parallel CPU-based tile processing with cache hierarchy optimization (L1/L2/L3-aware)
- GPU-aware heterogeneous scheduling (CPU vs GPU decision per tile)
- Pre-allocated VRAM pool (`VramManager`) with condition-variable blocking for zero per-tile `cudaMalloc`
- CUDA stream pool (`CUDAStreamManager`) for async H2D / kernel / D2H overlap
- CUDA kernel implementation for:
  - Color conversion (RGB → grayscale)
  - 2D convolution (single-channel float)
  - Geometric affine warp (bilinear sampling)
- Support for image transformations:
  - Grayscale conversion
  - 90° clockwise rotation (inverse mapping)
- Overlap handling for correct boundary conditions in filter operations
- Streaming output writer — no full-image memory materialization
- Pinned (page-locked) host memory on `Tile` for fast async transfers

---

## Project Structure

```
gigapixel-pipeline/
├── images/
│   └── input.tiff
├── AsyncGPUWorker.cpp / .h
├── BoundedTileQueue.h
├── CPUWorker.h
├── CUDAKernels.cu / .h / .cuh
├── CUDAKernelsStub.cpp
├── CUDAStreamManager.cpp / .h
├── GPUWorker.h
├── IWorker.h
├── main.cpp
├── OptimizedCPUWorker.cpp / .h
├── OutputWriter.cpp / .h
├── Scheduler.h
├── Tile.h
├── TileOptimizer.cpp / .h
├── TileReader.cpp / .h
├── tiffio.h / tiffio_mock.cpp
├── Transform.h
├── transform.cpp
└── VramManager.cpp / .h
```

---

## Running on GPU (recommended) — Google Colab

The primary way to run this project is on Google Colab with a T4 GPU.  
All dependencies (CUDA 12.8, `libtiff`) are available on the Colab runtime with no extra setup beyond one `apt-get` line.

### Step 1 — Open Colab and set runtime

Open the notebook `Run_on_GPU.ipynb` in Google Colab.  
Go to **Runtime → Change runtime type → T4 GPU** before running any cells.

### Step 2 — Verify GPU

```python
!nvidia-smi
!nvcc --version
```

Expected: Tesla T4, CUDA 12.8.

### Step 3 — Install libtiff

```python
!apt-get install -y libtiff-dev libomp-dev 2>&1 | tail -5
```

> **Important:** Colab runtimes reset when they disconnect. If you reconnect and get linker errors (`undefined reference to TIFFOpen`), re-run this cell before compiling.

### Step 4 — Upload source files

Run the upload cell and select all `.cpp`, `.cu`, `.h`, and `.cuh` files from the project.  
All files are saved flat into `/content/src/`.

### Step 5 — Compile

```bash
cd /content/src && nvcc \
  -std=c++17 \
  -O2 \
  -DUSE_CUDA \
  -arch=sm_75 \
  -Xcompiler -fopenmp \
  -I. \
  main.cpp \
  TileReader.cpp \
  transform.cpp \
  OptimizedCPUWorker.cpp \
  AsyncGPUWorker.cpp \
  CUDAStreamManager.cpp \
  VramManager.cpp \
  TileOptimizer.cpp \
  OutputWriter.cpp \
  CUDAKernels.cu \
  -L/usr/lib/x86_64-linux-gnu -ltiff \
  -lpthread \
  -lgomp \
  -o /content/gigapixel_pipeline
```

Expected output: `Build OK` with a ~1.1 MB binary at `/content/gigapixel_pipeline`.

> The `-L/usr/lib/x86_64-linux-gnu` flag is required on Colab because `libtiff.so` lives there rather than in the default linker search path.

### Step 6 — Run

```python
import subprocess
result = subprocess.run(['/content/gigapixel_pipeline'], input='1\n', text=True, capture_output=True)
# input '1' = grayscale, '2' = rotate 90 CW
```

Or interactively:

```bash
cd /content && ./gigapixel_pipeline
```

The pipeline prompts:
```
1. Grayscale (Point Operation)
2. Rotate 90 CW (Geometric Inverse Map)
Select an operation (1 or 2):
```

### Observed results (4096×4096 RGB, T4 GPU, CUDA 12.8)

| Operation | Time | Throughput |
|---|---|---|
| Grayscale | 0.62 s | 26.9 MP/s |
| Rotate 90 CW | 0.65 s | 26.0 MP/s |

Output is written to `/content/output.tiff`.

---

## Build Requirements

| Dependency | Version (tested) | Purpose |
|---|---|---|
| `nvcc` | CUDA 12.8 | CUDA compiler |
| `libtiff-dev` | 4.3.0 | TIFF image I/O |
| `libomp-dev` | 14.0 | OpenMP (writer thread) |
| `libpthread` | system | Worker threads |
| `libgomp` | system | OpenMP runtime |

---

## Pipeline Architecture

```
main thread (reader)
  └─ TileReader::getTile()          reads tiles from disk into pinned memory
       └─ BoundedTileQueue::push()  blocks when queue is full

worker threads (numCores - 2)
  └─ BoundedTileQueue::pop()        blocks when queue is empty
       └─ VramManager::acquireBuffer()  blocks until a VRAM buffer is free
            └─ Scheduler::dispatch()
                 ├─ OptimizedCPUWorker  (small tiles, rotation)
                 └─ AsyncGPUWorker      (large tiles, grayscale)
                      └─ CUDAStreamManager: H2D → kernel → D2H → sync
       └─ VramManager::releaseBuffer()
       └─ TiledOutputWriter::submit_tile()

writer thread (dedicated)
  └─ TiledOutputWriter::run()       strips overlap, pads, calls TIFFWriteTile
```
