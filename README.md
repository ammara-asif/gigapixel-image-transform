# Gigapixel Image Processing Pipeline (CPU + GPU)

## Overview
This project implements a high-performance tiled image processing pipeline designed for gigapixel images.  
It supports out-of-core processing, parallel CPU execution, and a heterogeneous CPU–GPU scheduling framework.

The system processes images in tiles to avoid loading the full image into memory, enabling scalable processing of extremely large datasets.

---

## Features

- Tiled image reading (memory-efficient processing)
- Parallel CPU-based tile processing
- GPU-aware scheduling (heterogeneous design)
- Support for image transformations:
  - Grayscale conversion
  - 90° rotation (inverse mapping)
- Overlap handling for correct filtering
- Streaming output writer (no full-image memory usage)

---

## Project Structure
├── images/         # Test images (.tiff)
├── include/        # Header files (.h)
│   ├── Tile.h
│   ├── IWorker.h
│   ├── CPUWorker.h
│   ├── GPUWorker.h
│   ├── Scheduler.h
│   ├── TileReader.h
│   ├── BoundedTileQueue.h
│   ├── OutputWriter.h
│   └── Transform.h
├── src/            # Implementation files (.cpp)
│   ├── TileReader.cpp
│   ├── OutputWriter.cpp
│   └── Transform.cpp
└── main.cpp        # Entry point and pipeline orchestration

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
g++ -Iinclude main.cpp src/TileReader.cpp src/OutputWriter.cpp src/Transform.cpp \
    -o main -ltiff -fopenmp -pthread
```

### Run

```bash
./main
```