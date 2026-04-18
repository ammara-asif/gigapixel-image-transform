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
images/ -> Images sed for testing (.tiff)
include/ → Header files (.h)
src/ → Implementation files (.cpp)
main.cpp → Entry point


---

## Build Instructions

### Compile (Windows / MinGW)

```bash
g++ -Iinclude main.cpp src/TileReader.cpp src/OutputWriter.cpp src/transform.cpp -o main -ltiff -fopenmp -pthread
```

## Run
./main

## Design Overview

TileReader (Producer)
        ↓
Bounded Tile Queue
        ↓
Worker Threads (CPU + Scheduler)
        ↓
GPU / CPU Execution (Heterogeneous)
        ↓
OutputWriter (Streaming output)

## Dependencies

libtiff
OpenMP
pthread
g++ (MinGW or GCC)