#pragma once
#include <atomic>
#include "CPUWorker.h"
#include "GPUWorker.h"


class Scheduler {
private:
    CPUWorker cpu;
    GPUWorker gpu;

    std::atomic<int> gpuQueueDepth{0};  // lightweight proxy for GPU load
    static constexpr int GPU_OVERLOAD_THRESHOLD = 8; // tune based on GPU

    DeviceType decide_device(const Tile& tile) {

        int pixel_count = tile.width * tile.height;
         // Rule 1: Small tiles always go to CPU (GPU launch overhead not worth it)
        if (pixel_count < 128 * 128)
            return DeviceType::CPU;

        // Rule 2: Irregular access patterns → CPU
        // (Rotation involves non-sequential memory access, bad for GPU coalescing)
        if (tile.operation == TransformOperation::ROTATE_90_CW)
            return DeviceType::CPU;

        // Rule 3: Dense arithmetic on large tiles → GPU if not overloaded
        if (tile.operation == TransformOperation::GRAYSCALE) {
            if (gpuQueueDepth.load() < GPU_OVERLOAD_THRESHOLD)
                return DeviceType::GPU;
        }

        // Fallback
        return DeviceType::CPU;
    }
    
public:

    void dispatch(Tile& task) {
        DeviceType device = decide_device(task);

        //atomic->thread safe
        if (device == DeviceType::GPU) {
            gpuQueueDepth.fetch_add(1);
            gpu.execute(task);
            gpuQueueDepth.fetch_sub(1);
        } else {
            cpu.execute(task);
        }
    }
};