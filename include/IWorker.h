#pragma once
#include "Tile.h"

enum class DeviceType {
    CPU,
    GPU
};

class IWorker {
public:
    virtual void execute(Tile& task) = 0;
    virtual ~IWorker() = default;
};
