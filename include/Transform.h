#pragma once

#include "Tile.h"
#include "PipelineComposition.h"
#include <memory>

// Applies a transformation algorithm to a single image tile.
// Modifies the tile.data buffer in-place or in temporary buffers as needed.
void processTile(Tile &tile);

/**
 * Execute a single transform stage on a tile
 * Returns temporary buffer if output dimensions differ from input
 * (for geometric transforms), otherwise modifies in-place
 */
std::shared_ptr<uint8_t> processTransformStage(
    Tile& tile,
    const PipelineStage& stage,
    bool needsOutputBuffer = false);

/**
 * Execute a complete pipeline on a tile
 * Handles intermediate buffer management and dimension changes
 */
void processTilePipeline(Tile& tile, const std::shared_ptr<TransformPipeline>& pipeline);