#pragma once

#include <memory>
#include "IWorker.h"
#include "Tile.h"
#include "FusionOptimizer.h"

/**
 * FusionWorker: Worker implementing cache-optimized fusion for multiple transforms
 *
 * Features:
 * - Cache-line aware blocking strategies
 * - SIMD-friendly memory layouts (aligned buffers, row-major access)
 * - Fused operation execution (e.g., grayscale + rotation in single pass)
 * - Reduced memory traffic through single-pass processing
 * - OpenMP parallelization with proper data alignment
 */
class FusionWorker : public IWorker
{
public:
    FusionWorker();
    ~FusionWorker() = default;

    /**
     * Execute tile with cache-optimized fusion strategies
     */
    void execute(Tile &tile) override;

    /**
     * Get performance metrics from last execution
     */
    FusionOptimizer::PerformanceMetrics getMetrics() const;

    /**
     * Enable/disable fusion for combined operations
     */
    void setFusionEnabled(bool enable) { fusionEnabled = enable; }

    /**
     * Get fusion status
     */
    bool isFusionEnabled() const { return fusionEnabled; }

private:
    std::unique_ptr<FusionOptimizer> optimizer;
    bool fusionEnabled;

    /**
     * Determine if fusion should be used for this tile
     */
    bool shouldUseFusion(const Tile &tile) const;
};
