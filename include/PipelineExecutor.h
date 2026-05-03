#pragma once

#include <memory>
#include <chrono>
#include "Tile.h"
#include "PipelineComposition.h"

/**
 * PipelineExecutor.h: Execution engine for multi-stage pipelines
 * 
 * Handles:
 * - Sequential execution of pipeline stages
 * - Memory management (intermediate buffers for geometric transforms)
 * - Output dimension calculation
 * - Performance tracking and statistics
 * - Device-specific optimization hints
 */

/**
 * Statistics for pipeline execution
 * Useful for performance analysis and benchmarking
 */
struct PipelineStats
{
    std::chrono::milliseconds executionTime{0};
    size_t inputPixels = 0;
    size_t outputPixels = 0;
    int stageCount = 0;
    bool wasGPUAccelerated = false;
    float throughputMPixelsPerSec = 0.0f;
    
    /**
     * Calculate throughput (megapixels/second)
     */
    void calculateThroughput()
    {
        if (executionTime.count() > 0)
        {
            throughputMPixelsPerSec = static_cast<float>(outputPixels) / 
                                      (executionTime.count() * 1000.0f);
        }
    }
};

/**
 * PipelineExecutor: Executes multi-stage transformation pipelines
 * 
 * Features:
 * - Handles memory allocation for intermediate results
 * - Manages output dimension calculations
 * - Provides statistics collection
 * - Device-aware optimization decisions
 */
class PipelineExecutor
{
public:
    /**
     * Execute pipeline on a tile
     * 
     * @param tile Input/Output tile (modified in-place where possible)
     * @param pipeline The pipeline to execute
     * @param stats Output parameter for execution statistics
     */
    static void execute(Tile& tile, 
                       const std::shared_ptr<TransformPipeline>& pipeline,
                       PipelineStats* stats = nullptr);

    /**
     * Calculate output dimensions after pipeline execution
     * 
     * @param inputWidth Input tile width
     * @param inputHeight Input tile height
     * @param pipeline Pipeline to analyze
     * @param outputWidth Output width (result)
     * @param outputHeight Output height (result)
     */
    static void calculateOutputDimensions(int inputWidth, int inputHeight,
                                         const std::shared_ptr<TransformPipeline>& pipeline,
                                         int& outputWidth, int& outputHeight);

    /**
     * Calculate required overlap region for pipeline
     * 
     * @param pipeline Pipeline to analyze
     * @return Overlap in pixels
     */
    static int calculateRequiredOverlap(const std::shared_ptr<TransformPipeline>& pipeline)
    {
        return pipeline->computeRequiredOverlap();
    }

    /**
     * Check if pipeline can be executed entirely on GPU
     * (May not be possible for all operation types)
     * 
     * @param pipeline Pipeline to analyze
     * @return true if GPU-compatible
     */
    static bool canExecuteOnGPU(const std::shared_ptr<TransformPipeline>& pipeline);

    /**
     * Get memory requirements for executing pipeline
     * (Accounts for intermediate buffers)
     * 
     * @param inputWidth Input tile width
     * @param inputHeight Input tile height  
     * @param pipeline Pipeline to analyze
     * @return Memory required in bytes
     */
    static size_t getMemoryRequirement(int inputWidth, int inputHeight,
                                      const std::shared_ptr<TransformPipeline>& pipeline);
};

