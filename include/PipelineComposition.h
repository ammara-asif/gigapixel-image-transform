#pragma once

#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include "TransformType.h"

/**
 * PipelineComposition.h: Pipeline builder and composition API
 * 
 * Allows creation of multi-stage processing pipelines where transforms
 * are executed sequentially on the same tile.
 * 
 * Example:
 *   auto pipeline = std::make_shared<TransformPipeline>("my_pipeline");
 *   pipeline->add(ROTATE_90_CW)
 *           .add(RESIZE, {scaleX: 0.5, scaleY: 0.5})
 *           .add(GRAYSCALE);
 */

/**
 * Single stage in a pipeline: operation + parameters
 */
struct PipelineStage
{
    TransformOperation operation;
    TransformParams params;
    
    PipelineStage(TransformOperation op, const TransformParams& p = TransformParams())
        : operation(op), params(p) {}
};

/**
 * TransformPipeline: Represents a sequence of transforms
 * 
 * Thread-safe for reading after construction.
 * Fluent builder interface for easy composition.
 */
class TransformPipeline
{
private:
    std::string name;
    std::vector<PipelineStage> stages;
    
    // Cached properties
    mutable bool requiresOverlapCache = false;
    mutable int maxKernelSizeCache = 0;
    mutable bool isCached = false;

public:
    /**
     * Create named pipeline
     */
    explicit TransformPipeline(const std::string& pipelineName = "unnamed")
        : name(pipelineName) {}

    /**
     * Add a transform stage to the pipeline
     * Returns reference to self for fluent interface
     */
    TransformPipeline& add(TransformOperation op, const TransformParams& params = TransformParams())
    {
        stages.emplace_back(op, params);
        isCached = false;  // Invalidate cache
        return *this;
    }

    /**
     * Get pipeline name
     */
    const std::string& getName() const { return name; }

    /**
     * Get number of stages in pipeline
     */
    size_t getStageCount() const { return stages.size(); }

    /**
     * Check if pipeline is empty
     */
    bool isEmpty() const { return stages.empty(); }

    /**
     * Get stage at index
     */
    const PipelineStage& getStage(size_t index) const
    {
        if (index >= stages.size())
            throw std::out_of_range("Pipeline stage index out of range");
        return stages[index];
    }

    /**
     * Get all stages
     */
    const std::vector<PipelineStage>& getStages() const { return stages; }

    /**
     * Check if any stage requires overlap region
     * Used for boundary handling
     */
    bool requiresOverlapRegion() const
    {
        if (!isCached) updateCache();
        return requiresOverlapCache;
    }

    /**
     * Get maximum kernel size needed across all filter operations
     * Used to calculate appropriate overlap region
     */
    int getMaxKernelSize() const
    {
        if (!isCached) updateCache();
        return maxKernelSizeCache;
    }

    /**
     * Compute required overlap region (in pixels) for boundary correctness
     * For filters: overlap = (maxKernelSize - 1) / 2
     */
    int computeRequiredOverlap() const
    {
        int maxKernel = getMaxKernelSize();
        return (maxKernel > 0) ? (maxKernel - 1) / 2 : 0;
    }

    /**
     * Check if pipeline is "simple" (all point operations)
     * Allows optimization opportunities (fusion, in-place processing)
     */
    bool isSimplePointOperation() const
    {
        if (isEmpty()) return false;
        for (const auto& stage : stages)
        {
            if (!TransformInfo::isPointOperation(stage.operation))
                return false;
        }
        return true;
    }

    /**
     * Get total computation cost estimate (for scheduling decisions)
     * Higher = more expensive to process
     */
    float getTotalCost() const
    {
        float cost = 0.0f;
        for (const auto& stage : stages)
        {
            cost += TransformInfo::getComputationCost(stage.operation);
        }
        return cost;
    }

    /**
     * Get human-readable description of pipeline
     */
    std::string describe() const
    {
        if (isEmpty())
            return "Empty pipeline";

        std::string desc;
        for (size_t i = 0; i < stages.size(); ++i)
        {
            desc += TransformInfo::getName(stages[i].operation);
            if (i < stages.size() - 1)
                desc += " → ";
        }
        return desc;
    }

    /**
     * Clone this pipeline
     */
    std::shared_ptr<TransformPipeline> clone() const
    {
        auto cloned = std::make_shared<TransformPipeline>(name + "_clone");
        for (const auto& stage : stages)
        {
            cloned->add(stage.operation, stage.params);
        }
        return cloned;
    }

private:
    /**
     * Update cached properties
     */
    void updateCache() const
    {
        requiresOverlapCache = false;
        maxKernelSizeCache = 0;

        for (const auto& stage : stages)
        {
            if (TransformInfo::requiresOverlap(stage.operation))
            {
                requiresOverlapCache = true;
                maxKernelSizeCache = std::max(maxKernelSizeCache, stage.params.kernelSize);
            }
        }
        isCached = true;
    }
};

/**
 * Helper class: PipelineComposer
 * Pre-built pipelines for common use cases
 */
namespace PipelineComposer
{
    /**
     * Rotate → Resize → Grayscale
     */
    inline std::shared_ptr<TransformPipeline> rotateResizeGrayscale(float scaleX = 0.5f, float scaleY = 0.5f)
    {
        auto pipeline = std::make_shared<TransformPipeline>("rotate_resize_grayscale");
        TransformParams resizeParams;
        resizeParams.scaleX = scaleX;
        resizeParams.scaleY = scaleY;

        pipeline->add(TransformOperation::ROTATE_90_CW)
                .add(TransformOperation::RESIZE, resizeParams)
                .add(TransformOperation::GRAYSCALE);
        return pipeline;
    }

    /**
     * Rotate → Gaussian Filter → Color Correction
     */
    inline std::shared_ptr<TransformPipeline> rotateFilterColorCorrect(int kernelSize = 5, float sigma = 1.0f)
    {
        auto pipeline = std::make_shared<TransformPipeline>("rotate_filter_colorcorrect");
        TransformParams gaussParams;
        gaussParams.kernelSize = kernelSize;
        gaussParams.sigma = sigma;

        TransformParams colorParams;
        colorParams.brightness = 10.0f;
        colorParams.contrast = 1.1f;

        pipeline->add(TransformOperation::ROTATE_90_CW)
                .add(TransformOperation::FILTER_GAUSSIAN, gaussParams)
                .add(TransformOperation::COLOR_CORRECTION, colorParams);
        return pipeline;
    }

    /**
     * Gaussian Filter → Sobel Edge Detection
     */
    inline std::shared_ptr<TransformPipeline> edgeDetectionPipeline()
    {
        auto pipeline = std::make_shared<TransformPipeline>("edge_detection");
        TransformParams gaussParams;
        gaussParams.kernelSize = 3;
        gaussParams.sigma = 0.5f;

        pipeline->add(TransformOperation::FILTER_GAUSSIAN, gaussParams)
                .add(TransformOperation::FILTER_SOBEL);
        return pipeline;
    }

    /**
     * Median Filter → Gaussian Filter (noise reduction)
     */
    inline std::shared_ptr<TransformPipeline> noiseReductionPipeline()
    {
        auto pipeline = std::make_shared<TransformPipeline>("noise_reduction");
        TransformParams medianParams;
        medianParams.kernelSize = 5;

        TransformParams gaussParams;
        gaussParams.kernelSize = 3;
        gaussParams.sigma = 0.5f;

        pipeline->add(TransformOperation::FILTER_MEDIAN, medianParams)
                .add(TransformOperation::FILTER_GAUSSIAN, gaussParams);
        return pipeline;
    }

    /**
     * Simple Grayscale (single stage, for comparison)
     */
    inline std::shared_ptr<TransformPipeline> simpleSingleGrayscale()
    {
        auto pipeline = std::make_shared<TransformPipeline>("single_grayscale");
        pipeline->add(TransformOperation::GRAYSCALE);
        return pipeline;
    }
}


