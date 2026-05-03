#include "PipelineExecutor.h"
#include "Transform.h"
#include <algorithm>
#include <cmath>
#include <chrono>

void PipelineExecutor::execute(Tile& tile, 
                               const std::shared_ptr<TransformPipeline>& pipeline,
                               PipelineStats* stats)
{
    if (!pipeline || pipeline->isEmpty())
    {
        return;
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    // Collect initial tile info for stats
    size_t inputPixels = tile.width * tile.height;

    // Execute each stage in sequence
    processTilePipeline(tile, pipeline);

    // Calculate output pixels for stats
    size_t outputPixels = tile.width * tile.height;

    // Collect statistics if requested
    if (stats)
    {
        auto endTime = std::chrono::high_resolution_clock::now();
        stats->executionTime = std::chrono::duration_cast<std::chrono::milliseconds>(
            endTime - startTime);
        stats->inputPixels = inputPixels;
        stats->outputPixels = outputPixels;
        stats->stageCount = static_cast<int>(pipeline->getStageCount());
        stats->wasGPUAccelerated = false;  // Set by scheduler if needed
        stats->calculateThroughput();
    }
}

void PipelineExecutor::calculateOutputDimensions(int inputWidth, int inputHeight,
                                                const std::shared_ptr<TransformPipeline>& pipeline,
                                                int& outputWidth, int& outputHeight)
{
    outputWidth = inputWidth;
    outputHeight = inputHeight;

    if (!pipeline)
        return;

    for (size_t i = 0; i < pipeline->getStageCount(); ++i)
    {
        const auto& stage = pipeline->getStage(i);

        switch (stage.operation)
        {
        case TransformOperation::ROTATE_90_CW:
            // 90 degree rotation swaps dimensions
            std::swap(outputWidth, outputHeight);
            break;

        case TransformOperation::RESIZE:
        {
            // Apply scaling factors
            float scaleX = stage.params.scaleX;
            float scaleY = stage.params.scaleY;
            outputWidth = static_cast<int>(outputWidth * scaleX);
            outputHeight = static_cast<int>(outputHeight * scaleY);
            
            // Ensure at least 1 pixel
            outputWidth = std::max(1, outputWidth);
            outputHeight = std::max(1, outputHeight);
            break;
        }

        case TransformOperation::GRAYSCALE:
        case TransformOperation::COLOR_CORRECTION:
        case TransformOperation::FILTER_GAUSSIAN:
        case TransformOperation::FILTER_MEDIAN:
        case TransformOperation::FILTER_SOBEL:
        case TransformOperation::IDENTITY:
            // These operations don't change dimensions
            break;
        }
    }
}

bool PipelineExecutor::canExecuteOnGPU(const std::shared_ptr<TransformPipeline>& pipeline)
{
    if (!pipeline)
        return false;

    // GPU can handle point operations and simple pipelines
    // Currently, we prioritize CPU for filters due to memory complexity
    for (size_t i = 0; i < pipeline->getStageCount(); ++i)
    {
        const auto& stage = pipeline->getStage(i);
        
        // Filters are difficult on GPU in this implementation
        if (stage.operation == TransformOperation::FILTER_GAUSSIAN ||
            stage.operation == TransformOperation::FILTER_MEDIAN ||
            stage.operation == TransformOperation::FILTER_SOBEL)
        {
            return false;
        }

        // Rotation has poor GPU coalescing
        if (stage.operation == TransformOperation::ROTATE_90_CW)
        {
            return false;
        }
    }

    return true;
}

size_t PipelineExecutor::getMemoryRequirement(int inputWidth, int inputHeight,
                                             const std::shared_ptr<TransformPipeline>& pipeline)
{
    if (!pipeline)
        return 0;

    size_t totalMemory = inputWidth * inputHeight * 3;  // Base RGB
    int currentWidth = inputWidth;
    int currentHeight = inputHeight;

    // Account for intermediate buffers needed by geometric transforms
    for (size_t i = 0; i < pipeline->getStageCount(); ++i)
    {
        const auto& stage = pipeline->getStage(i);

        if (stage.operation == TransformOperation::ROTATE_90_CW)
        {
            // Rotation needs temporary buffer (swaps dimensions)
            std::swap(currentWidth, currentHeight);
            totalMemory += currentWidth * currentHeight * 3;
        }
        else if (stage.operation == TransformOperation::RESIZE)
        {
            // Resize needs output buffer
            float scaleX = stage.params.scaleX;
            float scaleY = stage.params.scaleY;
            int newWidth = static_cast<int>(currentWidth * scaleX);
            int newHeight = static_cast<int>(currentHeight * scaleY);
            newWidth = std::max(1, newWidth);
            newHeight = std::max(1, newHeight);
            
            totalMemory += newWidth * newHeight * 3;
            
            currentWidth = newWidth;
            currentHeight = newHeight;
        }
    }

    return totalMemory;
}
