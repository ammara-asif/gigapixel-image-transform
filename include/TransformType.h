#pragma once

#include <string>
#include <cstdint>

/**
 * TransformType.h: Extended transform definitions for pipeline composition
 * 
 * Supports all transforms in Milestone 3:
 * - Point operations: GRAYSCALE, COLOR_CORRECTION, IDENTITY
 * - Geometric: ROTATE_90_CW, RESIZE
 * - Filters: FILTER_GAUSSIAN, FILTER_MEDIAN, FILTER_SOBEL
 */

enum class TransformOperation
{
    // Basic operations (Milestone 1-2)
    GRAYSCALE = 0,
    ROTATE_90_CW = 1,
    
    // New Milestone 3 operations
    RESIZE = 2,              // Geometric transform with scaling
    COLOR_CORRECTION = 3,    // Point operation: brightness/contrast
    FILTER_GAUSSIAN = 4,     // Spatial filter: blur
    FILTER_MEDIAN = 5,       // Spatial filter: noise reduction
    FILTER_SOBEL = 6,        // Edge detection filter
    IDENTITY = 7             // No-op (for testing)
};

/**
 * Parameters for transforms that need additional configuration
 * Used in pipelines to pass operation-specific parameters
 */
struct TransformParams
{
    // RESIZE
    float scaleX = 1.0f;
    float scaleY = 1.0f;
    
    // COLOR_CORRECTION
    float brightness = 0.0f;    // -100 to +100
    float contrast = 1.0f;      // 0.5 to 2.0
    
    // FILTER_GAUSSIAN / FILTER_MEDIAN
    int kernelSize = 3;         // Must be odd (3, 5, 7, ...)
    float sigma = 1.0f;         // For Gaussian: standard deviation
    
    // FILTER_SOBEL
    bool sobelX = true;         // Detect X gradients
    bool sobelY = true;         // Detect Y gradients
    
    TransformParams() = default;
};

/**
 * Utility functions for transform information
 */
namespace TransformInfo
{
    /**
     * Check if operation is a point operation (no overlap needed)
     */
    inline bool isPointOperation(TransformOperation op)
    {
        return (op == TransformOperation::GRAYSCALE ||
                op == TransformOperation::COLOR_CORRECTION ||
                op == TransformOperation::IDENTITY);
    }
    
    /**
     * Check if operation is a geometric transform (changes dimensions)
     */
    inline bool isGeometric(TransformOperation op)
    {
        return (op == TransformOperation::ROTATE_90_CW ||
                op == TransformOperation::RESIZE);
    }
    
    /**
     * Check if operation requires overlap region
     */
    inline bool requiresOverlap(TransformOperation op)
    {
        return (op == TransformOperation::FILTER_GAUSSIAN ||
                op == TransformOperation::FILTER_MEDIAN ||
                op == TransformOperation::FILTER_SOBEL);
    }
    
    /**
     * Get human-readable name for operation
     */
    inline std::string getName(TransformOperation op)
    {
        switch (op)
        {
        case TransformOperation::GRAYSCALE:
            return "Grayscale";
        case TransformOperation::ROTATE_90_CW:
            return "Rotate 90° CW";
        case TransformOperation::RESIZE:
            return "Resize";
        case TransformOperation::COLOR_CORRECTION:
            return "Color Correction";
        case TransformOperation::FILTER_GAUSSIAN:
            return "Gaussian Filter";
        case TransformOperation::FILTER_MEDIAN:
            return "Median Filter";
        case TransformOperation::FILTER_SOBEL:
            return "Sobel Edge Detection";
        case TransformOperation::IDENTITY:
            return "Identity";
        default:
            return "Unknown";
        }
    }
    
    /**
     * Estimate computation cost (relative, for scheduling)
     * Higher = more expensive operation
     */
    inline float getComputationCost(TransformOperation op)
    {
        switch (op)
        {
        case TransformOperation::GRAYSCALE:
        case TransformOperation::COLOR_CORRECTION:
        case TransformOperation::IDENTITY:
            return 0.5f;  // Point operations: very cheap
        case TransformOperation::ROTATE_90_CW:
            return 1.0f;  // Geometric: moderate
        case TransformOperation::RESIZE:
            return 1.5f;  // Interpolation: moderate-high
        case TransformOperation::FILTER_GAUSSIAN:
        case TransformOperation::FILTER_MEDIAN:
        case TransformOperation::FILTER_SOBEL:
            return 3.0f;  // Filters: expensive (kernel operations)
        default:
            return 1.0f;
        }
    }
}


