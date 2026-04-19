#pragma once

#include <vector>
#include <cstddef>
#include <cstdint>
#include <memory>

/**
 * CUDAStreamManager: Manages CUDA streams for asynchronous data transfer
 * 
 * Supports overlapping of:
 * - H2D transfers (Host-to-Device)
 * - GPU computation
 * - D2H transfers (Device-to-Host)
 * 
 * Triple buffering pattern:
 *   - Stream 1: H2D for tile N+1 while Stream 0 computes on tile N
 *   - Stream 0: GPU computation on tile N
 *   - Stream 2: D2H for tile N-1 while both are busy
 */
class CUDAStreamManager {
public:
    enum class StreamPurpose {
        H2D_TRANSFER,      // Host-to-Device transfers
        COMPUTATION,       // GPU kernel execution
        D2H_TRANSFER,      // Device-to-Host transfers
        SYNCHRONIZATION    // Synchronization operations
    };

    struct StreamInfo {
        int streamId;
        StreamPurpose purpose;
        bool isAvailable;
    };

    /**
     * Initialize CUDA streams for asynchronous operations
     * @param numStreams: Number of concurrent streams (default: 3 for triple buffering)
     */
    explicit CUDAStreamManager(int numStreams = 3);
    ~CUDAStreamManager();

    /**
     * Get an available stream for the specified purpose
     * Returns nullptr if no stream available
     */
    void* getStream(StreamPurpose purpose);

    /**
     * Mark a stream as available after async operation completes
     */
    void releaseStream(int streamId);

    /**
     * Launch async H2D transfer
     * @param dst: Device memory destination
     * @param src: Host memory source
     * @param numBytes: Number of bytes to transfer
     */
    int launchH2DTransfer(void* dst, const void* src, size_t numBytes);

    /**
     * Launch async D2H transfer
     * @param dst: Host memory destination
     * @param src: Device memory source
     * @param numBytes: Number of bytes to transfer
     */
    int launchD2DTransfer(void* dst, const void* src, size_t numBytes);

    /**
     * Synchronize all streams
     */
    void synchronizeAllStreams();

    /**
     * Synchronize a specific stream
     */
    void synchronizeStream(int streamId);

    /**
     * Check if a stream has completed
     */
    bool isStreamComplete(int streamId);

    /**
     * Get number of active streams
     */
    int getNumStreams() const;

    /**
     * Enable/disable profiling
     */
    void enableProfiling(bool enable);

private:
    int numStreams;
    std::vector<void*> streams;           // CUDA stream handles
    std::vector<bool> streamAvailable;    // Availability status
    std::vector<StreamPurpose> streamPurpose;
    bool profilingEnabled;

    // Helper to check CUDA errors
    void checkCudaError(const char* operation);
};
