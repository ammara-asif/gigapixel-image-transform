#include "CUDAStreamManager.h"
#include <iostream>
#include <stdexcept>
#include <algorithm>

CUDAStreamManager::CUDAStreamManager(int numStreams) 
    : numStreams(numStreams), profilingEnabled(false) {
    
    streams.resize(numStreams, nullptr);
    streamAvailable.resize(numStreams, true);
    streamPurpose.resize(numStreams, StreamPurpose::SYNCHRONIZATION);

    try {
        // Initialize CUDA streams
        // TODO: Actual CUDA stream creation
        // cudaError_t status;
        // for (int i = 0; i < numStreams; i++) {
        //     status = cudaStreamCreate((cudaStream_t*)&streams[i]);
        //     if (status != cudaSuccess) {
        //         throw std::runtime_error("Failed to create CUDA stream");
        //     }
        // }
        
        std::cout << "[CUDAStreamManager] Initialized with " << numStreams << " streams" << std::endl;
        
        // Assign stream purposes for triple buffering
        if (numStreams >= 3) {
            streamPurpose[0] = StreamPurpose::H2D_TRANSFER;
            streamPurpose[1] = StreamPurpose::COMPUTATION;
            streamPurpose[2] = StreamPurpose::D2H_TRANSFER;
        }
    } catch (const std::exception& e) {
        std::cerr << "[CUDAStreamManager] Initialization failed: " << e.what() << std::endl;
        throw;
    }
}

CUDAStreamManager::~CUDAStreamManager() {
    // Clean up CUDA streams
    // TODO: Actual CUDA cleanup
    // for (int i = 0; i < numStreams; i++) {
    //     if (streams[i] != nullptr) {
    //         cudaStreamDestroy((cudaStream_t)streams[i]);
    //     }
    // }
}

void* CUDAStreamManager::getStream(StreamPurpose purpose) {
    // Find an available stream with matching purpose
    for (int i = 0; i < numStreams; i++) {
        if (streamAvailable[i] && streamPurpose[i] == purpose) {
            streamAvailable[i] = false;
            return streams[i];
        }
    }
    
    // If no stream with exact purpose, take any available stream
    for (int i = 0; i < numStreams; i++) {
        if (streamAvailable[i]) {
            streamAvailable[i] = false;
            streamPurpose[i] = purpose;
            return streams[i];
        }
    }
    
    std::cerr << "[CUDAStreamManager] No available streams for purpose " 
              << static_cast<int>(purpose) << std::endl;
    return nullptr;
}

void CUDAStreamManager::releaseStream(int streamId) {
    if (streamId >= 0 && streamId < numStreams) {
        streamAvailable[streamId] = true;
    }
}

int CUDAStreamManager::launchH2DTransfer(void* dst, const void* src, size_t numBytes) {
    void* stream = getStream(StreamPurpose::H2D_TRANSFER);
    if (stream == nullptr) {
        std::cerr << "[CUDAStreamManager] Failed to get stream for H2D transfer" << std::endl;
        return -1;
    }

    if (profilingEnabled) {
        std::cout << "[CUDAStreamManager] Launching H2D transfer: " << numBytes << " bytes" << std::endl;
    }

    // TODO: Actual async H2D transfer
    // cudaMemcpyAsync(dst, src, numBytes, cudaMemcpyHostToDevice, (cudaStream_t)stream);

    return 0; // stream ID
}

int CUDAStreamManager::launchD2DTransfer(void* dst, const void* src, size_t numBytes) {
    void* stream = getStream(StreamPurpose::D2H_TRANSFER);
    if (stream == nullptr) {
        std::cerr << "[CUDAStreamManager] Failed to get stream for D2H transfer" << std::endl;
        return -1;
    }

    if (profilingEnabled) {
        std::cout << "[CUDAStreamManager] Launching D2H transfer: " << numBytes << " bytes" << std::endl;
    }

    // TODO: Actual async D2H transfer
    // cudaMemcpyAsync(dst, src, numBytes, cudaMemcpyDeviceToHost, (cudaStream_t)stream);

    return 0; // stream ID
}

void CUDAStreamManager::synchronizeAllStreams() {
    // TODO: Synchronize all CUDA streams
    // cudaDeviceSynchronize();
    
    std::fill(streamAvailable.begin(), streamAvailable.end(), true);
}

void CUDAStreamManager::synchronizeStream(int streamId) {
    if (streamId < 0 || streamId >= numStreams) {
        return;
    }

    // TODO: Synchronize specific stream
    // cudaStreamSynchronize((cudaStream_t)streams[streamId]);
    
    streamAvailable[streamId] = true;
}

bool CUDAStreamManager::isStreamComplete(int streamId) {
    if (streamId < 0 || streamId >= numStreams) {
        return false;
    }

    // TODO: Check stream status
    // cudaError_t status = cudaStreamQuery((cudaStream_t)streams[streamId]);
    // return status == cudaSuccess;
    
    return streamAvailable[streamId];
}

int CUDAStreamManager::getNumStreams() const {
    return numStreams;
}

void CUDAStreamManager::enableProfiling(bool enable) {
    profilingEnabled = enable;
}

void CUDAStreamManager::checkCudaError(const char* operation) {
    // TODO: Check and report CUDA errors
    // cudaError_t error = cudaGetLastError();
    // if (error != cudaSuccess) {
    //     std::cerr << "[CUDA Error] " << operation << ": " 
    //               << cudaGetErrorString(error) << std::endl;
    // }
}
