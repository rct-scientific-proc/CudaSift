#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#include <cstdio>
#include <iostream>
#include <string>
#include <stdexcept>
#include <sstream>

#ifdef _WIN32
#include <intrin.h>
#endif

// Exception type used internally by the library.  Carries the originating
// source file/line and (when applicable) the underlying CUDA error code so
// the public API guard can populate the thread-local error state without
// re-parsing message strings.
struct CusiftError : public std::runtime_error
{
    std::string file;
    int         line;
    cudaError_t cuda_error;

    CusiftError(std::string f, int l, const std::string &msg, cudaError_t ce = cudaSuccess)
        : std::runtime_error(msg), file(std::move(f)), line(l), cuda_error(ce) {}
};

#define safeCall(err) cusift_safe_call(err, __FILE__, __LINE__)
#define safeThreadSync() cusift_safe_thread_sync(__FILE__, __LINE__)
#define checkMsg(msg) cusift_check_msg(msg, __FILE__, __LINE__)

inline void cusift_safe_call(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        std::ostringstream oss;
        oss << "CUDA error: " << cudaGetErrorString(err);
        throw CusiftError(file, line, oss.str(), err);
    }
}

inline void cusift_safe_thread_sync(const char *file, const int line)
{
    cudaError err = cudaDeviceSynchronize();
    if (cudaSuccess != err)
    {
        std::ostringstream oss;
        oss << "CUDA error during cudaDeviceSynchronize(): " << cudaGetErrorString(err);
        throw CusiftError(file, line, oss.str(), err);
    }
}

inline void cusift_check_msg(const char *errorMessage, const char *file, const int line)
{
    // Use cudaGetLastError() (not cudaPeekAtLastError()) so the sticky
    // last-error flag is cleared after we observe it; otherwise every
    // subsequent checkMsg() would keep reporting the same stale error.
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        std::ostringstream oss;
        oss << "CUDA error: " << errorMessage << " : " << cudaGetErrorString(err);
        throw CusiftError(file, line, oss.str(), err);
    }
}

inline bool deviceInit(int dev)
{
    int deviceCount;
    safeCall(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0)
    {
        fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
        return false;
    }
    if (dev < 0)
        dev = 0;
    if (dev > deviceCount - 1)
        dev = deviceCount - 1;
    cudaDeviceProp deviceProp;
    safeCall(cudaGetDeviceProperties(&deviceProp, dev));
    if (deviceProp.major < 1)
    {
        fprintf(stderr, "error: device does not support CUDA.\n");
        return false;
    }
    safeCall(cudaSetDevice(dev));
    return true;
}

template <class T>
__device__ __inline__ T ShiftDown(T var, unsigned int delta, int width = 32)
{
#if (CUDART_VERSION >= 9000)
    return __shfl_down_sync(0xffffffff, var, delta, width);
#else
    return __shfl_down(var, delta, width);
#endif
}

template <class T>
__device__ __inline__ T ShiftUp(T var, unsigned int delta, int width = 32)
{
#if (CUDART_VERSION >= 9000)
    return __shfl_up_sync(0xffffffff, var, delta, width);
#else
    return __shfl_up(var, delta, width);
#endif
}

template <class T>
__device__ __inline__ T Shuffle(T var, unsigned int lane, int width = 32)
{
#if (CUDART_VERSION >= 9000)
    return __shfl_sync(0xffffffff, var, lane, width);
#else
    return __shfl(var, lane, width);
#endif
}

#endif
