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

#define safeCall(err) __safeCall(err, __FILE__, __LINE__)
#define safeThreadSync() __safeThreadSync(__FILE__, __LINE__)
#define checkMsg(msg) __checkMsg(msg, __FILE__, __LINE__)

inline void __safeCall(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        // Throw a runtime error with the error message, line number, and filename
        std::ostringstream oss;
        oss << "CUDA error in file '" << file << "' in line " << line << " : " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

inline void __safeThreadSync(const char *file, const int line)
{
    cudaError err = cudaDeviceSynchronize();
    if (cudaSuccess != err)
    {
        std::ostringstream oss;
        oss << "CUDA error during cudaDeviceSynchronize() in file '" << file << "' in line " << line << " : " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

inline void __checkMsg(const char *errorMessage, const char *file, const int line)
{
    cudaError_t err = cudaPeekAtLastError();
    if (cudaSuccess != err)
    {
        std::ostringstream oss;
        oss << "CUDA error: " << errorMessage << " in file '" << file << "' in line " << line << " : " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
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
