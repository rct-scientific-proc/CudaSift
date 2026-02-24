#pragma once
#ifndef RAII_GAURDS_HPP
#define RAII_GAURDS_HPP

#include "cudaImage.h"
#include "cusift.h"
#include "cudaSift.h"

// ====== CudaImage RAII Guard ======
class CudaImageGuard
{
private:
    CudaImage img_;
public:
    CudaImageGuard() { CudaImage_init(&img_); }
    ~CudaImageGuard() { CudaImage_destroy(&img_); }

    CudaImage* get() { return &img_; }
    const CudaImage* get() const { return &img_; }
};


// ====== SiftData RAII Guard ======
class SiftDataGuard
{
private:
    SiftData data_;
public:
    SiftDataGuard() {}
    ~SiftDataGuard() { FreeSiftData(&data_); }

    SiftData* get() { return &data_; }
    const SiftData* get() const { return &data_; }
};

// ====== SiftTempMemory RAII Guard ======
class SiftTempMemoryGuard
{
private:
    float* temp_memory_;
public:
    SiftTempMemoryGuard() : temp_memory_(nullptr) {}
    explicit SiftTempMemoryGuard(float* p) : temp_memory_(p) {}
    ~SiftTempMemoryGuard() { FreeSiftTempMemory(temp_memory_); }

    SiftTempMemoryGuard(const SiftTempMemoryGuard&) = delete;
    SiftTempMemoryGuard& operator=(const SiftTempMemoryGuard&) = delete;

    void reset(float* p) { FreeSiftTempMemory(temp_memory_); temp_memory_ = p; }
    float* get() { return temp_memory_; }
    const float* get() const { return temp_memory_; }
};

// ====== Common device pointer RAII Guard ======
template<typename T>
class DevicePtrGuard
{
private:
    T* device_ptr_;
public:
    DevicePtrGuard() : device_ptr_(nullptr) {}
    ~DevicePtrGuard() { if (device_ptr_) cudaFree(device_ptr_); }

    DevicePtrGuard(const DevicePtrGuard&) = delete;
    DevicePtrGuard& operator=(const DevicePtrGuard&) = delete;

    T* get() { return device_ptr_; }
    const T* get() const { return device_ptr_; }

    // We need to return a reference to the pointer for cudaMalloc / cudaMalloc2D
    T*& getRef() { return device_ptr_; }
};

// ====== Host malloc pointer RAII Guard ======
template<typename T>
class HostPtrGuard
{
private:
    T* host_ptr_;
public:
    HostPtrGuard() : host_ptr_(nullptr) {}
    explicit HostPtrGuard(T* p) : host_ptr_(p) {}
    ~HostPtrGuard() { free(host_ptr_); }

    HostPtrGuard(const HostPtrGuard&) = delete;
    HostPtrGuard& operator=(const HostPtrGuard&) = delete;

    void reset(T* p) { free(host_ptr_); host_ptr_ = p; }
    T* get() { return host_ptr_; }
    const T* get() const { return host_ptr_; }
};



#endif /* RAII_GAURDS_HPP */
