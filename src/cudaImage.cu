//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//

#include <cstdio>
#include <cfloat>

#include "cudautils.h"
#include "cudaImage.h"
#include "RAII_Guards.hpp"

int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }
int iDivDown(int a, int b) { return a / b; }
int iAlignUp(int a, int b) { return (a % b != 0) ? (a - a % b + b) : a; }
int iAlignDown(int a, int b) { return a - a % b; }

// Keep
void CudaImage_init(CudaImage *img)
{
    img->width = 0;
    img->height = 0;
    img->pitch = 0;
    img->h_data = NULL;
    img->d_data = NULL;
    img->t_data = NULL;
    img->d_internalAlloc = false;
    img->h_internalAlloc = false;
}

void CudaImage_destroy(CudaImage *img)
{
    // Use cudaFree directly (not safeCall) so this function is safe
    // to call from destructors during stack unwinding.
    if (img->d_internalAlloc && img->d_data != NULL)
        cudaFree(img->d_data);
    img->d_data = NULL;
    if (img->h_internalAlloc && img->h_data != NULL)
        free(img->h_data);
    img->h_data = NULL;
    if (img->t_data != NULL)
        cudaFreeArray((cudaArray *)img->t_data);
    img->t_data = NULL;
}

void CudaImage_Allocate(CudaImage *img, int w, int h, int p, bool host, float *devmem, float *hostmem)
{
    img->width = w;
    img->height = h;
    img->pitch = p;
    img->d_data = devmem;
    img->h_data = hostmem;
    img->t_data = NULL;
    if (devmem == NULL)
    {
        size_t pitch;
        safeCall(cudaMallocPitch((void **)&img->d_data, &pitch, (size_t)(sizeof(float) * img->width), (size_t)img->height));
        img->pitch = (int)(pitch / sizeof(float));
        if (img->d_data == NULL)
            printf("Failed to allocate device data\n");
        img->d_internalAlloc = true;
    }
    if (host && hostmem == NULL)
    {
        img->h_data = (float *)malloc(sizeof(float) * img->pitch * img->height);
        img->h_internalAlloc = true;
    }
}

// Keep
void CudaImage_Download(CudaImage *img)
{
    int p = sizeof(float) * img->pitch;
    //if (img->d_data != NULL && img->h_data != NULL)
    safeCall(cudaMemcpy2D(img->d_data, p, img->h_data, sizeof(float) * img->width, sizeof(float) * img->width, img->height, cudaMemcpyDefault));
}

void CudaImage_Readback(CudaImage *img)
{
    int p = sizeof(float) * img->pitch;
    safeCall(cudaMemcpy2D(img->h_data, sizeof(float) * img->width, img->d_data, p, sizeof(float) * img->width, img->height, cudaMemcpyDeviceToHost));
}

///////////////////////////////////////////////////////////////////////////////
// Min-Max normalization: maps pixel values to [0, 1]
///////////////////////////////////////////////////////////////////////////////

// Ordered float atomics.  atomicMin/atomicMax on the raw int bits only order
// non-negative floats correctly, and the caller reads these back as plain
// floats, so a monotonic bit-flip encoding would have to be undone on the host.
// A CAS loop is cheap here: it runs once per block, not once per pixel.
__device__ __forceinline__ void atomicMinFloat(float *addr, float value)
{
    int *as_int = (int *)addr;
    int old = *as_int, assumed;
    do
    {
        assumed = old;
        if (__int_as_float(assumed) <= value)
            break;
        old = atomicCAS(as_int, assumed, __float_as_int(value));
    } while (assumed != old);
}

__device__ __forceinline__ void atomicMaxFloat(float *addr, float value)
{
    int *as_int = (int *)addr;
    int old = *as_int, assumed;
    do
    {
        assumed = old;
        if (__int_as_float(assumed) >= value)
            break;
        old = atomicCAS(as_int, assumed, __float_as_int(value));
    } while (assumed != old);
}

__global__ void MinMaxKernel(float *d_data, int width, int pitch, int height, float *d_min, float *d_max)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Out-of-range lanes carry the identity element, which removes the need to
    // track per-lane validity through the reduction.
    const bool inRange = (x < width && y < height);
    const float v = inRange ? d_data[(size_t)y * pitch + x] : 0.0f;
    float mn = inRange ? v : FLT_MAX;
    float mx = inRange ? v : -FLT_MAX;

    // Warp reduction.  Every thread of the block reaches this, so the full mask
    // is valid (the previous version called __any_sync from inside a
    // lane-0-only branch, which is undefined).
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        mn = fminf(mn, __shfl_down_sync(0xffffffff, mn, offset));
        mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, offset));
    }

    const int lane = threadIdx.x + threadIdx.y * blockDim.x;
    const int warpId = lane >> 5;
    const int laneInWarp = lane & 31;
    const int numWarps = (blockDim.x * blockDim.y + 31) / 32;

    __shared__ float sMin[32]; // up to 1024 threads = 32 warps
    __shared__ float sMax[32];
    if (laneInWarp == 0)
    {
        sMin[warpId] = mn;
        sMax[warpId] = mx;
    }
    __syncthreads();

    // Fold the per-warp results in one thread.  The previous tree reduction
    // read sMin[lane + offset] but never wrote the partial results back to
    // shared memory, so all but a few warps were silently discarded.
    if (lane == 0)
    {
        float bmn = sMin[0];
        float bmx = sMax[0];
        for (int w = 1; w < numWarps; w++)
        {
            bmn = fminf(bmn, sMin[w]);
            bmx = fmaxf(bmx, sMax[w]);
        }
        if (bmn <= bmx) // false only when the block covered no valid pixel
        {
            atomicMinFloat(d_min, bmn);
            atomicMaxFloat(d_max, bmx);
        }
    }
}

__global__ void NormalizeKernel(float *d_data, int width, int pitch, int height, float minVal, float range)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        int idx = y * pitch + x;
        d_data[idx] = (d_data[idx] - minVal) * range;
    }
}

void CudaImage_Normalize(CudaImage *img)
{
    if (img->d_data == NULL)
    {
        printf("CudaImage_Normalize: no device data\n");
        return;
    }

    int width = img->width;
    int height = img->height;
    int pitch = img->pitch;

    // Allocate device memory for min/max results.  RAII guards free these even
    // if a kernel-launch check or memcpy below throws.
    DevicePtrGuard<float> d_min_guard, d_max_guard;
    safeCall(cudaMalloc((void **)&d_min_guard.getRef(), sizeof(float)));
    safeCall(cudaMalloc((void **)&d_max_guard.getRef(), sizeof(float)));
    float *d_min = d_min_guard.get();
    float *d_max = d_max_guard.get();
    // Initialize: min to +FLT_MAX, max to -FLT_MAX
    float initMin = FLT_MAX;
    float initMax = -FLT_MAX;
    safeCall(cudaMemcpy(d_min, &initMin, sizeof(float), cudaMemcpyHostToDevice));
    safeCall(cudaMemcpy(d_max, &initMax, sizeof(float), cudaMemcpyHostToDevice));

    // Find min/max only over valid (width x height) pixels, skipping pitch padding
    dim3 threads(32, 8);
    dim3 blocks(iDivUp(width, threads.x), iDivUp(height, threads.y));
    MinMaxKernel<<<blocks, threads>>>(img->d_data, width, pitch, height, d_min, d_max);
    checkMsg("MinMaxKernel() execution failed\n");

    float minVal, maxVal;
    safeCall(cudaMemcpy(&minVal, d_min, sizeof(float), cudaMemcpyDeviceToHost));
    safeCall(cudaMemcpy(&maxVal, d_max, sizeof(float), cudaMemcpyDeviceToHost));

    float diff = maxVal - minVal;
    if (diff < 1e-12f)
    {
        // Constant image — zero everything out rather than dividing by ~0
        safeCall(cudaMemset(img->d_data, 0, sizeof(float) * pitch * height));
        return;
    }

    float invRange = 1.0f / diff;

    NormalizeKernel<<<blocks, threads>>>(img->d_data, width, pitch, height, minVal, invRange);
    checkMsg("CudaImage_Normalize() execution failed\n");
}

