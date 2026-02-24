//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//

#include <cstdio>
#include <cfloat>

#include "cudautils.h"
#include "cudaImage.h"

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
    if (img->d_internalAlloc && img->d_data != NULL)
        safeCall(cudaFree(img->d_data));
    img->d_data = NULL;
    if (img->h_internalAlloc && img->h_data != NULL)
        free(img->h_data);
    img->h_data = NULL;
    if (img->t_data != NULL)
        safeCall(cudaFreeArray((cudaArray *)img->t_data));
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
    if (img->d_data != NULL && img->h_data != NULL)
        safeCall(cudaMemcpy2D(img->d_data, p, img->h_data, sizeof(float) * img->width, sizeof(float) * img->width, img->height, cudaMemcpyHostToDevice));
}

void CudaImage_Readback(CudaImage *img)
{
    int p = sizeof(float) * img->pitch;
    safeCall(cudaMemcpy2D(img->h_data, sizeof(float) * img->width, img->d_data, p, sizeof(float) * img->width, img->height, cudaMemcpyDeviceToHost));
}

///////////////////////////////////////////////////////////////////////////////
// Min-Max normalization: maps pixel values to [0, 1]
///////////////////////////////////////////////////////////////////////////////

__global__ void MinMaxKernel(float *d_data, int width, int pitch, int height, float *d_min, float *d_max)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float val = (x < width && y < height) ? d_data[y * pitch + x] : 0.0f;
    bool valid = (x < width && y < height);

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        bool otherValid = __shfl_down_sync(0xffffffff, (int)valid, offset);
        if (otherValid && (!valid || other < val))
        {
            val = other;
            valid = true;
        }
    }
    float warpMin = val;

    // Reset for max reduction
    val = (x < width && y < height) ? d_data[y * pitch + x] : 0.0f;
    valid = (x < width && y < height);
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        float other = __shfl_down_sync(0xffffffff, val, offset);
        bool otherValid = __shfl_down_sync(0xffffffff, (int)valid, offset);
        if (otherValid && (!valid || other > val))
        {
            val = other;
            valid = true;
        }
    }
    float warpMax = val;

    int lane = threadIdx.x + threadIdx.y * blockDim.x;
    int warpId = lane / 32;
    int laneInWarp = lane % 32;

    __shared__ float sMin[8]; // up to 256 threads = 8 warps
    __shared__ float sMax[8];
    __shared__ bool sValid[8];

    if (laneInWarp == 0)
    {
        sMin[warpId] = warpMin;
        sMax[warpId] = warpMax;
        sValid[warpId] = (x < width && y < height) || __any_sync(0xffffffff, (x < width && y < height));
    }
    __syncthreads();

    // Final reduction across warps in first warp
    int numWarps = (blockDim.x * blockDim.y + 31) / 32;
    if (lane < numWarps)
    {
        float localMin = sMin[lane];
        float localMax = sMax[lane];
        bool localValid = sValid[lane];
        for (int offset = numWarps / 2; offset > 0; offset >>= 1)
        {
            if (lane + offset < numWarps)
            {
                if (sValid[lane + offset] && (!localValid || sMin[lane + offset] < localMin))
                {
                    localMin = sMin[lane + offset];
                    localValid = true;
                }
                if (sValid[lane + offset] && sMax[lane + offset] > localMax)
                    localMax = sMax[lane + offset];
            }
        }
        if (lane == 0 && localValid)
        {
            atomicMin((int *)d_min, __float_as_int(localMin));
            // atomicMax for floats: use int atomicMax with sign-magnitude trick
            // Since we expect non-negative pixel values, __float_as_int preserves order
            atomicMax((int *)d_max, __float_as_int(localMax));
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

    // Allocate device memory for min/max results
    float *d_min = NULL, *d_max = NULL;
    safeCall(cudaMalloc(&d_min, sizeof(float)));
    safeCall(cudaMalloc(&d_max, sizeof(float)));
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
    safeCall(cudaFree(d_min));
    safeCall(cudaFree(d_max));

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

