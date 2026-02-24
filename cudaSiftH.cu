//********************************************************//
// CUDA SIFT extractor by Mårten Björkman aka Celebrandil //
//********************************************************//

#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "cudautils.h"

#include "cudaImage.h"
#include "cudaSift.h"
#include "cudaSiftD.h"
#include "cudaSiftH.h"

#include "cudaSiftD.cu"

// Keep
void InitCuda(int devNum)
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if (!nDevices)
    {
        std::cerr << "No CUDA devices available" << std::endl;
        return;
    }
    devNum = std::min(nDevices - 1, devNum);
    deviceInit(devNum);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, devNum);
}

// Keep
float *AllocSiftTempMemory(int width, int height, int numOctaves)
{
    const int nd = NUM_SCALES + 3;
    int w = width;
    int h = height;
    int p = iAlignUp(w, 128);
    int size = h * p;         // image sizes
    int sizeTmp = nd * h * p; // laplace buffer sizes
    for (int i = 0; i < numOctaves; i++)
    {
        w /= 2;
        h /= 2;
        p = iAlignUp(w, 128);
        size += h * p;
        sizeTmp += nd * h * p;
    }
    float *memoryTmp = NULL;
    size_t pitch;
    size += sizeTmp;
    safeCall(cudaMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096, (size + 4095) / 4096 * sizeof(float)));
    return memoryTmp;
}

// Keep
void FreeSiftTempMemory(float *memoryTmp)
{
    if (memoryTmp)
        safeCall(cudaFree(memoryTmp));
}

// Keep
void ExtractSift(SiftData *siftData, CudaImage *img, int numOctaves, float initBlur, float thresh, float lowestScale, float *tempMemory)
{
    unsigned int *d_PointCounterAddr;
    safeCall(cudaGetSymbolAddress((void **)&d_PointCounterAddr, d_PointCounter));
    safeCall(cudaMemset(d_PointCounterAddr, 0, (8 * 2 + 1) * sizeof(int)));
    safeCall(cudaMemcpyToSymbol(d_MaxNumPoints, &siftData->maxPts, sizeof(int)));

    const int nd = NUM_SCALES + 3;
    int w = img->width;
    int h = img->height;
    int p = iAlignUp(w, 128);
    int width = w, height = h;
    int size = h * p;         // image sizes
    int sizeTmp = nd * h * p; // laplace buffer sizes
    for (int i = 0; i < numOctaves; i++)
    {
        w /= 2;
        h /= 2;
        p = iAlignUp(w, 128);
        size += h * p;
        sizeTmp += nd * h * p;
    }
    float *memoryTmp = tempMemory;
    size += sizeTmp;
    if (!tempMemory)
    {
        size_t pitch;
        safeCall(cudaMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096, (size + 4095) / 4096 * sizeof(float)));
    }
    float *memorySub = memoryTmp + sizeTmp;

    CudaImage lowImg;
    CudaImage_init(&lowImg);
    CudaImage_Allocate(&lowImg, width, height, iAlignUp(width, 128), false, memorySub, NULL);
    float kernel[8 * 12 * 16];
    PrepareLaplaceKernels(numOctaves, 0.0f, kernel);
    safeCall(cudaMemcpyToSymbolAsync(d_LaplaceKernel, kernel, 8 * 12 * 16 * sizeof(float)));
    LowPass(&lowImg, img, max(initBlur, 0.001f));
    ExtractSiftLoop(siftData, &lowImg, numOctaves, 0.0f, thresh, lowestScale, 1.0f, memoryTmp, memorySub + height * iAlignUp(width, 128));
    safeCall(cudaMemcpy(&siftData->numPts, &d_PointCounterAddr[2 * numOctaves], sizeof(int), cudaMemcpyDeviceToHost));
    siftData->numPts = (siftData->numPts < siftData->maxPts ? siftData->numPts : siftData->maxPts);

    if (!tempMemory)
        safeCall(cudaFree(memoryTmp));
    if (siftData->h_data)
        safeCall(cudaMemcpy(siftData->h_data, siftData->d_data, sizeof(SiftPoint) * siftData->numPts, cudaMemcpyDeviceToHost));
    CudaImage_destroy(&lowImg);
}

// Keep
int ExtractSiftLoop(SiftData *siftData, CudaImage *img, int numOctaves, float initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp, float *memorySub)
{
    int w = img->width;
    int h = img->height;
    if (numOctaves > 1)
    {
        CudaImage subImg;
        CudaImage_init(&subImg);
        int p = iAlignUp(w / 2, 128);
        CudaImage_Allocate(&subImg, w / 2, h / 2, p, false, memorySub, NULL);
        ScaleDown(&subImg, img, 0.5f);
        float totInitBlur = (float)sqrt(initBlur * initBlur + 0.5f * 0.5f) / 2.0f;
        ExtractSiftLoop(siftData, &subImg, numOctaves - 1, totInitBlur, thresh, lowestScale, subsampling * 2.0f, memoryTmp, memorySub + (h / 2) * p);
        CudaImage_destroy(&subImg);
    }
    ExtractSiftOctave(siftData, img, numOctaves, thresh, lowestScale, subsampling, memoryTmp);
    return 0;
}

// Keep
void ExtractSiftOctave(SiftData *siftData, CudaImage *img, int octave, float thresh, float lowestScale, float subsampling, float *memoryTmp)
{
    const int nd = NUM_SCALES + 3;
    CudaImage diffImg[nd];
    int w = img->width;
    int h = img->height;
    int p = iAlignUp(w, 128);
    for (int i = 0; i < nd - 1; i++)
    {
        CudaImage_init(&diffImg[i]);
        CudaImage_Allocate(&diffImg[i], w, h, p, false, memoryTmp + i * p * h, NULL);
    }

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = img->d_data;
    resDesc.res.pitch2D.width = img->width;
    resDesc.res.pitch2D.height = img->height;
    resDesc.res.pitch2D.pitchInBytes = img->pitch * sizeof(float);
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    float baseBlur = pow(2.0f, -1.0f / NUM_SCALES);
    float diffScale = pow(2.0f, 1.0f / NUM_SCALES);
    LaplaceMulti(texObj, img, diffImg, octave);
    FindPointsMulti(diffImg, siftData, thresh, 10.0f, 1.0f / NUM_SCALES, lowestScale / subsampling, subsampling, octave);
    ComputeOrientations(texObj, img, siftData, octave);
    ExtractSiftDescriptors(texObj, siftData, subsampling, octave);

    safeCall(cudaDestroyTextureObject(texObj));
    for (int i = 0; i < nd - 1; i++)
        CudaImage_destroy(&diffImg[i]);
}

// Keep
void InitSiftData(SiftData *data, int num, bool host, bool dev)
{
    data->numPts = 0;
    data->maxPts = num;
    int sz = sizeof(SiftPoint) * num;
    data->h_data = NULL;
    if (host)
        data->h_data = (SiftPoint *)malloc(sz);
    data->d_data = NULL;
    if (dev)
        safeCall(cudaMalloc((void **)&data->d_data, sz));
}

// Keep
void FreeSiftData(SiftData *data)
{
    if (data->d_data != NULL)
        safeCall(cudaFree(data->d_data));
    data->d_data = NULL;
    if (data->h_data != NULL)
        free(data->h_data);
    data->numPts = 0;
    data->maxPts = 0;
}

///////////////////////////////////////////////////////////////////////////////
// Host side master functions
///////////////////////////////////////////////////////////////////////////////

// Keep
double ScaleDown(CudaImage *res, CudaImage *src, float variance)
{
    float oldVariance = -1.0f;
    if (res->d_data == NULL || src->d_data == NULL)
    {
        printf("ScaleDown: missing data\n");
        return 0.0;
    }
    if (oldVariance != variance)
    {
        float h_Kernel[5];
        float kernelSum = 0.0f;
        for (int j = 0; j < 5; j++)
        {
            h_Kernel[j] = (float)expf(-(double)(j - 2) * (j - 2) / 2.0 / variance);
            kernelSum += h_Kernel[j];
        }
        for (int j = 0; j < 5; j++)
            h_Kernel[j] /= kernelSum;
        safeCall(cudaMemcpyToSymbol(d_ScaleDownKernel, h_Kernel, 5 * sizeof(float)));
        oldVariance = variance;
    }
    dim3 blocks(iDivUp(src->width, SCALEDOWN_W), iDivUp(src->height, SCALEDOWN_H));
    dim3 threads(SCALEDOWN_W + 4);
    ScaleDown<<<blocks, threads>>>(res->d_data, src->d_data, src->width, src->pitch, src->height, res->pitch);
    checkMsg("ScaleDown() execution failed\n");
    return 0.0;
}

// Keep
double ComputeOrientations(cudaTextureObject_t texObj, CudaImage *src, SiftData *siftData, int octave)
{
    dim3 blocks(512);
    dim3 threads(11 * 11);
    ComputeOrientationsCONST<<<blocks, threads>>>(texObj, siftData->d_data, octave);
    checkMsg("ComputeOrientations() execution failed\n");
    return 0.0;
}

// Keep
double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData *siftData, float subsampling, int octave)
{
    dim3 blocks(512);
    dim3 threads(16, 8);
    ExtractSiftDescriptorsCONSTNew<<<blocks, threads>>>(texObj, siftData->d_data, subsampling, octave);
    checkMsg("ExtractSiftDescriptors() execution failed\n");
    return 0.0;
}

// Keep
double LowPass(CudaImage *res, CudaImage *src, float scale)
{
    float kernel[2 * LOWPASS_R + 1];
    float oldScale = -1.0f;
    if (scale != oldScale)
    {
        float kernelSum = 0.0f;
        float ivar2 = 1.0f / (2.0f * scale * scale);
        for (int j = -LOWPASS_R; j <= LOWPASS_R; j++)
        {
            kernel[j + LOWPASS_R] = (float)expf(-(double)j * j * ivar2);
            kernelSum += kernel[j + LOWPASS_R];
        }
        for (int j = -LOWPASS_R; j <= LOWPASS_R; j++)
            kernel[j + LOWPASS_R] /= kernelSum;
        safeCall(cudaMemcpyToSymbol(d_LowPassKernel, kernel, (2 * LOWPASS_R + 1) * sizeof(float)));
        oldScale = scale;
    }
    int width = res->width;
    int pitch = res->pitch;
    int height = res->height;
    dim3 blocks(iDivUp(width, LOWPASS_W), iDivUp(height, LOWPASS_H));
    dim3 threads(LOWPASS_W + 2 * LOWPASS_R, 4);
    LowPassBlock<<<blocks, threads>>>(src->d_data, res->d_data, width, pitch, height);
    checkMsg("LowPass() execution failed\n");
    return 0.0;
}

//==================== Multi-scale functions ===================//

// Keep
void PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel)
{
    if (numOctaves > 1)
    {
        float totInitBlur = (float)sqrt(initBlur * initBlur + 0.5f * 0.5f) / 2.0f;
        PrepareLaplaceKernels(numOctaves - 1, totInitBlur, kernel);
    }
    float scale = pow(2.0f, -1.0f / NUM_SCALES);
    float diffScale = pow(2.0f, 1.0f / NUM_SCALES);
    for (int i = 0; i < NUM_SCALES + 3; i++)
    {
        float kernelSum = 0.0f;
        float var = scale * scale - initBlur * initBlur;
        for (int j = 0; j <= LAPLACE_R; j++)
        {
            kernel[numOctaves * 12 * 16 + 16 * i + j] = (float)expf(-(double)j * j / 2.0 / var);
            kernelSum += (j == 0 ? 1 : 2) * kernel[numOctaves * 12 * 16 + 16 * i + j];
        }
        for (int j = 0; j <= LAPLACE_R; j++)
            kernel[numOctaves * 12 * 16 + 16 * i + j] /= kernelSum;
        scale *= diffScale;
    }
}

// Keep
double LaplaceMulti(cudaTextureObject_t texObj, CudaImage *baseImage, CudaImage *results, int octave)
{
    int width = results[0].width;
    int pitch = results[0].pitch;
    int height = results[0].height;
    dim3 threads(LAPLACE_W + 2 * LAPLACE_R);
    dim3 blocks(iDivUp(width, LAPLACE_W), height);
    LaplaceMultiMem<<<blocks, threads>>>(baseImage->d_data, results[0].d_data, width, pitch, height, octave);
    checkMsg("LaplaceMulti() execution failed\n");
    return 0.0;
}

// Keep
double FindPointsMulti(CudaImage *sources, SiftData *siftData, float thresh, float edgeLimit, float factor, float lowestScale, float subsampling, int octave)
{
    if (sources->d_data == NULL)
    {
        printf("FindPointsMulti: missing data\n");
        return 0.0;
    }
    int w = sources->width;
    int p = sources->pitch;
    int h = sources->height;
    dim3 blocks(iDivUp(w, MINMAX_W) * NUM_SCALES, iDivUp(h, MINMAX_H));
    dim3 threads(MINMAX_W + 2);
    FindPointsMultiNew<<<blocks, threads>>>(sources->d_data, siftData->d_data, w, p, h, subsampling, lowestScale, thresh, factor, edgeLimit, octave);
    checkMsg("FindPointsMulti() execution failed\n");
    return 0.0;
}
