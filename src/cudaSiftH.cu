//********************************************************//
// CUDA SIFT extractor by Mårten Björkman aka Celebrandil //
//********************************************************//

#include <cstdio>
#include <cstring>
#include <climits>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>
#include <thrust/sort.h>

#include "cudautils.h"
#include "cusift.h"

#include "cudaImage.h"
#include "cudaSift.h"
#include "cudaSiftD.h"
#include "cudaSiftH.h"
#include "RAII_Guards.hpp"

#include "cudaSiftD.cu"

struct SiftPointCompare
{
    __host__ __device__ bool operator()(const SiftPoint &a, const SiftPoint &b) const
    {
        if (a.ypos != b.ypos)
            return a.ypos < b.ypos;
        if (a.xpos != b.xpos)
            return a.xpos < b.xpos;
        return a.scale < b.scale;
    }
};

// Keep
void InitCuda(int devNum)
{
    int nDevices;
    safeCall(cudaGetDeviceCount(&nDevices));
    if (!nDevices)
    {
        std::cerr << "No CUDA devices available" << std::endl;
        return;
    }
    devNum = std::min(nDevices - 1, devNum);
    deviceInit(devNum);
    cudaDeviceProp prop;
    safeCall(cudaGetDeviceProperties(&prop, devNum));
}

// Keep
float *AllocSiftTempMemory(int width, int height, int numOctaves)
{
    const int nd = NUM_SCALES + 3;
    int w = width;
    int h = height;
    int p = iAlignUp(w, 128);
    size_t size = (size_t)h * p;         // image sizes
    size_t sizeTmp = (size_t)nd * h * p; // laplace buffer sizes
    for (int i = 0; i < numOctaves; i++)
    {
        w /= 2;
        h /= 2;
        p = iAlignUp(w, 128);
        size += (size_t)h * p;
        sizeTmp += (size_t)nd * h * p;
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
    // Use cudaFree directly (not safeCall) so this function is safe
    // to call from destructors during stack unwinding.
    if (memoryTmp)
        cudaFree(memoryTmp);
}

// Keep
void ExtractSift(SiftData *siftData, CudaImage *img, int numOctaves, float initBlur, float thresh, float lowestScale, float highestScale, float edgeLimit, float *tempMemory)
{
    // Never publish a stale count: if anything below throws, the caller sees 0.
    siftData->numPts = 0;
    // The public API clamps octaves to [3, 7]; this internal entry point did
    // not, and numOctaves >= 8 overflowed the kernel[] table below and the
    // 17-entry device counter, while numOctaves <= 0 read d_pointCounter[-1].
    if (numOctaves < 1 || numOctaves > 7)
        cusift_fail(__FILE__, __LINE__, "ExtractSift: numOctaves must be in [1, 7]");
    if (siftData->d_data == NULL || siftData->maxPts <= 0)
        cusift_fail(__FILE__, __LINE__, "ExtractSift: SiftData is not initialised (call InitSiftData first)");
    if (img->d_data == NULL || img->width <= 0 || img->height <= 0)
        cusift_fail(__FILE__, __LINE__, "ExtractSift: image has no device data");
    // Row offsets inside the kernels are still 32-bit; one plane must fit.
    if ((size_t)img->pitch * (size_t)img->height > (size_t)INT_MAX)
        cusift_fail(__FILE__, __LINE__, "ExtractSift: image too large (pitch * height must fit in a 32-bit int)");

    // ---- Per-call device context (replaces module-level __constant__/__device__ globals) ----
    SiftDeviceContext ctx;
    ctx.maxNumPoints = siftData->maxPts;

    const size_t counterBytes   = (8 * 2 + 1) * sizeof(unsigned int);
    const size_t laplaceBytes   = 8 * 12 * 16 * sizeof(float);
    const size_t scaleDownBytes = 5 * sizeof(float);
    const size_t lowPassBytes   = (2 * LOWPASS_R + 1) * sizeof(float);
    const size_t totalBytes     = counterBytes + laplaceBytes + scaleDownBytes + lowPassBytes;

    DevicePtrGuard<char> contextMemGuard;
    safeCall(cudaMalloc((void **)&contextMemGuard.getRef(), totalBytes));
    char *base            = contextMemGuard.get();
    ctx.d_pointCounter    = (unsigned int *)base;
    ctx.d_laplaceKernel   = (float *)(base + counterBytes);
    ctx.d_scaleDownKernel = (float *)(base + counterBytes + laplaceBytes);
    ctx.d_lowPassKernel   = (float *)(base + counterBytes + laplaceBytes + scaleDownBytes);

    safeCall(cudaMemset(ctx.d_pointCounter, 0, counterBytes));

    const int nd = NUM_SCALES + 3;
    int w = img->width;
    int h = img->height;
    int p = iAlignUp(w, 128);
    int width = w, height = h;
    size_t size = (size_t)h * p;         // image sizes
    size_t sizeTmp = (size_t)nd * h * p; // laplace buffer sizes
    for (int i = 0; i < numOctaves; i++)
    {
        w /= 2;
        h /= 2;
        p = iAlignUp(w, 128);
        size += (size_t)h * p;
        sizeTmp += (size_t)nd * h * p;
    }
    float *memoryTmp = tempMemory;
    size += sizeTmp;
    DevicePtrGuard<float> memoryTmpGuard; // only owns memory if we allocate it
    if (!tempMemory)
    {
        size_t pitch;
        safeCall(cudaMallocPitch((void **)&memoryTmpGuard.getRef(), &pitch, (size_t)4096, (size + 4095) / 4096 * sizeof(float)));
        memoryTmp = memoryTmpGuard.get();
    }
    float *memorySub = memoryTmp + sizeTmp;

    CudaImageGuard lowImgGuard;
    CudaImage_Allocate(lowImgGuard.get(), width, height, iAlignUp(width, 128), false, memorySub, NULL);
    float kernel[8 * 12 * 16];
    PrepareLaplaceKernels(numOctaves, initBlur, kernel);
    safeCall(cudaMemcpy(ctx.d_laplaceKernel, kernel, laplaceBytes, cudaMemcpyHostToDevice));
    LowPass(lowImgGuard.get(), img, max(initBlur, 0.001f), ctx);
    ExtractSiftLoop(siftData, lowImgGuard.get(), numOctaves, initBlur, thresh, lowestScale, highestScale, edgeLimit, 1.0f, memoryTmp, memorySub + (size_t)height * iAlignUp(width, 128), ctx);
    // The final cumulative keypoint total lives in slot [2*numOctaves + 1]:
    // ComputeOrientationsCONST appends the last octave's secondary-orientation
    // features via atomicInc on d_PointCounter[2*octave + 1], and the descriptor
    // kernel computes descriptors up to that same slot.  Reading [2*numOctaves]
    // instead dropped the finest octave's secondary orientations (a real
    // undercount, since that octave usually yields the most keypoints).
    // Safe because numOctaves is capped at 7 → index 15 < counter length 17.
    unsigned int devCount = 0;
    safeCall(cudaMemcpy(&devCount, &ctx.d_pointCounter[2 * numOctaves + 1], sizeof(unsigned int), cudaMemcpyDeviceToHost));
    const int numPts = (devCount < (unsigned int)siftData->maxPts) ? (int)devCount : siftData->maxPts;

    // Sync device before sorting and copying to host
    safeCall(cudaDeviceSynchronize());

    // Sort by ypos, then xpos, then scale using thrust
    thrust::sort(thrust::device, siftData->d_data, siftData->d_data + numPts, SiftPointCompare());

    if (siftData->h_data)
        safeCall(cudaMemcpy(siftData->h_data, siftData->d_data, sizeof(SiftPoint) * (size_t)numPts, cudaMemcpyDeviceToHost));
    // Publish the count only once h_data is valid, so a throw from the sync,
    // the sort (thrust allocates temporary storage) or the copy never leaves
    // numPts > 0 over unwritten host memory.
    siftData->numPts = numPts;
    // lowImgGuard, memoryTmpGuard, and contextMemGuard are cleaned up automatically
}

// Keep
int ExtractSiftLoop(SiftData *siftData, CudaImage *img, int numOctaves, float initBlur, float thresh, float lowestScale, float highestScale, float edgeLimit, float subsampling, float *memoryTmp, float *memorySub, SiftDeviceContext &ctx)
{
    int w = img->width;
    int h = img->height;
    if (numOctaves > 1)
    {
        CudaImageGuard subImgGuard;
        int p = iAlignUp(w / 2, 128);
        CudaImage_Allocate(subImgGuard.get(), w / 2, h / 2, p, false, memorySub, NULL);
        ScaleDown(subImgGuard.get(), img, 0.5f, ctx);
        float totInitBlur = (float)sqrt(initBlur * initBlur + 0.5f * 0.5f) / 2.0f;
        ExtractSiftLoop(siftData, subImgGuard.get(), numOctaves - 1, totInitBlur, thresh, lowestScale, highestScale, edgeLimit, subsampling * 2.0f, memoryTmp, memorySub + (size_t)(h / 2) * p, ctx);
    }
    ExtractSiftOctave(siftData, img, numOctaves, thresh, lowestScale, highestScale, edgeLimit, subsampling, memoryTmp, ctx);
    return 0;
}

// Keep
void ExtractSiftOctave(SiftData *siftData, CudaImage *img, int octave, float thresh, float lowestScale, float highestScale, float edgeLimit, float subsampling, float *memoryTmp, SiftDeviceContext &ctx)
{
    const int nd = NUM_SCALES + 3;
    CudaImage diffImg[nd];
    int w = img->width;
    int h = img->height;
    int p = iAlignUp(w, 128);
    for (int i = 0; i < nd - 1; i++)
    {
        CudaImage_init(&diffImg[i]);
        CudaImage_Allocate(&diffImg[i], w, h, p, false, memoryTmp + (size_t)i * p * h, NULL);
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
    safeCall(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
    TextureObjectGuard texGuard(texObj);

    LaplaceMulti(texObj, img, diffImg, octave, ctx);
    FindPointsMulti(diffImg, siftData, thresh, edgeLimit, 1.0f / NUM_SCALES, lowestScale / subsampling, highestScale / subsampling, subsampling, octave, ctx);
    ComputeOrientations(texObj, img, siftData, octave, ctx);
    ExtractSiftDescriptors(texObj, siftData, subsampling, octave, ctx);

    texGuard.free(); // free texture object before freeing memory
    for (int i = 0; i < nd - 1; i++)
        CudaImage_destroy(&diffImg[i]);
}

// Keep
// NOTE: This treats *data as write-only: it never reads or frees the incoming
// h_data/d_data pointers (the public entry points call FreeSiftData() first,
// under the zero-initialisation contract documented in cusift.h).  On any
// failure the struct is left empty (maxPts == 0, NULL buffers) so that
// FreeSiftData() is always safe afterwards.
void InitSiftData(SiftData *data, int num, bool host, bool dev)
{
    data->numPts = 0;
    data->maxPts = 0;
    data->h_data = NULL;
    data->d_data = NULL;
    // The byte count used to be computed in a 32-bit int: a large num silently
    // wrapped to a tiny allocation while maxPts kept the large value, and every
    // kernel then wrote past the buffer.  Validate, and size in size_t.
    const int maxNum = (int)(INT_MAX / sizeof(SiftPoint));
    if (num <= 0 || num > maxNum)
        cusift_fail(__FILE__, __LINE__, "InitSiftData: number of keypoints must be in [1, " + std::to_string(maxNum) + "]");
    const size_t sz = sizeof(SiftPoint) * (size_t)num;
    if (host)
    {
        // calloc, not malloc: the caller may read fields the extraction kernels
        // never write, and heap garbage must not be exported as keypoint data.
        data->h_data = (SiftPoint *)calloc((size_t)num, sizeof(SiftPoint));
        if (data->h_data == NULL)
            cusift_fail(__FILE__, __LINE__, "InitSiftData: host allocation of " + std::to_string(sz) + " bytes failed");
    }
    if (dev)
    {
        cudaError_t err = cudaMalloc((void **)&data->d_data, sz);
        if (err == cudaSuccess)
            err = cudaMemset(data->d_data, 0, sz);
        if (err != cudaSuccess)
        {
            if (data->d_data)
                cudaFree(data->d_data);
            data->d_data = NULL;
            free(data->h_data);
            data->h_data = NULL;
            cusift_safe_call(err, __FILE__, __LINE__); // throws
        }
    }
    data->maxPts = num;
}

// Keep
void FreeSiftData(SiftData *data)
{
    // Use cudaFree directly (not safeCall) so this function is safe
    // to call from destructors during stack unwinding.
    if (data->d_data != NULL)
        cudaFree(data->d_data);
    data->d_data = NULL;
    if (data->h_data != NULL)
        free(data->h_data);
    data->numPts = 0;
    data->maxPts = 0;
    data->h_data = NULL;
}

///////////////////////////////////////////////////////////////////////////////
// Host side master functions
///////////////////////////////////////////////////////////////////////////////

// Keep
double ScaleDown(CudaImage *res, CudaImage *src, float variance, SiftDeviceContext &ctx)
{
    if (res->d_data == NULL || src->d_data == NULL)
    {
        printf("ScaleDown: missing data\n");
        return 0.0;
    }
    float h_Kernel[5];
    float kernelSum = 0.0f;
    for (int j = 0; j < 5; j++)
    {
        h_Kernel[j] = expf(-(float)(j - 2) * (j - 2) / 2.0f / variance);
        kernelSum += h_Kernel[j];
    }
    for (int j = 0; j < 5; j++)
        h_Kernel[j] /= kernelSum;
    safeCall(cudaMemcpy(ctx.d_scaleDownKernel, h_Kernel, 5 * sizeof(float), cudaMemcpyHostToDevice));
    dim3 blocks(iDivUp(src->width, SCALEDOWN_W), iDivUp(src->height, SCALEDOWN_H));
    dim3 threads(SCALEDOWN_W + 4);
    ScaleDown<<<blocks, threads>>>(res->d_data, src->d_data, src->width, src->pitch, src->height, res->pitch, ctx.d_scaleDownKernel);
    checkMsg("ScaleDown() execution failed\n");
    return 0.0;
}

// Keep
double ComputeOrientations(cudaTextureObject_t texObj, CudaImage * /*src*/, SiftData *siftData, int octave, SiftDeviceContext &ctx)
{
    dim3 blocks(512);
    dim3 threads(11 * 11);
    ComputeOrientationsCONST<<<blocks, threads>>>(texObj, siftData->d_data, octave, ctx.maxNumPoints, ctx.d_pointCounter);
    checkMsg("ComputeOrientations() execution failed\n");
    return 0.0;
}

// Keep
double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData *siftData, float subsampling, int octave, SiftDeviceContext &ctx)
{
    dim3 blocks(512);
    dim3 threads(16, 8);
    ExtractSiftDescriptorsCONSTNew<<<blocks, threads>>>(texObj, siftData->d_data, subsampling, octave, ctx.maxNumPoints, ctx.d_pointCounter);
    checkMsg("ExtractSiftDescriptors() execution failed\n");
    return 0.0;
}

// Keep
double LowPass(CudaImage *res, CudaImage *src, float scale, SiftDeviceContext &ctx)
{
    float kernel[2 * LOWPASS_R + 1];
    float kernelSum = 0.0f;
    float ivar2 = 1.0f / (2.0f * scale * scale);
    for (int j = -LOWPASS_R; j <= LOWPASS_R; j++)
    {
        kernel[j + LOWPASS_R] = expf(-(float)j * j * ivar2);
        kernelSum += kernel[j + LOWPASS_R];
    }
    for (int j = -LOWPASS_R; j <= LOWPASS_R; j++)
        kernel[j + LOWPASS_R] /= kernelSum;
    safeCall(cudaMemcpy(ctx.d_lowPassKernel, kernel, (2 * LOWPASS_R + 1) * sizeof(float), cudaMemcpyHostToDevice));
    int width = res->width;
    int pitch = res->pitch;
    int height = res->height;
    dim3 blocks(iDivUp(width, LOWPASS_W), iDivUp(height, LOWPASS_H));
    dim3 threads(LOWPASS_W + 2 * LOWPASS_R, 4);
    LowPassBlock<<<blocks, threads>>>(src->d_data, res->d_data, width, src->pitch, pitch, height, ctx.d_lowPassKernel);
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
            kernel[numOctaves * 12 * 16 + 16 * i + j] = expf(-(float)j * j / 2.0f / var);
            kernelSum += (j == 0 ? 1 : 2) * kernel[numOctaves * 12 * 16 + 16 * i + j];
        }
        for (int j = 0; j <= LAPLACE_R; j++)
            kernel[numOctaves * 12 * 16 + 16 * i + j] /= kernelSum;
        scale *= diffScale;
    }
}

// Keep
double LaplaceMulti(cudaTextureObject_t /*texObj*/, CudaImage *baseImage, CudaImage *results, int octave, SiftDeviceContext &ctx)
{
    int width = results[0].width;
    int pitch = results[0].pitch;
    int height = results[0].height;
    dim3 threads(LAPLACE_W + 2 * LAPLACE_R);
    dim3 blocks(iDivUp(width, LAPLACE_W), height);
    LaplaceMultiMem<<<blocks, threads>>>(baseImage->d_data, results[0].d_data, width, pitch, height, octave, ctx.d_laplaceKernel);
    checkMsg("LaplaceMulti() execution failed\n");
    return 0.0;
}

// Keep
double FindPointsMulti(CudaImage *sources, SiftData *siftData, float thresh, float edgeLimit, float factor, float lowestScale, float highestScale, float subsampling, int octave, SiftDeviceContext &ctx)
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
    FindPointsMultiNew<<<blocks, threads>>>(sources->d_data, siftData->d_data, w, p, h, subsampling, lowestScale, highestScale, thresh, factor, edgeLimit, octave, ctx.maxNumPoints, ctx.d_pointCounter);
    checkMsg("FindPointsMulti() execution failed\n");
    return 0.0;
}

///////////////////////////////////////////////////////////////////////////////
// Scale-based non-maximum suppression
///////////////////////////////////////////////////////////////////////////////

// Keep
void SuppressEmbeddedPoints(SiftData *data, float radiusScale)
{
    if (!data || !data->h_data || data->numPts <= 1)
        return;

    const int n = data->numPts;
    SiftPoint *pts = data->h_data;

    // --- Build an index sorted by scale (descending) --------------------
    std::vector<int> order(n);
    for (int i = 0; i < n; i++)
        order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return pts[a].scale > pts[b].scale;
    });

    // --- Spatial grid for fast neighbour lookup -------------------------
    // Cell size = radius of the largest keypoint so that a 3x3 neighbourhood
    // of cells covers every suppression radius.  Everything is computed in
    // double and clamped on both sides before any float->int cast.
    if (!(radiusScale > 0.0f))
        return;
    float maxScale = pts[order[0]].scale;
    double cellSize = (double)radiusScale * (double)maxScale;
    if (!(cellSize >= 1e-6))
        return;  // degenerate (all scales ~0) or NaN

    // Find bounding box
    float minX = pts[0].xpos, maxX = minX;
    float minY = pts[0].ypos, maxY = minY;
    for (int i = 1; i < n; i++) {
        float x = pts[i].xpos, y = pts[i].ypos;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
    }
    const double spanX = (double)maxX - (double)minX;
    const double spanY = (double)maxY - (double)minY;
    if (!std::isfinite(spanX) || !std::isfinite(spanY))
        return;  // non-finite coordinates: nothing sensible to suppress

    // Bound the grid by the keypoint count rather than a fixed cell cap: the
    // old 16M-cell limit let a large image with a small largest scale allocate
    // ~400 MB of std::vector headers per call.  If the natural grid has more
    // than ~4 cells per point, coarsen the cells.  cellSize only ever grows, so
    // it still covers every radius and the result is unchanged.
    const double maxCells = std::max(1024.0, 4.0 * (double)n);
    const double cells = (std::floor(spanX / cellSize) + 1.0) * (std::floor(spanY / cellSize) + 1.0);
    if (cells > maxCells)
        cellSize *= std::sqrt(cells / maxCells);
    const int gridW = (int)std::floor(spanX / cellSize) + 1;
    const int gridH = (int)std::floor(spanY / cellSize) + 1;

    auto cellOf = [cellSize](float v, float lo, int gridN) -> int {
        double c = ((double)v - (double)lo) / cellSize;
        if (!(c > 0.0))
            return 0;  // also catches NaN
        if (c >= (double)(gridN - 1))
            return gridN - 1;
        return (int)c;
    };

    // Map from grid cell -> list of point indices (in scale-descending order)
    std::vector<std::vector<int>> grid((size_t)gridW * (size_t)gridH);
    for (int i = 0; i < n; i++) {
        int idx = order[i];
        int cx = cellOf(pts[idx].xpos, minX, gridW);
        int cy = cellOf(pts[idx].ypos, minY, gridH);
        grid[(size_t)cy * gridW + cx].push_back(idx);
    }

    // --- Suppression pass ------------------------------------------------
    std::vector<bool> suppressed(n, false);

    for (int ii = 0; ii < n; ii++) {
        int i = order[ii];
        if (suppressed[i])
            continue;

        float xi = pts[i].xpos;
        float yi = pts[i].ypos;
        float ri = radiusScale * pts[i].scale;
        float ri2 = ri * ri;

        int cx = cellOf(xi, minX, gridW);
        int cy = cellOf(yi, minY, gridH);

        // ri <= cellSize by construction, so a radius reaches at most into the
        // adjacent cell; widen to 2 only when it sits right on the boundary.
        int reach = ((double)ri >= cellSize) ? 2 : 1;
        int gx0 = std::max(0, cx - reach);
        int gx1 = std::min(gridW - 1, cx + reach);
        int gy0 = std::max(0, cy - reach);
        int gy1 = std::min(gridH - 1, cy + reach);

        for (int gy = gy0; gy <= gy1; gy++) {
            for (int gx = gx0; gx <= gx1; gx++) {
                for (int j : grid[(size_t)gy * gridW + gx]) {
                    if (j == i || suppressed[j])
                        continue;
                    // j must be smaller-or-equal scale (processed after i)
                    if (pts[j].scale > pts[i].scale)
                        continue;
                    float dx = pts[j].xpos - xi;
                    float dy = pts[j].ypos - yi;
                    if (dx * dx + dy * dy < ri2)
                        suppressed[j] = true;
                }
            }
        }
    }

    // --- Compact ---------------------------------------------------------
    int dst = 0;
    for (int i = 0; i < n; i++) {
        if (!suppressed[i]) {
            if (dst != i)
                pts[dst] = pts[i];
            dst++;
        }
    }
    data->numPts = dst;

    // --- Re-upload to device if device buffer exists ---------------------
    if (data->d_data && dst > 0)
        safeCall(cudaMemcpy(data->d_data, data->h_data,
                            sizeof(SiftPoint) * dst, cudaMemcpyHostToDevice));
}
