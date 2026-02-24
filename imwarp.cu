
#include "cusift.h"
#include "RAII_Gaurds.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <omp.h>


// ── Helper: 3x3 matrix inverse (row-major) ──────────────
static bool Invert3x3(const float* M, float* out)
{
    float a = M[0], b = M[1], c = M[2];
    float d = M[3], e = M[4], f = M[5];
    float g = M[6], h = M[7], i = M[8];

    float det = a * (e * i - f * h)
              - b * (d * i - f * g)
              + c * (d * h - e * g);
    if (fabsf(det) < 1e-12f)
        return false;

    float inv_det = 1.0f / det;
    out[0] = (e * i - f * h) * inv_det;
    out[1] = (c * h - b * i) * inv_det;
    out[2] = (b * f - c * e) * inv_det;
    out[3] = (f * g - d * i) * inv_det;
    out[4] = (a * i - c * g) * inv_det;
    out[5] = (c * d - a * f) * inv_det;
    out[6] = (d * h - e * g) * inv_det;
    out[7] = (b * g - a * h) * inv_det;
    out[8] = (a * e - b * d) * inv_det;
    return true;
}

// ── Bilinear sample (CPU) ───────────────────────────────
static inline float SampleBilinear(const float* img, int w, int h, float x, float y)
{
    if (x < 0.0f || x >= (float)(w - 1) || y < 0.0f || y >= (float)(h - 1))
        return 0.0f;

    int x0 = (int)x;
    int y0 = (int)y;
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Clamp to be safe at exact boundaries
    x1 = x1 < w ? x1 : w - 1;
    y1 = y1 < h ? y1 : h - 1;

    float fx = x - (float)x0;
    float fy = y - (float)y0;

    float v00 = img[y0 * w + x0];
    float v10 = img[y0 * w + x1];
    float v01 = img[y1 * w + x0];
    float v11 = img[y1 * w + x1];

    return (1.0f - fx) * (1.0f - fy) * v00
         + fx          * (1.0f - fy) * v10
         + (1.0f - fx) * fy          * v01
         + fx          * fy          * v11;
}

// ── GPU kernels ─────────────────────────────────────────
__global__ void WarpIdentityKernel(const float* __restrict__ src, float* __restrict__ dst,
                                   int srcW, int srcH, int dstW, int dstH,
                                   float originU, float originV)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH)
        return;

    // Canvas coords → source coords
    float sx = (float)x + originU;
    float sy = (float)y + originV;

    float val = 0.0f;
    if (sx >= 0.0f && sx < (float)(srcW - 1) && sy >= 0.0f && sy < (float)(srcH - 1))
    {
        int x0 = (int)sx;
        int y0 = (int)sy;
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        float fx = sx - (float)x0;
        float fy = sy - (float)y0;
        val = (1.0f - fx) * (1.0f - fy) * src[y0 * srcW + x0]
            + fx          * (1.0f - fy) * src[y0 * srcW + x1]
            + (1.0f - fx) * fy          * src[y1 * srcW + x0]
            + fx          * fy          * src[y1 * srcW + x1];
    }
    dst[y * dstW + x] = val;
}

__global__ void WarpHomographyKernel(const float* __restrict__ src, float* __restrict__ dst,
                                     int srcW, int srcH, int dstW, int dstH,
                                     float originU, float originV,
                                     float h00, float h01, float h02,
                                     float h10, float h11, float h12,
                                     float h20, float h21, float h22)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH)
        return;

    // Canvas coords
    float u = (float)x + originU;
    float v = (float)y + originV;

    // Apply homography to get source coords
    float z = h20 * u + h21 * v + h22;
    float sx = (h00 * u + h01 * v + h02) / z;
    float sy = (h10 * u + h11 * v + h12) / z;

    float val = 0.0f;
    if (sx >= 0.0f && sx < (float)(srcW - 1) && sy >= 0.0f && sy < (float)(srcH - 1))
    {
        int x0 = (int)sx;
        int y0 = (int)sy;
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        float fx = sx - (float)x0;
        float fy = sy - (float)y0;
        val = (1.0f - fx) * (1.0f - fy) * src[y0 * srcW + x0]
            + fx          * (1.0f - fy) * src[y0 * srcW + x1]
            + (1.0f - fx) * fy          * src[y1 * srcW + x0]
            + fx          * fy          * src[y1 * srcW + x1];
    }
    dst[y * dstW + x] = val;
}

// ── Main entry point ────────────────────────────────────
void WarpImages(const Image_t* image1, const Image_t* image2, const float* homography,
                Image_t* warped_image1, Image_t* warped_image2, bool useGPU)
{
    int w1 = image1->width_, h1 = image1->height_;
    int w2 = image2->width_, h2 = image2->height_;

    // ── Step 1: Compute inv(homography) and project image2 corners ──
    float Hinv[9];
    if (!Invert3x3(homography, Hinv))
    {
        fprintf(stderr, "WarpImages: homography is singular\n");
        return;
    }

    // Image2 corners in 0-based convention: (col, row, 1)
    float corners2[4][3] = {
        { 0.0f,            0.0f,            1.0f },
        { (float)(w2 - 1), 0.0f,            1.0f },
        { (float)(w2 - 1), (float)(h2 - 1), 1.0f },
        { 0.0f,            (float)(h2 - 1), 1.0f }
    };

    // Project corners through inv(H) → image1's coordinate frame
    float cx[4], cy[4];
    for (int i = 0; i < 4; i++)
    {
        float x = corners2[i][0], y = corners2[i][1];
        float z = Hinv[6] * x + Hinv[7] * y + Hinv[8];
        cx[i] = (Hinv[0] * x + Hinv[1] * y + Hinv[2]) / z;
        cy[i] = (Hinv[3] * x + Hinv[4] * y + Hinv[5]) / z;
    }

    // ── Step 2: Determine output canvas bounding box ────
    float uMin = 0.0f, uMax = (float)(w1 - 1);
    float vMin = 0.0f, vMax = (float)(h1 - 1);
    for (int i = 0; i < 4; i++)
    {
        uMin = std::min(uMin, cx[i]);
        uMax = std::max(uMax, cx[i]);
        vMin = std::min(vMin, cy[i]);
        vMax = std::max(vMax, cy[i]);
    }

    // Integer range (MATLAB-style: ur = min:max with integer steps)
    int u0 = (int)floorf(uMin);
    int u1 = (int)ceilf(uMax);
    int v0 = (int)floorf(vMin);
    int v1 = (int)ceilf(vMax);

    int outW = u1 - u0 + 1;
    int outH = v1 - v0 + 1;

    // Origin offset: canvas pixel (0,0) corresponds to 1-based coord (u0, v0)
    float originU = (float)u0;
    float originV = (float)v0;

    // ── Step 3: Allocate output images ──────────────────
    size_t nPixels = (size_t)outW * outH;
    HostPtrGuard<float> out1Guard((float*)malloc(sizeof(float) * nPixels));
    HostPtrGuard<float> out2Guard((float*)malloc(sizeof(float) * nPixels));
    if (!out1Guard.get() || !out2Guard.get())
    {
        fprintf(stderr, "WarpImages: allocation failed\n");
        return;
    }

    // Homography row-major elements
    float H00 = homography[0], H01 = homography[1], H02 = homography[2];
    float H10 = homography[3], H11 = homography[4], H12 = homography[5];
    float H20 = homography[6], H21 = homography[7], H22 = homography[8];

    if (useGPU)
    {
        // ── GPU path ────────────────────────────────────
        DevicePtrGuard<float> d_src1Guard, d_src2Guard;
        DevicePtrGuard<float> d_out1Guard, d_out2Guard;

        size_t src1Bytes = (size_t)w1 * h1 * sizeof(float);
        size_t src2Bytes = (size_t)w2 * h2 * sizeof(float);
        size_t outBytes  = nPixels * sizeof(float);

        cudaMalloc(&d_src1Guard.getRef(), src1Bytes);
        cudaMalloc(&d_src2Guard.getRef(), src2Bytes);
        cudaMalloc(&d_out1Guard.getRef(), outBytes);
        cudaMalloc(&d_out2Guard.getRef(), outBytes);

        cudaMemcpy(d_src1Guard.get(), image1->host_img_, src1Bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_src2Guard.get(), image2->host_img_, src2Bytes, cudaMemcpyHostToDevice);

        dim3 threads(16, 16);
        dim3 blocks((outW + threads.x - 1) / threads.x, (outH + threads.y - 1) / threads.y);

        // Warped image1: identity warp (translate into canvas)
        WarpIdentityKernel<<<blocks, threads>>>(d_src1Guard.get(), d_out1Guard.get(), w1, h1, outW, outH, originU, originV);

        // Warped image2: apply homography
        WarpHomographyKernel<<<blocks, threads>>>(d_src2Guard.get(), d_out2Guard.get(), w2, h2, outW, outH,
                                                   originU, originV,
                                                   H00, H01, H02,
                                                   H10, H11, H12,
                                                   H20, H21, H22);

        cudaDeviceSynchronize();

        cudaMemcpy(out1Guard.get(), d_out1Guard.get(), outBytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(out2Guard.get(), d_out2Guard.get(), outBytes, cudaMemcpyDeviceToHost);
        // d_src1Guard, d_src2Guard, d_out1Guard, d_out2Guard freed automatically
    }
    else
    {
        float* out1 = out1Guard.get();
        float* out2 = out2Guard.get();
        // ── CPU path ────────────────────────────────────
        #pragma omp parallel for schedule(dynamic, 16)
        for (int y = 0; y < outH; y++)
        {
            float v = (float)y + originV;
            for (int x = 0; x < outW; x++)
            {
                float u = (float)x + originU;

                // Warped image1: identity (just sample at canvas coords)
                out1[y * outW + x] = SampleBilinear(image1->host_img_, w1, h1, u, v);

                // Warped image2: apply homography to get source coords
                float z = H20 * u + H21 * v + H22;
                float su = (H00 * u + H01 * v + H02) / z;
                float sv = (H10 * u + H11 * v + H12) / z;
                out2[y * outW + x] = SampleBilinear(image2->host_img_, w2, h2, su, sv);
            }
        }
    }

    // ── Step 4: Fill output structs, release ownership ──
    warped_image1->host_img_ = out1Guard.release();
    warped_image1->width_    = outW;
    warped_image1->height_   = outH;

    warped_image2->host_img_ = out2Guard.release();
    warped_image2->width_    = outW;
    warped_image2->height_   = outH;
}