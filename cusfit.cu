#include "cusift.h"
#include "cudautils.h"
#include "cudaImage.h"
#include "cudaSift.h"
#include "geomFuncs.h"
#include "RAII_Gaurds.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <omp.h>

static int p_iAlignUp(int a, int b) { return (a % b != 0) ? (a - a % b + b) : a; }

static bool Invert3x3(const float *M, float *out);
static inline float SampleBilinear(const float *img, int w, int h, float x, float y);
static __global__ void WarpDualKernel(
    const float *__restrict__ src1, float *__restrict__ dst1, int src1W, int src1H, int src1Pitch,
    const float *__restrict__ src2, float *__restrict__ dst2, int src2W, int src2H, int src2Pitch,
    int dstW, int dstH, int dstPitch, float originU, float originV,
    float h00, float h01, float h02, float h10, float h11, float h12, float h20, float h21, float h22);

// ── Thread-local error storage ──────────────────────────
static thread_local int s_lastErrorLine = 0;
static thread_local char s_lastErrorFile[256] = {};
static thread_local char s_lastErrorMessage[256] = {};
static thread_local bool s_hadError = false;

static void cusift_clear_error()
{
    s_hadError = false;
    s_lastErrorLine = 0;
    s_lastErrorFile[0] = '\0';
    s_lastErrorMessage[0] = '\0';
}

// Parse file/line from __safeCall message format:
//   "CUDA error in file 'FILE' in line LINE : MSG"
static void cusift_store_error_from_exception(const char *what)
{
    s_hadError = true;

    const char *file_start = strstr(what, "in file '");
    const char *line_start = strstr(what, "in line ");

    if (file_start && line_start)
    {
        file_start += 9; // skip "in file '"
        const char *file_end = strchr(file_start, '\'');
        if (file_end)
        {
            size_t len = (size_t)(file_end - file_start);
            if (len > 255)
                len = 255;
            memcpy(s_lastErrorFile, file_start, len);
            s_lastErrorFile[len] = '\0';
        }

        line_start += 8; // skip "in line "
        s_lastErrorLine = atoi(line_start);
    }
    else
    {
        s_lastErrorFile[0] = '\0';
        s_lastErrorLine = 0;
    }

    strncpy(s_lastErrorMessage, what, 255);
    s_lastErrorMessage[255] = '\0';
}

// Wrap an API body: clear error state, run fn(), catch and store any exception.
template <typename F>
static void cusift_api_guard(F &&fn)
{
    cusift_clear_error();
    try
    {
        fn();
    }
    catch (const std::exception &e)
    {
        cusift_store_error_from_exception(e.what());
    }
    catch (...)
    {
        s_hadError = true;
        s_lastErrorLine = 0;
        s_lastErrorFile[0] = '\0';
        strncpy(s_lastErrorMessage, "Unknown exception", 255);
        s_lastErrorMessage[255] = '\0';
    }
}

void CusiftGetLastErrorString(int *line_number, char filename[256], char error_message[256])
{
    if (line_number)
        *line_number = s_lastErrorLine;
    if (filename)
    {
        strncpy(filename, s_lastErrorFile, 255);
        filename[255] = '\0';
    }
    if (error_message)
    {
        strncpy(error_message, s_lastErrorMessage, 255);
        error_message[255] = '\0';
    }
}

int CusiftHadError()
{
    return s_hadError ? 1 : 0;
}

void InitializeCudaSift()
{
    cusift_api_guard([&]()
                     {
        int nDevices;
        cudaGetDeviceCount(&nDevices);
        if (!nDevices)
        {
            std::cerr << "No CUDA devices available" << std::endl;
            return;
        }
        int devNum = std::min(nDevices - 1, 0);
        safeCall(cudaSetDevice(devNum)); });
}

void ExtractSiftFromImage(const Image_t *image, SiftData *sift_data, const ExtractSiftOptions_t *options)
{
    cusift_api_guard([&]()
                     {
        CudaImageGuard cuda_image;

        InitSiftData(sift_data, options->max_keypoints_, true, true);

        CudaImage_Allocate(
            cuda_image.get(),
            image->width_,
            image->height_,
            p_iAlignUp(image->width_, 128),
            false,
            nullptr,
            image->host_img_);

        CudaImage_Download(cuda_image.get());
        //CudaImage_Normalize(cuda_image.get());

        // Get the smallest dimension of the image to determine the maximum number of octaves
        int minDim = std::min(image->width_, image->height_);
        int maxOctaves = static_cast<int>(std::floor(std::log2(minDim))) - 3;
        int octaves = std::min(options->num_octaves_, maxOctaves);
        if (options->num_octaves_ > maxOctaves)
        {
            std::cerr << "Warning: Requested number of octaves (" << options->num_octaves_ << ") exceeds the maximum possible (" << maxOctaves << ") for the given image size. Reducing to " << maxOctaves << "." << std::endl;
        }

        SiftTempMemoryGuard tempMemory(AllocSiftTempMemory(image->width_, image->height_, octaves));

        ExtractSift(
            sift_data,
            cuda_image.get(),
            octaves,
            options->init_blur_,
            options->thresh_,
            options->lowest_scale_,
            options->edge_thresh_,
            tempMemory.get()); });
}

void MatchSiftData(SiftData *data1, SiftData *data2)
{
    cusift_api_guard([&]()
                     { MatchSiftData_private(data1, data2); });
}

void FindHomography(SiftData *data, float *homography, int *num_matches, const FindHomographyOptions_t *options)
{
    cusift_api_guard([&]()
                     {
        FindHomography_private(
            data,
            homography,
            num_matches,
            options->num_loops_,
            options->min_score_,
            options->max_ambiguity_,
            options->thresh_,
            options->seed_);

        ImproveHomography(
            data,
            homography,
            options->improve_num_loops_,
            options->improve_min_score_,
            options->improve_max_ambiguity_,
            options->improve_thresh_); });
}

void DeleteSiftData(SiftData *sift_data)
{
    // FreeSiftData uses cudaFree directly (not safeCall), so it is
    // destructor-safe and won't throw.  We still wrap for consistency.
    cusift_api_guard([&]()
                     { FreeSiftData(sift_data); });
}

void FreeImage(Image_t *image)
{
    if (image)
        if (image->host_img_)
        {
            free(image->host_img_);
            image->host_img_ = nullptr;
            image->width_ = 0;
            image->height_ = 0;
        }
}

void SaveSiftData(const char *filename, const SiftData *sift_data)
{
    if (!sift_data || !sift_data->h_data || sift_data->numPts <= 0)
    {
        std::cerr << "SaveSiftData: no data to save" << std::endl;
        return;
    }

    FILE *f = fopen(filename, "w");
    if (!f)
    {
        std::cerr << "SaveSiftData: could not open file " << filename << std::endl;
        return;
    }

    fprintf(f, "{\n");
    fprintf(f, "  \"num_keypoints\": %d,\n", sift_data->numPts);
    fprintf(f, "  \"keypoints\": [\n");

    for (int i = 0; i < sift_data->numPts; i++)
    {
        const SiftPoint *pt = &sift_data->h_data[i];
        fprintf(f, "    {\n");
        fprintf(f, "      \"x\": %.6f,\n", pt->xpos);
        fprintf(f, "      \"y\": %.6f,\n", pt->ypos);
        fprintf(f, "      \"scale\": %.6f,\n", pt->scale);
        fprintf(f, "      \"sharpness\": %.6f,\n", pt->sharpness);
        fprintf(f, "      \"edgeness\": %.6f,\n", pt->edgeness);
        fprintf(f, "      \"orientation\": %.6f,\n", pt->orientation);
        fprintf(f, "      \"score\": %.6f,\n", pt->score);
        fprintf(f, "      \"ambiguity\": %.6f,\n", pt->ambiguity);
        fprintf(f, "      \"match\": %d,\n", pt->match);
        fprintf(f, "      \"match_x\": %.6f,\n", pt->match_xpos);
        fprintf(f, "      \"match_y\": %.6f,\n", pt->match_ypos);
        fprintf(f, "      \"match_error\": %.6f,\n", pt->match_error);
        fprintf(f, "      \"descriptor\": [");
        for (int j = 0; j < 128; j++)
        {
            fprintf(f, "%.6f", pt->data[j]);
            if (j < 127)
                fprintf(f, ", ");
        }
        fprintf(f, "]\n");
        fprintf(f, "    }%s\n", (i < sift_data->numPts - 1) ? "," : "");
    }

    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    fclose(f);
}

void ExtractAndMatchSift(const Image_t *image1, const Image_t *image2, SiftData *sift_data1, SiftData *sift_data2, const ExtractSiftOptions_t *extract_options)
{
    cusift_api_guard([&]()
                     {
        int maxW = std::max(image1->width_, image2->width_);
        int maxH = std::max(image1->height_, image2->height_);

        CudaImageGuard cuda_image1;
        CudaImageGuard cuda_image2;

        // Clamp octaves to what the smallest image dimension supports
        int minDim = std::min({image1->width_, image1->height_, image2->width_, image2->height_});
        int maxOctaves = static_cast<int>(std::floor(std::log2(minDim))) - 3;
        int octaves = std::min(extract_options->num_octaves_, maxOctaves);

        // Only allocate a single temporary buffer for both images since they won't be processed at the same time
        SiftTempMemoryGuard tempMemory(AllocSiftTempMemory(maxW, maxH, octaves));

        // Extract from image 1
        InitSiftData(sift_data1, extract_options->max_keypoints_, true, true);
        CudaImage_Allocate(cuda_image1.get(), image1->width_, image1->height_,
                           p_iAlignUp(image1->width_, 128), false, nullptr, image1->host_img_);
        CudaImage_Download(cuda_image1.get());
        //CudaImage_Normalize(cuda_image1.get());
        ExtractSift(sift_data1, cuda_image1.get(), octaves,
                    extract_options->init_blur_, extract_options->thresh_,
                    extract_options->lowest_scale_, extract_options->edge_thresh_,
                    tempMemory.get());

        // Extract from image 2
        InitSiftData(sift_data2, extract_options->max_keypoints_, true, true);
        CudaImage_Allocate(cuda_image2.get(), image2->width_, image2->height_,
                           p_iAlignUp(image2->width_, 128), false, nullptr, image2->host_img_);
        CudaImage_Download(cuda_image2.get());
        //CudaImage_Normalize(cuda_image2.get());
        ExtractSift(sift_data2, cuda_image2.get(), octaves,
                    extract_options->init_blur_, extract_options->thresh_,
                    extract_options->lowest_scale_, extract_options->edge_thresh_,
                    tempMemory.get());

        // Match
        MatchSiftData_private(sift_data1, sift_data2); });
}

void ExtractAndMatchAndFindHomography(const Image_t *image1, const Image_t *image2, SiftData *sift_data1, SiftData *sift_data2, float *homography, int *num_matches, const ExtractSiftOptions_t *extract_options, const FindHomographyOptions_t *homography_options)
{
    cusift_api_guard([&]()
                     {
        int maxW = std::max(image1->width_, image2->width_);
        int maxH = std::max(image1->height_, image2->height_);

        CudaImageGuard cuda_image1;
        CudaImageGuard cuda_image2;

        // Clamp octaves to what the smallest image dimension supports
        int minDim = std::min({image1->width_, image1->height_, image2->width_, image2->height_});
        int maxOctaves = static_cast<int>(std::floor(std::log2(minDim))) - 3;
        int octaves = std::min(extract_options->num_octaves_, maxOctaves);

        // Only allocate a single temporary buffer for both images since they won't be processed at the same time
        SiftTempMemoryGuard tempMemory(AllocSiftTempMemory(maxW, maxH, octaves));

        // Extract from image 1
        InitSiftData(sift_data1, extract_options->max_keypoints_, true, true);
        CudaImage_Allocate(cuda_image1.get(), image1->width_, image1->height_,
                           p_iAlignUp(image1->width_, 128), false, nullptr, image1->host_img_);
        CudaImage_Download(cuda_image1.get());
        //CudaImage_Normalize(cuda_image1.get());
        ExtractSift(sift_data1, cuda_image1.get(), octaves,
                    extract_options->init_blur_, extract_options->thresh_,
                    extract_options->lowest_scale_, extract_options->edge_thresh_,
                    tempMemory.get());

        // Extract from image 2
        InitSiftData(sift_data2, extract_options->max_keypoints_, true, true);
        CudaImage_Allocate(cuda_image2.get(), image2->width_, image2->height_,
                           p_iAlignUp(image2->width_, 128), false, nullptr, image2->host_img_);
        CudaImage_Download(cuda_image2.get());
        //CudaImage_Normalize(cuda_image2.get());
        ExtractSift(sift_data2, cuda_image2.get(), octaves,
                    extract_options->init_blur_, extract_options->thresh_,
                    extract_options->lowest_scale_, extract_options->edge_thresh_,
                    tempMemory.get());

        // Match
        MatchSiftData_private(sift_data1, sift_data2);

        FindHomography_private(
            sift_data1, homography, num_matches,
            homography_options->num_loops_,
            homography_options->min_score_,
            homography_options->max_ambiguity_,
            homography_options->thresh_,
            homography_options->seed_);

        ImproveHomography(
            sift_data1, homography,
            homography_options->improve_num_loops_,
            homography_options->improve_min_score_,
            homography_options->improve_max_ambiguity_,
            homography_options->improve_thresh_); });
}

// ── Helper: 3x3 matrix inverse (row-major) ──────────────
bool Invert3x3(const float *M, float *out)
{
    float a = M[0], b = M[1], c = M[2];
    float d = M[3], e = M[4], f = M[5];
    float g = M[6], h = M[7], i = M[8];

    float det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
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
inline float SampleBilinear(const float *img, int w, int h, float x, float y)
{
    if (x < 0.0f || x >= (float)(w - 1) || y < 0.0f || y >= (float)(h - 1))
        return std::nanf(""); // Out of bounds; caller should check before sampling but return nan just in case

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

    return (1.0f - fx) * (1.0f - fy) * v00 + fx * (1.0f - fy) * v10 + (1.0f - fx) * fy * v01 + fx * fy * v11;
}

// ── GPU kernel ──────────────────────────────────────────
// Dual warp kernel: warps both images in a single launch.
// Image 1 is identity-warped (translated into canvas).
// Image 2 is warped through the supplied homography.
// srcPitch / dstPitch are row strides in floats (may differ from width
// when the allocation is pitch-aligned).
__global__ void WarpDualKernel(
    const float *__restrict__ src1, float *__restrict__ dst1, int src1W, int src1H, int src1Pitch,
    const float *__restrict__ src2, float *__restrict__ dst2, int src2W, int src2H, int src2Pitch,
    int dstW, int dstH, int dstPitch, float originU, float originV,
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

    // ── Image 1: identity warp (just translate into canvas) ──
    float val1 = std::nanf(""); // Padded pixels get nan so they show up as black in visualization and don't contribute to error metrics
    if (u >= 0.0f && u < (float)(src1W - 1) && v >= 0.0f && v < (float)(src1H - 1))
    {
        int x0 = (int)u;
        int y0 = (int)v;
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        float fx = u - (float)x0;
        float fy = v - (float)y0;
        val1 = (1.0f - fx) * (1.0f - fy) * src1[y0 * src1Pitch + x0] + fx * (1.0f - fy) * src1[y0 * src1Pitch + x1] + (1.0f - fx) * fy * src1[y1 * src1Pitch + x0] + fx * fy * src1[y1 * src1Pitch + x1];
    }
    dst1[y * dstPitch + x] = val1;

    // ── Image 2: homography warp ──
    float z = h20 * u + h21 * v + h22;
    float sx = (h00 * u + h01 * v + h02) / z;
    float sy = (h10 * u + h11 * v + h12) / z;

    float val2 = std::nanf(""); // Padded pixels get nan so they show up as black in visualization and don't contribute to error metrics
    if (sx >= 0.0f && sx < (float)(src2W - 1) && sy >= 0.0f && sy < (float)(src2H - 1))
    {
        int x0 = (int)sx;
        int y0 = (int)sy;
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        float fx = sx - (float)x0;
        float fy = sy - (float)y0;
        val2 = (1.0f - fx) * (1.0f - fy) * src2[y0 * src2Pitch + x0] + fx * (1.0f - fy) * src2[y0 * src2Pitch + x1] + (1.0f - fx) * fy * src2[y1 * src2Pitch + x0] + fx * fy * src2[y1 * src2Pitch + x1];
    }
    dst2[y * dstPitch + x] = val2;
}

// ── Main entry point ────────────────────────────────────
void WarpImages(const Image_t *image1, const Image_t *image2, const float *homography,
                Image_t *warped_image1, Image_t *warped_image2, bool useGPU)
{
    cusift_api_guard([&]()
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

        size_t src1Pitch = 0, src2Pitch = 0;
        size_t out1Pitch = 0, out2Pitch = 0;

        safeCall(cudaMallocPitch(&d_src1Guard.getRef(), &src1Pitch, w1 * sizeof(float), h1));
        safeCall(cudaMallocPitch(&d_src2Guard.getRef(), &src2Pitch, w2 * sizeof(float), h2));
        safeCall(cudaMallocPitch(&d_out1Guard.getRef(), &out1Pitch, outW * sizeof(float), outH));
        safeCall(cudaMallocPitch(&d_out2Guard.getRef(), &out2Pitch, outW * sizeof(float), outH));

        safeCall(cudaMemcpy2D(d_src1Guard.get(), src1Pitch, image1->host_img_, w1 * sizeof(float), w1 * sizeof(float), h1, cudaMemcpyHostToDevice));
        safeCall(cudaMemcpy2D(d_src2Guard.get(), src2Pitch, image2->host_img_, w2 * sizeof(float), w2 * sizeof(float), h2, cudaMemcpyHostToDevice));

        // Strides in elements (pixels) for the kernel
        int src1StrideElem = (int)(src1Pitch / sizeof(float));
        int src2StrideElem = (int)(src2Pitch / sizeof(float));
        int out1StrideElem = (int)(out1Pitch / sizeof(float));
        // out2 stride same width as out1 since both are outW

        dim3 threads(16, 16);
        dim3 blocks((outW + threads.x - 1) / threads.x, (outH + threads.y - 1) / threads.y);

        // Warp both images in a single kernel launch
        WarpDualKernel<<<blocks, threads>>>(
            d_src1Guard.get(), d_out1Guard.get(), w1, h1, src1StrideElem,
            d_src2Guard.get(), d_out2Guard.get(), w2, h2, src2StrideElem,
            outW, outH, out1StrideElem, originU, originV,
            H00, H01, H02,
            H10, H11, H12,
            H20, H21, H22);

        cudaDeviceSynchronize();

        cudaMemcpy2D(out1Guard.get(), outW * sizeof(float), d_out1Guard.get(), out1Pitch, outW * sizeof(float), outH, cudaMemcpyDeviceToHost);
        cudaMemcpy2D(out2Guard.get(), outW * sizeof(float), d_out2Guard.get(), out2Pitch, outW * sizeof(float), outH, cudaMemcpyDeviceToHost);

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
    warped_image2->height_   = outH; }); // end cusift_api_guard for WarpImages
}

void ExtractAndMatchAndFindHomographyAndWarp(const Image_t *image1, const Image_t *image2, SiftData *sift_data1, SiftData *sift_data2, float *homography, int *num_matches, const ExtractSiftOptions_t *extract_options, const FindHomographyOptions_t *homography_options, Image_t *warped_image1, Image_t *warped_image2)
{
    cusift_api_guard([&]()
                     {
    int w1 = image1->width_, h1 = image1->height_;
    int w2 = image2->width_, h2 = image2->height_;
    int maxW = std::max(w1, w2);
    int maxH = std::max(h1, h2);

    // ── Extract SIFT features (images stay on device via CudaImageGuard) ──
    CudaImageGuard cuda_image1;
    CudaImageGuard cuda_image2;

    int minDim = std::min({w1, h1, w2, h2});
    int maxOctaves = static_cast<int>(std::floor(std::log2(minDim))) - 3;
    int octaves = std::min(extract_options->num_octaves_, maxOctaves);

    SiftTempMemoryGuard tempMemory(AllocSiftTempMemory(maxW, maxH, octaves));

    // Image 1
    InitSiftData(sift_data1, extract_options->max_keypoints_, true, true);
    CudaImage_Allocate(cuda_image1.get(), w1, h1,
                       p_iAlignUp(w1, 128), false, nullptr, image1->host_img_);
    CudaImage_Download(cuda_image1.get());
    ExtractSift(sift_data1, cuda_image1.get(), octaves,
                extract_options->init_blur_, extract_options->thresh_,
                extract_options->lowest_scale_, extract_options->edge_thresh_,
                tempMemory.get());

    // Image 2
    InitSiftData(sift_data2, extract_options->max_keypoints_, true, true);
    CudaImage_Allocate(cuda_image2.get(), w2, h2,
                       p_iAlignUp(w2, 128), false, nullptr, image2->host_img_);
    CudaImage_Download(cuda_image2.get());
    ExtractSift(sift_data2, cuda_image2.get(), octaves,
                extract_options->init_blur_, extract_options->thresh_,
                extract_options->lowest_scale_, extract_options->edge_thresh_,
                tempMemory.get());

    // ── Match ────────────────────────────────────────────────────────────
    MatchSiftData_private(sift_data1, sift_data2);

    // ── Find homography ─────────────────────────────────────────────────
    FindHomography_private(
        sift_data1, homography, num_matches,
        homography_options->num_loops_,
        homography_options->min_score_,
        homography_options->max_ambiguity_,
        homography_options->thresh_,
        homography_options->seed_);

    ImproveHomography(
        sift_data1, homography,
        homography_options->improve_num_loops_,
        homography_options->improve_min_score_,
        homography_options->improve_max_ambiguity_,
        homography_options->improve_thresh_);

    // ── Compute canvas bounding box ─────────────────────────────────────
    float Hinv[9];
    if (!Invert3x3(homography, Hinv))
    {
        fprintf(stderr, "ExtractAndMatchAndFindHomographyAndWarp: homography is singular\n");
        return;
    }

    float corners2[4][3] = {
        { 0.0f,            0.0f,            1.0f },
        { (float)(w2 - 1), 0.0f,            1.0f },
        { (float)(w2 - 1), (float)(h2 - 1), 1.0f },
        { 0.0f,            (float)(h2 - 1), 1.0f }
    };

    float cx[4], cy[4];
    for (int i = 0; i < 4; i++)
    {
        float x = corners2[i][0], y = corners2[i][1];
        float z = Hinv[6] * x + Hinv[7] * y + Hinv[8];
        cx[i] = (Hinv[0] * x + Hinv[1] * y + Hinv[2]) / z;
        cy[i] = (Hinv[3] * x + Hinv[4] * y + Hinv[5]) / z;
    }

    float uMin = 0.0f, uMax = (float)(w1 - 1);
    float vMin = 0.0f, vMax = (float)(h1 - 1);
    for (int i = 0; i < 4; i++)
    {
        uMin = std::min(uMin, cx[i]);
        uMax = std::max(uMax, cx[i]);
        vMin = std::min(vMin, cy[i]);
        vMax = std::max(vMax, cy[i]);
    }

    int u0 = (int)floorf(uMin);
    int u1 = (int)ceilf(uMax);
    int v0 = (int)floorf(vMin);
    int v1 = (int)ceilf(vMax);

    int outW = u1 - u0 + 1;
    int outH = v1 - v0 + 1;
    float originU = (float)u0;
    float originV = (float)v0;

    // ── Warp on GPU — source data is already device-resident ────────────
    // Reuse the CudaImageGuard device pointers (no extra upload needed).
    float* d_src1 = cuda_image1.get()->d_data;
    float* d_src2 = cuda_image2.get()->d_data;
    int src1Pitch = cuda_image1.get()->pitch;
    int src2Pitch = cuda_image2.get()->pitch;

    DevicePtrGuard<float> d_out1Guard, d_out2Guard;

    size_t out1Stride = 0;
    size_t out2Stride = 0;

    safeCall(cudaMallocPitch(&d_out1Guard.getRef(), &out1Stride, outW * sizeof(float), outH));
    safeCall(cudaMallocPitch(&d_out2Guard.getRef(), &out2Stride, outW * sizeof(float), outH));

    float H00 = homography[0], H01 = homography[1], H02 = homography[2];
    float H10 = homography[3], H11 = homography[4], H12 = homography[5];
    float H20 = homography[6], H21 = homography[7], H22 = homography[8];

    dim3 threads(16, 16);
    dim3 blocks((outW + threads.x - 1) / threads.x,
                (outH + threads.y - 1) / threads.y);

    // Warp both images in a single kernel launch
    WarpDualKernel<<<blocks, threads>>>(
        d_src1, d_out1Guard.get(), w1, h1, src1Pitch,
        d_src2, d_out2Guard.get(), w2, h2, src2Pitch,
        outW, outH, outW, originU, originV,
        H00, H01, H02,
        H10, H11, H12,
        H20, H21, H22);

    cudaDeviceSynchronize();

    // ── Copy results to host and fill output structs ────────────────────
    size_t nPixels = (size_t)outW * outH;
    HostPtrGuard<float> out1Guard((float*)malloc(sizeof(float) * nPixels));
    HostPtrGuard<float> out2Guard((float*)malloc(sizeof(float) * nPixels));
    if (!out1Guard.get() || !out2Guard.get())
    {
        fprintf(stderr, "ExtractAndMatchAndFindHomographyAndWarp: host allocation failed\n");
        return;
    }

    cudaMemcpy2D(out1Guard.get(), outW * sizeof(float), d_out1Guard.get(), out1Stride, outW * sizeof(float), outH, cudaMemcpyDeviceToHost);
    cudaMemcpy2D(out2Guard.get(), outW * sizeof(float), d_out2Guard.get(), out2Stride, outW * sizeof(float), outH, cudaMemcpyDeviceToHost);

    warped_image1->host_img_ = out1Guard.release();
    warped_image1->width_    = outW;
    warped_image1->height_   = outH;

    warped_image2->host_img_ = out2Guard.release();
    warped_image2->width_    = outW;
    warped_image2->height_   = outH; }); // end cusift_api_guard for ExtractAndMatchAndFindHomographyAndWarp
}