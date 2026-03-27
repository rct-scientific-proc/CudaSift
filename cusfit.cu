#include "cusift.h"
#include "cudautils.h"
#include "cudaImage.h"
#include "cudaSift.h"
#include "cudaSiftH.h"
#include "geomFuncs.h"
#include "RAII_Gaurds.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <string>
#include <omp.h>

#define ERROR(msg) __throw_error(__FILE__, __LINE__, msg)

static void __throw_error(const char *file, int line, const char *msg)
{
    std::ostringstream ss;
    ss << "\nError in file '" << file << "' in line " << line << " : " << msg;
    throw std::runtime_error(ss.str());
}

static int p_iAlignUp(int a, int b) { return (a % b != 0) ? (a - a % b + b) : a; }

// Dispatch to either full homography or similarity RANSAC based on model_type_
static void FindGeometricModel(SiftData *data, float *homography, int *numMatches,
                                const FindHomographyOptions_t *opts)
{
    if (opts->model_type_ == CUSIFT_MODEL_SIMILARITY)
    {
        FindSimilarity_private(
            data, homography, numMatches,
            opts->num_loops_, opts->min_score_, opts->max_ambiguity_,
            opts->thresh_, opts->seed_);
    }
    else
    {
        FindHomography_private(
            data, homography, numMatches,
            opts->num_loops_, opts->min_score_, opts->max_ambiguity_,
            opts->thresh_, opts->seed_);
    }
}

static bool Invert3x3(const float *M, float *out);
static inline float SampleBilinear(const float *img, int w, int h, float x, float y);
static __global__ void WarpDualKernel(
    const float *__restrict__ src1, float *__restrict__ dst1, int src1W, int src1H, int src1Pitch,
    const float *__restrict__ src2, float *__restrict__ dst2, int src2W, int src2H, int src2Pitch,
    int dstW, int dstH, int dstPitch, float originU, float originV,
    float h00, float h01, float h02, float h10, float h11, float h12, float h20, float h21, float h22);

// ── Thread-local error storage ──────────────────────────
struct CusiftErrorState
{
    bool        had_error = false;
    int         line      = 0;
    std::string file;
    std::string message;

    void clear()
    {
        had_error = false;
        line      = 0;
        file.clear();
        message.clear();
    }

    // Parse file/line from __safeCall message format:
    //   "CUDA error in file 'FILE' in line LINE : MSG"
    void store_from_exception(const char *what)
    {
        had_error = true;
        message   = what;

        const char *file_start = strstr(what, "in file '");
        const char *line_start = strstr(what, "in line ");

        if (file_start && line_start)
        {
            file_start += 9; // skip "in file '"
            const char *file_end = strchr(file_start, '\'');
            if (file_end)
                file.assign(file_start, file_end);
            else
                file.clear();

            line_start += 8; // skip "in line "
            line = atoi(line_start);
        }
        else
        {
            file.clear();
            line = 0;
        }
    }
};

static thread_local CusiftErrorState s_error;

// Wrap an API body: clear error state, run fn(), catch and store any exception.
template <typename F>
static void cusift_api_guard(F &&fn)
{
    s_error.clear();
    try
    {
        fn();
    }
    catch (const std::exception &e)
    {
        s_error.store_from_exception(e.what());
    }
    catch (...)
    {
        s_error.had_error = true;
        s_error.line      = 0;
        s_error.file.clear();
        s_error.message = "Unknown exception";
    }
}

void CusiftGetLastErrorString(int *line_number, char filename[256], char error_message[256])
{
    if (line_number)
        *line_number = s_error.line;
    if (filename)
    {
        strncpy(filename, s_error.file.c_str(), 255);
        filename[255] = '\0';
    }
    if (error_message)
    {
        strncpy(error_message, s_error.message.c_str(), 255);
        error_message[255] = '\0';
    }
}

int CusiftHadError()
{
    return s_error.had_error ? 1 : 0;
}

void InitializeCudaSift()
{
    cusift_api_guard([&]()
    {
        int nDevices;
        safeCall(cudaGetDeviceCount(&nDevices));
        if (!nDevices)
        {
            std::cerr << "No CUDA devices available" << std::endl;
            return;
        }
        int devNum = std::min(nDevices - 1, 0);
        safeCall(cudaSetDevice(devNum));
    });
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

        // Limit num octaves to 7, quietly
        octaves = std::min(octaves, 7);
        octaves = std::max(octaves, 3); // Ensure at least 3 octaves

        SiftTempMemoryGuard tempMemory(AllocSiftTempMemory(image->width_, image->height_, octaves));

        ExtractSift(
            sift_data,
            cuda_image.get(),
            octaves,
            options->init_blur_,
            options->thresh_,
            options->lowest_scale_,
            options->highest_scale_,
            options->edge_thresh_,
            tempMemory.get());
        if (options->scale_suppression_radius_ > 0.0f)
            SuppressEmbeddedPoints(sift_data, options->scale_suppression_radius_);
    });
}

void MatchSiftData(SiftData *data1, SiftData *data2)
{
    cusift_api_guard([&]()
    {
        MatchSiftData_private(data1, data2);
    });
}

void FindHomography(SiftData *data, float *homography, int *num_matches, const FindHomographyOptions_t *options)
{
    cusift_api_guard([&]()
    {
        FindGeometricModel(
            data,
            homography,
            num_matches,
            options);
        
        // If any homography values are nan of inf, we should throw an error rather than proceeding to the warp which will produce garbage output and may cause CUDA errors.
        for (int i = 0; i < 8; i++)
        {
            if (!std::isfinite(homography[i]))
            {
                ERROR("Homograpy contains non-finite values");
            }
        }

        ImproveHomography(
            data,
            homography,
            options->improve_num_loops_,
            options->improve_min_score_,
            options->improve_max_ambiguity_,
            options->improve_thresh_);
    });
}

void DeleteSiftData(SiftData *sift_data)
{
    // FreeSiftData uses cudaFree directly (not safeCall), so it is
    // destructor-safe and won't throw.  We still wrap for consistency.
    cusift_api_guard([&]()
    {
        FreeSiftData(sift_data);
    });
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

void FreeImage_GPU(ImageStrided_t *image)
{
    if (image)
        if (image->strided_img_)
        {
            cudaFree(image->strided_img_);
            image->strided_img_ = nullptr;
            image->width_ = 0;
            image->height_ = 0;
            image->stride_ = 0;
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
        if (extract_options->num_octaves_ > maxOctaves)
        {
            std::cerr << "Warning: Requested number of octaves (" << extract_options->num_octaves_ << ") exceeds the maximum possible (" << maxOctaves << ") for the given image size. Reducing to " << maxOctaves << "." << std::endl;
        }

        // Limit num octaves to 7, quietly
        octaves = std::min(octaves, 7);
        octaves = std::max(octaves, 3); // Ensure at least 3 octaves

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
                    extract_options->lowest_scale_, extract_options->highest_scale_, extract_options->edge_thresh_,
                    tempMemory.get());
        if (extract_options->scale_suppression_radius_ > 0.0f)
            SuppressEmbeddedPoints(sift_data1, extract_options->scale_suppression_radius_);

        // Extract from image 2
        InitSiftData(sift_data2, extract_options->max_keypoints_, true, true);
        CudaImage_Allocate(cuda_image2.get(), image2->width_, image2->height_,
                           p_iAlignUp(image2->width_, 128), false, nullptr, image2->host_img_);
        CudaImage_Download(cuda_image2.get());
        //CudaImage_Normalize(cuda_image2.get());
        ExtractSift(sift_data2, cuda_image2.get(), octaves,
                    extract_options->init_blur_, extract_options->thresh_,
                    extract_options->lowest_scale_, extract_options->highest_scale_, extract_options->edge_thresh_,
                    tempMemory.get());
        if (extract_options->scale_suppression_radius_ > 0.0f)
            SuppressEmbeddedPoints(sift_data2, extract_options->scale_suppression_radius_);

        // Match
        MatchSiftData_private(sift_data1, sift_data2);
    });
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
        if (extract_options->num_octaves_ > maxOctaves)
        {
            std::cerr << "Warning: Requested number of octaves (" << extract_options->num_octaves_ << ") exceeds the maximum possible (" << maxOctaves << ") for the given image size. Reducing to " << maxOctaves << "." << std::endl;
        }

        // Limit num octaves to 7, quietly
        octaves = std::min(octaves, 7);
        octaves = std::max(octaves, 3); // Ensure at least 3 octaves

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
                    extract_options->lowest_scale_, extract_options->highest_scale_, extract_options->edge_thresh_,
                    tempMemory.get());
        if (extract_options->scale_suppression_radius_ > 0.0f)
            SuppressEmbeddedPoints(sift_data1, extract_options->scale_suppression_radius_);

        // Extract from image 2
        InitSiftData(sift_data2, extract_options->max_keypoints_, true, true);
        CudaImage_Allocate(cuda_image2.get(), image2->width_, image2->height_,
                           p_iAlignUp(image2->width_, 128), false, nullptr, image2->host_img_);
        CudaImage_Download(cuda_image2.get());
        //CudaImage_Normalize(cuda_image2.get());
        ExtractSift(sift_data2, cuda_image2.get(), octaves,
                    extract_options->init_blur_, extract_options->thresh_,
                    extract_options->lowest_scale_, extract_options->highest_scale_, extract_options->edge_thresh_,
                    tempMemory.get());
        if (extract_options->scale_suppression_radius_ > 0.0f)
            SuppressEmbeddedPoints(sift_data2, extract_options->scale_suppression_radius_);

        // Match
        MatchSiftData_private(sift_data1, sift_data2);

        FindGeometricModel(
            sift_data1, homography, num_matches,
            homography_options);
        
        // If any homography values are nan of inf, we should throw an error rather than proceeding to the warp which will produce garbage output and may cause CUDA errors.
        for (int i = 0; i < 8; i++)
        {
            if (!std::isfinite(homography[i]))
            {
                ERROR("Homograpy contains non-finite values");
            }
        }

        ImproveHomography(
            sift_data1, homography,
            homography_options->improve_num_loops_,
            homography_options->improve_min_score_,
            homography_options->improve_max_ambiguity_,
            homography_options->improve_thresh_);
    });
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
            ERROR("Homography is too close to singular to invert");
        }

        // Image2 corners in 0-based convention: (col, row, 1)
        float corners2[4][3] = {
            {0.0f, 0.0f, 1.0f},
            {(float)(w2 - 1), 0.0f, 1.0f},
            {(float)(w2 - 1), (float)(h2 - 1), 1.0f},
            {0.0f, (float)(h2 - 1), 1.0f}};

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

        int maxW_ = 2 * std::max(image1->width_, image2->width_);
        int maxH_ = 2 * std::max(image1->height_, image2->height_);
        if (outW > maxW_ || outH > maxH_)
        {
            std::stringstream ss;
            ss << "Warping: warped image too large (" << outW << "x" << outH << "), not attempting warp";
            ERROR(ss.str().c_str());
        }

        if (outW <= 0 || outH <= 0)
        {
            std::stringstream ss;
            ss << "Warping: invalid output image size (" << outW << "x" << outH << ")";
            ERROR(ss.str().c_str());
        }

        // Origin offset: canvas pixel (0,0) corresponds to 1-based coord (u0, v0)
        float originU = (float)u0;
        float originV = (float)v0;

        // ── Step 3: Allocate output images ──────────────────
        size_t nPixels = (size_t)outW * outH;
        HostPtrGuard<float> out1Guard((float *)malloc(sizeof(float) * nPixels));
        HostPtrGuard<float> out2Guard((float *)malloc(sizeof(float) * nPixels));
        if (!out1Guard.get() || !out2Guard.get())
        {
            ERROR("Warping: host return image allocation failed");
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

            safeCall(cudaDeviceSynchronize());

            safeCall(cudaMemcpy2D(out1Guard.get(), outW * sizeof(float), d_out1Guard.get(), out1Pitch, outW * sizeof(float), outH, cudaMemcpyDeviceToHost));
            safeCall(cudaMemcpy2D(out2Guard.get(), outW * sizeof(float), d_out2Guard.get(), out2Pitch, outW * sizeof(float), outH, cudaMemcpyDeviceToHost));

            // d_src1Guard, d_src2Guard, d_out1Guard, d_out2Guard freed automatically
        }
        else
        {
            float *out1 = out1Guard.get();
            float *out2 = out2Guard.get();
// ── CPU path ────────────────────────────────────
#pragma omp parallel for schedule(static)
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
        warped_image1->width_ = outW;
        warped_image1->height_ = outH;

        warped_image2->host_img_ = out2Guard.release();
        warped_image2->width_ = outW;
        warped_image2->height_ = outH;
    }); // end cusift_api_guard for WarpImages
}

void WarpImages_GPU(const Image_t *image1, const Image_t *image2, const float *homography,
                    ImageStrided_t *warped_image1, ImageStrided_t *warped_image2)
{
    cusift_api_guard([&]()
    {
        int w1 = image1->width_, h1 = image1->height_;
        int w2 = image2->width_, h2 = image2->height_;

        // ── Step 1: Compute inv(homography) and project image2 corners ──
        float Hinv[9];
        if (!Invert3x3(homography, Hinv))
        {
            ERROR("Homography is too close to singular to invert");
        }

        float corners2[4][3] = {
            {0.0f, 0.0f, 1.0f},
            {(float)(w2 - 1), 0.0f, 1.0f},
            {(float)(w2 - 1), (float)(h2 - 1), 1.0f},
            {0.0f, (float)(h2 - 1), 1.0f}};

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

        int u0 = (int)floorf(uMin);
        int u1 = (int)ceilf(uMax);
        int v0 = (int)floorf(vMin);
        int v1 = (int)ceilf(vMax);

        int outW = u1 - u0 + 1;
        int outH = v1 - v0 + 1;

        int maxW_ = 2 * std::max(image1->width_, image2->width_);
        int maxH_ = 2 * std::max(image1->height_, image2->height_);
        if (outW > maxW_ || outH > maxH_)
        {
            std::stringstream ss;
            ss << "Warping: warped image too large (" << outW << "x" << outH << "), not attempting warp";
            ERROR(ss.str().c_str());
        }

        if (outW <= 0 || outH <= 0)
        {
            std::stringstream ss;
            ss << "Warping: invalid output image size (" << outW << "x" << outH << ")";
            ERROR(ss.str().c_str());
        }

        float originU = (float)u0;
        float originV = (float)v0;

        // ── Step 3: Allocate GPU images ─────────────────────
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

        int src1StrideElem = (int)(src1Pitch / sizeof(float));
        int src2StrideElem = (int)(src2Pitch / sizeof(float));
        int out1StrideElem = (int)(out1Pitch / sizeof(float));

        float H00 = homography[0], H01 = homography[1], H02 = homography[2];
        float H10 = homography[3], H11 = homography[4], H12 = homography[5];
        float H20 = homography[6], H21 = homography[7], H22 = homography[8];

        dim3 threads(16, 16);
        dim3 blocks((outW + threads.x - 1) / threads.x, (outH + threads.y - 1) / threads.y);

        WarpDualKernel<<<blocks, threads>>>(
            d_src1Guard.get(), d_out1Guard.get(), w1, h1, src1StrideElem,
            d_src2Guard.get(), d_out2Guard.get(), w2, h2, src2StrideElem,
            outW, outH, out1StrideElem, originU, originV,
            H00, H01, H02,
            H10, H11, H12,
            H20, H21, H22);

        safeCall(cudaDeviceSynchronize());

        // ── Step 4: Fill output structs, release GPU ownership ──
        warped_image1->strided_img_ = d_out1Guard.release();
        warped_image1->width_ = outW;
        warped_image1->height_ = outH;
        warped_image1->stride_ = out1Pitch;

        warped_image2->strided_img_ = d_out2Guard.release();
        warped_image2->width_ = outW;
        warped_image2->height_ = outH;
        warped_image2->stride_ = out2Pitch;

        // d_src1Guard, d_src2Guard freed automatically
    }); // end cusift_api_guard for WarpImages_GPU
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
        if (extract_options->num_octaves_ > maxOctaves)
        {
            std::cerr << "Warning: Requested number of octaves (" << extract_options->num_octaves_ << ") exceeds the maximum possible (" << maxOctaves << ") for the given image size. Reducing to " << maxOctaves << "." << std::endl;
        }

        // Limit num octaves to 7, quietly
        octaves = std::min(octaves, 7);
        octaves = std::max(octaves, 3); // Ensure at least 3 octaves

        SiftTempMemoryGuard tempMemory(AllocSiftTempMemory(maxW, maxH, octaves));

        // Image 1
        InitSiftData(sift_data1, extract_options->max_keypoints_, true, true);
        CudaImage_Allocate(cuda_image1.get(), w1, h1,
                           p_iAlignUp(w1, 128), false, nullptr, image1->host_img_);
        CudaImage_Download(cuda_image1.get());
        ExtractSift(sift_data1, cuda_image1.get(), octaves,
                    extract_options->init_blur_, extract_options->thresh_,
                    extract_options->lowest_scale_, extract_options->highest_scale_, extract_options->edge_thresh_,
                    tempMemory.get());
        if (extract_options->scale_suppression_radius_ > 0.0f)
            SuppressEmbeddedPoints(sift_data1, extract_options->scale_suppression_radius_);

        // Image 2
        InitSiftData(sift_data2, extract_options->max_keypoints_, true, true);
        CudaImage_Allocate(cuda_image2.get(), w2, h2,
                           p_iAlignUp(w2, 128), false, nullptr, image2->host_img_);
        CudaImage_Download(cuda_image2.get());
        ExtractSift(sift_data2, cuda_image2.get(), octaves,
                    extract_options->init_blur_, extract_options->thresh_,
                    extract_options->lowest_scale_, extract_options->highest_scale_, extract_options->edge_thresh_,
                    tempMemory.get());
        if (extract_options->scale_suppression_radius_ > 0.0f)
            SuppressEmbeddedPoints(sift_data2, extract_options->scale_suppression_radius_);

        // ── Match ────────────────────────────────────────────────────────────
        MatchSiftData_private(sift_data1, sift_data2);

        // ── Find homography ─────────────────────────────────────────────────
        FindGeometricModel(
            sift_data1, homography, num_matches,
            homography_options);
        
        // If any homography values are nan of inf, we should throw an error rather than proceeding to the warp which will produce garbage output and may cause CUDA errors.
        for (int i = 0; i < 8; i++)
        {
            if (!std::isfinite(homography[i]))
            {
                ERROR("Homograpy contains non-finite values");
            }
        }

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
            ERROR("Homography is too close to singular to invert");
        }

        float corners2[4][3] = {
            {0.0f, 0.0f, 1.0f},
            {(float)(w2 - 1), 0.0f, 1.0f},
            {(float)(w2 - 1), (float)(h2 - 1), 1.0f},
            {0.0f, (float)(h2 - 1), 1.0f}};

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

        int maxW_ = 2 * std::max(image1->width_, image2->width_);
        int maxH_ = 2 * std::max(image1->height_, image2->height_);
        if (outW > maxW_ || outH > maxH_)
        {
            std::stringstream ss;
            ss << "Warping: warped image too large (" << outW << "x" << outH << "), not attempting warp";
            ERROR(ss.str().c_str());
        }
        if (outW <= 0 || outH <= 0)
        {
            std::stringstream ss;
            ss << "Warping: invalid output image size (" << outW << "x" << outH << ")";
            ERROR(ss.str().c_str());
        }

        // ── Warp on GPU — source data is already device-resident ────────────
        // Reuse the CudaImageGuard device pointers (no extra upload needed).
        float *d_src1 = cuda_image1.get()->d_data;
        float *d_src2 = cuda_image2.get()->d_data;
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
            outW, outH, static_cast<int>(out1Stride / sizeof(float)), originU, originV,
            H00, H01, H02,
            H10, H11, H12,
            H20, H21, H22);

        safeCall(cudaDeviceSynchronize());

        // ── Copy results to host and fill output structs ────────────────────
        size_t nPixels = (size_t)outW * outH;
        HostPtrGuard<float> out1Guard((float *)malloc(sizeof(float) * nPixels));
        HostPtrGuard<float> out2Guard((float *)malloc(sizeof(float) * nPixels));
        if (!out1Guard.get() || !out2Guard.get())
        {
            ERROR("Warping: return image host allocation failed");
        }

        safeCall(cudaMemcpy2D(out1Guard.get(), outW * sizeof(float), d_out1Guard.get(), out1Stride, outW * sizeof(float), outH, cudaMemcpyDeviceToHost));
        safeCall(cudaMemcpy2D(out2Guard.get(), outW * sizeof(float), d_out2Guard.get(), out2Stride, outW * sizeof(float), outH, cudaMemcpyDeviceToHost));

        warped_image1->host_img_ = out1Guard.release();
        warped_image1->width_ = outW;
        warped_image1->height_ = outH;

        warped_image2->host_img_ = out2Guard.release();
        warped_image2->width_ = outW;
        warped_image2->height_ = outH;
    }); // end cusift_api_guard for ExtractAndMatchAndFindHomographyAndWarp
}

void ExtractAndMatchAndFindHomographyAndWarp_GPU(const Image_t *image1, const Image_t *image2, SiftData *sift_data1, SiftData *sift_data2, float *homography, int *num_matches, const ExtractSiftOptions_t *extract_options, const FindHomographyOptions_t *homography_options, ImageStrided_t *warped_image1, ImageStrided_t *warped_image2)
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
        if (extract_options->num_octaves_ > maxOctaves)
        {
            std::cerr << "Warning: Requested number of octaves (" << extract_options->num_octaves_ << ") exceeds the maximum possible (" << maxOctaves << ") for the given image size. Reducing to " << maxOctaves << "." << std::endl;
        }

        octaves = std::min(octaves, 7);
        octaves = std::max(octaves, 3);

        SiftTempMemoryGuard tempMemory(AllocSiftTempMemory(maxW, maxH, octaves));

        // Image 1
        InitSiftData(sift_data1, extract_options->max_keypoints_, true, true);
        CudaImage_Allocate(cuda_image1.get(), w1, h1,
                           p_iAlignUp(w1, 128), false, nullptr, image1->host_img_);
        CudaImage_Download(cuda_image1.get());
        ExtractSift(sift_data1, cuda_image1.get(), octaves,
                    extract_options->init_blur_, extract_options->thresh_,
                    extract_options->lowest_scale_, extract_options->highest_scale_, extract_options->edge_thresh_,
                    tempMemory.get());
        if (extract_options->scale_suppression_radius_ > 0.0f)
            SuppressEmbeddedPoints(sift_data1, extract_options->scale_suppression_radius_);

        // Image 2
        InitSiftData(sift_data2, extract_options->max_keypoints_, true, true);
        CudaImage_Allocate(cuda_image2.get(), w2, h2,
                           p_iAlignUp(w2, 128), false, nullptr, image2->host_img_);
        CudaImage_Download(cuda_image2.get());
        ExtractSift(sift_data2, cuda_image2.get(), octaves,
                    extract_options->init_blur_, extract_options->thresh_,
                    extract_options->lowest_scale_, extract_options->highest_scale_, extract_options->edge_thresh_,
                    tempMemory.get());
        if (extract_options->scale_suppression_radius_ > 0.0f)
            SuppressEmbeddedPoints(sift_data2, extract_options->scale_suppression_radius_);

        // ── Match ────────────────────────────────────────────────────────────
        MatchSiftData_private(sift_data1, sift_data2);

        // ── Find homography ─────────────────────────────────────────────────
        FindGeometricModel(
            sift_data1, homography, num_matches,
            homography_options);
        
        for (int i = 0; i < 8; i++)
        {
            if (!std::isfinite(homography[i]))
            {
                ERROR("Homograpy contains non-finite values");
            }
        }

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
            ERROR("Homography is too close to singular to invert");
        }

        float corners2[4][3] = {
            {0.0f, 0.0f, 1.0f},
            {(float)(w2 - 1), 0.0f, 1.0f},
            {(float)(w2 - 1), (float)(h2 - 1), 1.0f},
            {0.0f, (float)(h2 - 1), 1.0f}};

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

        int maxW_ = 2 * std::max(image1->width_, image2->width_);
        int maxH_ = 2 * std::max(image1->height_, image2->height_);
        if (outW > maxW_ || outH > maxH_)
        {
            std::stringstream ss;
            ss << "Warping: warped image too large (" << outW << "x" << outH << "), not attempting warp";
            ERROR(ss.str().c_str());
        }
        if (outW <= 0 || outH <= 0)
        {
            std::stringstream ss;
            ss << "Warping: invalid output image size (" << outW << "x" << outH << ")";
            ERROR(ss.str().c_str());
        }

        // ── Warp on GPU — source data is already device-resident ────────────
        float *d_src1 = cuda_image1.get()->d_data;
        float *d_src2 = cuda_image2.get()->d_data;
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

        WarpDualKernel<<<blocks, threads>>>(
            d_src1, d_out1Guard.get(), w1, h1, src1Pitch,
            d_src2, d_out2Guard.get(), w2, h2, src2Pitch,
            outW, outH, static_cast<int>(out1Stride / sizeof(float)), originU, originV,
            H00, H01, H02,
            H10, H11, H12,
            H20, H21, H22);

        safeCall(cudaDeviceSynchronize());

        // ── Fill output structs, release GPU ownership ──────────────────────
        warped_image1->strided_img_ = d_out1Guard.release();
        warped_image1->width_ = outW;
        warped_image1->height_ = outH;
        warped_image1->stride_ = out1Stride;

        warped_image2->strided_img_ = d_out2Guard.release();
        warped_image2->width_ = outW;
        warped_image2->height_ = outH;
        warped_image2->stride_ = out2Stride;
    }); // end cusift_api_guard for ExtractAndMatchAndFindHomographyAndWarp_GPU
}

// ── VRAM estimation helpers ─────────────────────────────────────────────────

static size_t estimatePyramidBytes(int width, int height, int numOctaves)
{
    const int nd = 5 + 3; // NUM_SCALES + 3
    int w = width;
    int h = height;
    int p = p_iAlignUp(w, 128);
    long long size    = (long long)h * p;       // image buffers
    long long sizeTmp = (long long)nd * h * p;  // laplace buffers
    for (int i = 0; i < numOctaves; i++)
    {
        w /= 2;
        h /= 2;
        p = p_iAlignUp(w, 128);
        size    += (long long)h * p;
        sizeTmp += (long long)nd * h * p;
    }
    size += sizeTmp;
    // Actual allocation: cudaMallocPitch with width=4096, height=(size+4095)/4096
    long long rows = (size + 4095) / 4096;
    return (size_t)(rows * 4096 * sizeof(float));
}

static size_t estimateCudaImageBytes(int width, int height)
{
    // CudaImage_Allocate uses cudaMallocPitch; approximate pitch as iAlignUp(width, 128)
    int pitch = p_iAlignUp(width, 128);
    return (size_t)pitch * height * sizeof(float);
}

static size_t estimateSiftDataDeviceBytes(int maxKeypoints)
{
    return (size_t)sizeof(SiftPoint) * maxKeypoints;
}

static size_t estimateContextBytes()
{
    const size_t counterBytes   = (8 * 2 + 1) * sizeof(unsigned int);
    const size_t laplaceBytes   = 8 * 12 * 16 * sizeof(float);
    const size_t scaleDownBytes = 5 * sizeof(float);
    const size_t lowPassBytes   = (2 * 4 + 1) * sizeof(float); // LOWPASS_R = 4
    return counterBytes + laplaceBytes + scaleDownBytes + lowPassBytes;
}

static int clampOctaves(int requested, int width, int height)
{
    int minDim = std::min(width, height);
    int maxOctaves = (int)std::floor(std::log2((double)minDim)) - 3;
    int octaves = std::min(requested, maxOctaves);
    octaves = std::min(octaves, 7);
    octaves = std::max(octaves, 3);
    return octaves;
}

size_t EstimateVramExtractSift(int image_width, int image_height, const ExtractSiftOptions_t *options)
{
    int octaves = clampOctaves(options->num_octaves_, image_width, image_height);

    size_t siftBytes    = estimateSiftDataDeviceBytes(options->max_keypoints_);
    size_t imageBytes   = estimateCudaImageBytes(image_width, image_height);
    size_t pyramidBytes = estimatePyramidBytes(image_width, image_height, octaves);
    size_t contextBytes = estimateContextBytes();

    return siftBytes + imageBytes + pyramidBytes + contextBytes;
}

size_t EstimateVramMatchSift(int max_keypoints1, int max_keypoints2)
{
    return estimateSiftDataDeviceBytes(max_keypoints1)
         + estimateSiftDataDeviceBytes(max_keypoints2);
}

size_t EstimateVramFindHomography(int max_keypoints, const FindHomographyOptions_t *options)
{
    // Existing SiftData that must be resident
    size_t siftBytes = estimateSiftDataDeviceBytes(max_keypoints);

    // Temporary RANSAC buffers (mirroring matching.cu)
    int numPtsUp = ((max_keypoints + 15) / 16) * 16;
    int numLoops = ((options->num_loops_ + 15) / 16) * 16;

    size_t coordBytes    = 4 * sizeof(float) * numPtsUp;

    size_t randPtsBytes, modelBytes;
    if (options->model_type_ == CUSIFT_MODEL_SIMILARITY) {
        randPtsBytes = 2 * sizeof(int)   * numLoops;
        modelBytes   = 4 * sizeof(float) * numLoops;
    } else {
        randPtsBytes = 4 * sizeof(int)   * numLoops;
        modelBytes   = 8 * sizeof(float) * numLoops;
    }

    return siftBytes + coordBytes + randPtsBytes + modelBytes;
}

size_t EstimateVramWarpImages(int image_width1, int image_height1, int image_width2, int image_height2)
{
    // Source images on device (pitch-aligned)
    size_t src1 = estimateCudaImageBytes(image_width1, image_height1);
    size_t src2 = estimateCudaImageBytes(image_width2, image_height2);

    // Worst-case output canvas: 2x the larger dimension in each axis
    int outW = 2 * std::max(image_width1, image_width2);
    int outH = 2 * std::max(image_height1, image_height2);
    size_t out1 = estimateCudaImageBytes(outW, outH);
    size_t out2 = out1; // same canvas size

    return src1 + src2 + out1 + out2;
}

size_t EstimateVramFullPipeline(int image_width1, int image_height1,
                                int image_width2, int image_height2,
                                const ExtractSiftOptions_t *extract_options,
                                const FindHomographyOptions_t *homography_options)
{
    // SiftData for both images persists across all stages
    size_t siftBoth = estimateSiftDataDeviceBytes(extract_options->max_keypoints_) * 2;

    // Peak during extraction (image 1 or 2, whichever is larger)
    int maxW = std::max(image_width1, image_width2);
    int maxH = std::max(image_height1, image_height2);
    int octaves = clampOctaves(extract_options->num_octaves_, maxW, maxH);

    size_t extractPeak = siftBoth
                       + estimateCudaImageBytes(maxW, maxH)
                       + estimatePyramidBytes(maxW, maxH, octaves)
                       + estimateContextBytes();

    // Peak during homography (SiftData + RANSAC temporaries)
    size_t homographyPeak = siftBoth
                          + EstimateVramFindHomography(extract_options->max_keypoints_, homography_options)
                          - estimateSiftDataDeviceBytes(extract_options->max_keypoints_); // avoid double-counting the one SiftData

    // Peak during warping (SiftData already freed in convenience funcs, but be conservative)
    size_t warpPeak = siftBoth
                    + EstimateVramWarpImages(image_width1, image_height1, image_width2, image_height2);

    return std::max({extractPeak, homographyPeak, warpPeak});
}