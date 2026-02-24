#include "cusift.h"
#include "cudautils.h"
#include "cudaImage.h"
#include "cudaSift.h"
#include "geomFuncs.h"
#include "RAII_Gaurds.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

static int p_iAlignUp(int a, int b) { return (a % b != 0) ? (a - a % b + b) : a; }

void InitializeCudaSift()
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if (!nDevices)
    {
        std::cerr << "No CUDA devices available" << std::endl;
        return;
    }
    int devNum = std::min(nDevices - 1, 0);
    safeCall(cudaSetDevice(devNum));
}

void ExtractSiftFromImage(const Image_t* image, SiftData* sift_data, const ExtractSiftOptions_t* options)
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
    CudaImage_Normalize(cuda_image.get());

    // Get the smallest dimension of the image to determine the maximum number of octaves
    int minDim = std::min(image->width_, image->height_);
    int maxOctaves = static_cast<int>(std::floor(std::log2(minDim))) - 3; // Subtract 3 to ensure we have enough pixels in the smallest octave
    int octaves = std::min(options->num_octaves_, maxOctaves);
    if (options->num_octaves_ > maxOctaves)
    {
        // Quitly reduce the number of octaves to the maximum possible based on the image size
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
        tempMemory.get());
}

void MatchSiftData(SiftData* data1, SiftData* data2)
{
    MatchSiftData_private(data1, data2);
}

void FindHomography(SiftData* data, float* homography, int* num_matches, const FindHomographyOptions_t* options)
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
        options->improve_thresh_);
}


void DeleteSiftData(SiftData* sift_data)
{
    // Call FreeSiftData to free device memory
    FreeSiftData(sift_data);
}


void SaveSiftData(const char* filename, const SiftData* sift_data)
{
    if (!sift_data || !sift_data->h_data || sift_data->numPts <= 0)
    {
        std::cerr << "SaveSiftData: no data to save" << std::endl;
        return;
    }

    FILE* f = fopen(filename, "w");
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
        const SiftPoint* pt = &sift_data->h_data[i];
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
            if (j < 127) fprintf(f, ", ");
        }
        fprintf(f, "]\n");
        fprintf(f, "    }%s\n", (i < sift_data->numPts - 1) ? "," : "");
    }

    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    fclose(f);
}

void ExtractAndMatchSift(const Image_t* image1, const Image_t* image2, SiftData* sift_data1, SiftData* sift_data2, const ExtractSiftOptions_t* extract_options)
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
    CudaImage_Normalize(cuda_image1.get());
    ExtractSift(sift_data1, cuda_image1.get(), octaves,
                extract_options->init_blur_, extract_options->thresh_,
                extract_options->lowest_scale_, extract_options->edge_thresh_,
                tempMemory.get());

    // Extract from image 2
    InitSiftData(sift_data2, extract_options->max_keypoints_, true, true);
    CudaImage_Allocate(cuda_image2.get(), image2->width_, image2->height_,
                       p_iAlignUp(image2->width_, 128), false, nullptr, image2->host_img_);
    CudaImage_Download(cuda_image2.get());
    CudaImage_Normalize(cuda_image2.get());
    ExtractSift(sift_data2, cuda_image2.get(), octaves,
                extract_options->init_blur_, extract_options->thresh_,
                extract_options->lowest_scale_, extract_options->edge_thresh_,
                tempMemory.get());

    // Match
    MatchSiftData_private(sift_data1, sift_data2);
}

void ExtractAndMatchAndFindHomography(const Image_t* image1, const Image_t* image2, SiftData* sift_data1, SiftData* sift_data2, float* homography, int* num_matches, const ExtractSiftOptions_t* extract_options, const FindHomographyOptions_t* homography_options)
{
    ExtractAndMatchSift(image1, image2, sift_data1, sift_data2, extract_options);

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
}

