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
        options->thresh_);
    
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

