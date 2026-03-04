#ifndef CUDASIFTH_H
#define CUDASIFTH_H

#include "cudautils.h"
#include "cudaImage.h"

struct SiftDeviceContext;  // defined in cudaSiftD.h

//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

int ExtractSiftLoop(SiftData *siftData, CudaImage *img, int numOctaves, float initBlur, float thresh, float lowestScale, float edgeLimit, float subsampling, float *memoryTmp, float *memorySub, SiftDeviceContext &ctx);
void ExtractSiftOctave(SiftData *siftData, CudaImage *img, int octave, float thresh, float lowestScale, float edgeLimit, float subsampling, float *memoryTmp, SiftDeviceContext &ctx);
double ScaleDown(CudaImage *res, CudaImage *src, float variance, SiftDeviceContext &ctx);
double ComputeOrientations(cudaTextureObject_t texObj, CudaImage *src, SiftData *siftData, int octave, SiftDeviceContext &ctx);
double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData *siftData, float subsampling, int octave, SiftDeviceContext &ctx);
double LowPass(CudaImage *res, CudaImage *src, float scale, SiftDeviceContext &ctx);
void PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel);
double LaplaceMulti(cudaTextureObject_t texObj, CudaImage *baseImage, CudaImage *results, int octave, SiftDeviceContext &ctx);
double FindPointsMulti(CudaImage *sources, SiftData *siftData, float thresh, float edgeLimit, float factor, float lowestScale, float subsampling, int octave, SiftDeviceContext &ctx);

/**
 * @brief Remove keypoints that are spatially embedded inside a larger-scale keypoint.
 *
 * When multiple SIFT keypoints cluster around the same image feature at
 * different scales, this function keeps only the one with the largest
 * scale and discards the rest.  A smaller keypoint is "embedded" in a
 * larger one when the Euclidean distance between their positions is less
 * than @c radiusScale × <em>larger_scale</em>.
 *
 * Operates on the host-side @c h_data array.  After suppression the
 * device-side @c d_data is re-uploaded to stay in sync.
 *
 * @param data         SiftData whose @c h_data will be pruned in-place.
 * @param radiusScale  Multiplier on the larger keypoint's scale that
 *                     defines the suppression radius (default 6.0f,
 *                     which matches the SIFT descriptor patch radius).
 */
void SuppressEmbeddedPoints(SiftData *data, float radiusScale = 6.0f);

#endif
