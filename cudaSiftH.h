#ifndef CUDASIFTH_H
#define CUDASIFTH_H

#include "cudautils.h"
#include "cudaImage.h"

//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

int ExtractSiftLoop(SiftData *siftData, CudaImage *img, int numOctaves, float initBlur, float thresh, float lowestScale, float edgeLimit, float subsampling, float *memoryTmp, float *memorySub);
void ExtractSiftOctave(SiftData *siftData, CudaImage *img, int octave, float thresh, float lowestScale, float edgeLimit, float subsampling, float *memoryTmp);
double ScaleDown(CudaImage *res, CudaImage *src, float variance);
double ComputeOrientations(cudaTextureObject_t texObj, CudaImage *src, SiftData *siftData, int octave);
double ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData *siftData, float subsampling, int octave);
double LowPass(CudaImage *res, CudaImage *src, float scale);
void PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel);
double LaplaceMulti(cudaTextureObject_t texObj, CudaImage *baseImage, CudaImage *results, int octave);
double FindPointsMulti(CudaImage *sources, SiftData *siftData, float thresh, float edgeLimit, float factor, float lowestScale, float subsampling, int octave);

#endif
