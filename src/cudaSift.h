#ifndef CUDASIFT_H
#define CUDASIFT_H

#include "cudaImage.h"
#include "cusift.h"
#include <stdbool.h>


#ifdef __cplusplus
extern "C" {
#endif

void InitCuda(int devNum);
float *AllocSiftTempMemory(int width, int height, int numOctaves);
void FreeSiftTempMemory(float *memoryTmp);
void ExtractSift(SiftData *siftData, CudaImage *img, int numOctaves, float initBlur, float thresh, float lowestScale, float highestScale, float edgeLimit, float *tempMemory);
// Initializes a SiftData and allocates its host/device buffers.
// The struct is treated as WRITE-ONLY: the incoming pointers are not read, so
// a fresh/uninitialized struct is fine. It also means InitSiftData does NOT
// free buffers a previous call allocated. To reuse a SiftData, call
// FreeSiftData() first — otherwise the earlier host/device buffers leak.
void InitSiftData(SiftData *data, int num, bool host, bool dev);
void FreeSiftData(SiftData *data);
void SuppressEmbeddedPoints(SiftData *data, float radiusScale);
double MatchSiftData_private(SiftData *data1, SiftData *data2);
double FindHomography_private(SiftData *data, float *homography, int *numMatches, int numLoops, float minScore, float maxAmbiguity, float thresh, unsigned int seed);
double FindSimilarity_private(SiftData *data, float *homography, int *numMatches, int numLoops, float minScore, float maxAmbiguity, float thresh, unsigned int seed);


#ifdef __cplusplus
}
#endif

#endif
