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
void ExtractSift(SiftData *siftData, CudaImage *img, int numOctaves, float initBlur, float thresh, float lowestScale, float edgeLimit, float *tempMemory);
void InitSiftData(SiftData *data, int num, bool host, bool dev);
void FreeSiftData(SiftData *data);
double MatchSiftData_private(SiftData *data1, SiftData *data2);
double FindHomography_private(SiftData *data, float *homography, int *numMatches, int numLoops, float minScore, float maxAmbiguity, float thresh);


#ifdef __cplusplus
}
#endif

#endif
