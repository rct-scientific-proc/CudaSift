#ifndef CUDASIFT_H
#define CUDASIFT_H

#include "cudaImage.h"
#include "cusift.h"
#include <stdbool.h>

// The kernels and host pipeline keep the upstream CudaSift type names; they
// are the public types under their 3.0 names.
typedef cusift_keypoint  SiftPoint;
typedef cusift_keypoints SiftData;

#ifdef __cplusplus
extern "C" {
#endif

void InitCuda(int devNum);
float *AllocSiftTempMemory(int width, int height, int numOctaves);
void FreeSiftTempMemory(float *memoryTmp);
void ExtractSift(SiftData *siftData, CudaImage *img, int numOctaves, float initBlur, float thresh, float lowestScale, float highestScale, float edgeLimit, float *tempMemory);
// Initializes a SiftData and allocates (and zeroes) its host/device buffers.
// The struct is treated as WRITE-ONLY: the incoming pointers are not read, so
// InitSiftData does NOT free buffers a previous call allocated. To reuse a
// SiftData, call FreeSiftData() first (the public C API does this for you).
// Throws CusiftError if num is outside [1, INT_MAX / sizeof(SiftPoint)] or an
// allocation fails; on failure the struct is left empty (maxPts == 0).
void InitSiftData(SiftData *data, int num, bool host, bool dev);
void FreeSiftData(SiftData *data);
// Report a warning (severity 1) through the library's error sinks (log file
// and callback).  Never throws and allocates nothing the caller must manage,
// so it is safe from C and during stack unwinding.
void cusift_report_warning(const char *function, const char *message);
void SuppressEmbeddedPoints(SiftData *data, float radiusScale);
double MatchSiftData_private(SiftData *data1, SiftData *data2);
double FindHomography_private(SiftData *data, float *homography, int *numMatches, int numLoops, float minScore, float maxAmbiguity, float thresh, unsigned int seed);
double FindSimilarity_private(SiftData *data, float *homography, int *numMatches, int numLoops, float minScore, float maxAmbiguity, float thresh, unsigned int seed);


#ifdef __cplusplus
}
#endif

#endif
