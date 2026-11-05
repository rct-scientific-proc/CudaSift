#ifndef GEOMFUNCS_H
#define GEOMFUNCS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "cudaSift.h" /* public types plus the internal SiftData / SiftPoint names */

int ImproveHomography(SiftData *data, float *homography, int numLoops,
                      float minScore, float maxAmbiguity, float thresh);

#ifdef __cplusplus
}
#endif

#endif /* GEOMFUNCS_H */
