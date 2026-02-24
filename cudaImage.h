//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//

#ifndef CUDAIMAGE_H
#define CUDAIMAGE_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    int width, height;
    int pitch;
    float *h_data;
    float *d_data;
    float *t_data;
    bool d_internalAlloc;
    bool h_internalAlloc;
} CudaImage;

void CudaImage_init(CudaImage *img);
void CudaImage_destroy(CudaImage *img);
void CudaImage_Allocate(CudaImage *img, int width, int height, int pitch, bool withHost, float *devMem, float *hostMem);
void CudaImage_Download(CudaImage *img);
void CudaImage_Readback(CudaImage *img);

int iDivUp(int a, int b);
int iDivDown(int a, int b);
int iAlignUp(int a, int b);
int iAlignDown(int a, int b);

#ifdef __cplusplus
}
#endif

#endif // CUDAIMAGE_H
