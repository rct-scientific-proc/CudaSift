//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//

#include <cstdio>

#include "cudautils.h"
#include "cudaImage.h"

int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }
int iDivDown(int a, int b) { return a / b; }
int iAlignUp(int a, int b) { return (a % b != 0) ? (a - a % b + b) : a; }
int iAlignDown(int a, int b) { return a - a % b; }

// Keep
void CudaImage_init(CudaImage *img)
{
    img->width = 0;
    img->height = 0;
    img->pitch = 0;
    img->h_data = NULL;
    img->d_data = NULL;
    img->t_data = NULL;
    img->d_internalAlloc = false;
    img->h_internalAlloc = false;
}

void CudaImage_destroy(CudaImage *img)
{
    if (img->d_internalAlloc && img->d_data != NULL)
        safeCall(cudaFree(img->d_data));
    img->d_data = NULL;
    if (img->h_internalAlloc && img->h_data != NULL)
        free(img->h_data);
    img->h_data = NULL;
    if (img->t_data != NULL)
        safeCall(cudaFreeArray((cudaArray *)img->t_data));
    img->t_data = NULL;
}

void CudaImage_Allocate(CudaImage *img, int w, int h, int p, bool host, float *devmem, float *hostmem)
{
    img->width = w;
    img->height = h;
    img->pitch = p;
    img->d_data = devmem;
    img->h_data = hostmem;
    img->t_data = NULL;
    if (devmem == NULL)
    {
        safeCall(cudaMallocPitch((void **)&img->d_data, (size_t *)&img->pitch, (size_t)(sizeof(float) * img->width), (size_t)img->height));
        img->pitch /= sizeof(float);
        if (img->d_data == NULL)
            printf("Failed to allocate device data\n");
        img->d_internalAlloc = true;
    }
    if (host && hostmem == NULL)
    {
        img->h_data = (float *)malloc(sizeof(float) * img->pitch * img->height);
        img->h_internalAlloc = true;
    }
}

// Keep
double CudaImage_Download(CudaImage *img)
{
    int p = sizeof(float) * img->pitch;
    if (img->d_data != NULL && img->h_data != NULL)
        safeCall(cudaMemcpy2D(img->d_data, p, img->h_data, sizeof(float) * img->width, sizeof(float) * img->width, img->height, cudaMemcpyHostToDevice));
}

double CudaImage_Readback(CudaImage *img)
{
    int p = sizeof(float) * img->pitch;
    safeCall(cudaMemcpy2D(img->h_data, sizeof(float) * img->width, img->d_data, p, sizeof(float) * img->width, img->height, cudaMemcpyDeviceToHost));
}

