//********************************************************//
// CUDA SIFT extractor by Marten Bjorkman aka Celebrandil //
//********************************************************//  

#ifndef CUDASIFTD_H
#define CUDASIFTD_H

#define NUM_SCALES      5

// Scale down thread block width
// Chosen so that (SCALEDOWN_W + 4) is a multiple of the warp size (32),
// otherwise we launch with 68 threads -> 3 warps with the 3rd running
// only 4 active lanes for the entire kernel.
#define SCALEDOWN_W    60 // was 64; 60+4 = 64 = 2 warps

// Scale down thread block height
#define SCALEDOWN_H    16 // 8

// Scale up thread block width
#define SCALEUP_W      64

// Scale up thread block height
#define SCALEUP_H       8

// Find point thread block width
#define MINMAX_W       30 //32 

// Find point thread block height
#define MINMAX_H        8 //16 
 
// Laplace thread block width
#define LAPLACE_W     128 // 56

// Laplace rows per thread
#define LAPLACE_H       4

// Number of laplace scales
#define LAPLACE_S   (NUM_SCALES+3)

// Laplace filter kernel radius
#define LAPLACE_R       4

#define LOWPASS_W      24 //56
#define LOWPASS_H      32 //16
#define LOWPASS_R       4

///////////////////////////////////////////////////////////////////////////////
// Per-call device context replacing module-level __constant__/__device__
// globals.  Allocated and freed for each ExtractSift() invocation so that
// multiple host threads / processes can use the library concurrently without
// stomping on shared GPU state.
///////////////////////////////////////////////////////////////////////////////
struct SiftDeviceContext {
    unsigned int* d_pointCounter;      // [8*2+1]        replaces __device__   d_PointCounter
    float*        d_laplaceKernel;     // [8*12*16]      replaces __constant__ d_LaplaceKernel
    float*        d_scaleDownKernel;   // [5]            replaces __constant__ d_ScaleDownKernel
    float*        d_lowPassKernel;     // [2*LOWPASS_R+1] replaces __constant__ d_LowPassKernel
    int           maxNumPoints;        //                 replaces __constant__ d_MaxNumPoints
};

//====================== Number of threads ====================//
// ScaleDown:               SCALEDOWN_W + 4
// LaplaceMulti:            (LAPLACE_W+2*LAPLACE_R)*LAPLACE_S
// FindPointsMulti:         MINMAX_W + 2
// ComputeOrientations:     128
// ExtractSiftDescriptors:  256

//====================== Number of blocks ====================//
// ScaleDown:               (width/SCALEDOWN_W) * (height/SCALEDOWN_H)
// LaplceMulti:             (width+2*LAPLACE_R)/LAPLACE_W * height
// FindPointsMulti:         (width/MINMAX_W)*NUM_SCALES * (height/MINMAX_H)
// ComputeOrientations:     numpts
// ExtractSiftDescriptors:  numpts

#endif
