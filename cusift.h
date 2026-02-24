#ifndef CUSIFT_H
#define CUSIFT_H

// C linkage for easier interoperability with C and other languages
#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    float xpos;
    float ypos;
    float scale;
    float sharpness;
    float edgeness;
    float orientation;
    float score;
    float ambiguity;
    int match;
    float match_xpos;
    float match_ypos;
    float match_error;
    float subsampling;
    float empty[3];
    float data[128];
} SiftPoint;

typedef struct
{
    int numPts; // Number of available Sift points
    int maxPts; // Number of allocated Sift points

    SiftPoint *h_data; // Host (CPU) data
    SiftPoint *d_data; // Device (GPU) data
} SiftData;

typedef struct 
{
    float* host_img_;
    int width_;
    int height_;
} Image_t;

typedef struct
{
    float thresh_;
    float lowest_scale_;
    float edge_thresh_;
    float init_blur_;
    int max_keypoints_;
    int num_octaves_;
} ExtractSiftOptions_t;

typedef struct
{
    int num_loops_;
    float min_score_;
    float max_ambiguity_;
    float thresh_;

    int improve_num_loops_;
    float improve_min_score_;
    float improve_max_ambiguity_;
    float improve_thresh_;
} FindHomographyOptions_t;

void InitializeCudaSift();

void ExtractSiftFromImage(const Image_t* image, SiftData* sift_data, const ExtractSiftOptions_t* options);

void MatchSiftData(SiftData* data1, SiftData* data2);

void FindHomography(SiftData* data, float* homography, int* num_matches, const FindHomographyOptions_t* options);

void DeleteSiftData(SiftData* sift_data);

#ifdef __cplusplus
}
#endif

#endif /* CUSIFT_H */
