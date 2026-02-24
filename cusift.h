#ifndef CUSIFT_H
#define CUSIFT_H

// ── Export / import macros ──────────────────────────────
#ifdef CUSIFT_STATIC
  #define CUSIFT_API
#elif defined(_WIN32)
  #ifdef CUSIFT_EXPORTS
    #define CUSIFT_API __declspec(dllexport)
  #else
    #define CUSIFT_API __declspec(dllimport)
  #endif
#elif __GNUC__ >= 4
  #define CUSIFT_API __attribute__((visibility("default")))
#else
  #define CUSIFT_API
#endif

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

/**
 * @brief Options controlling SIFT feature extraction.
 *
 * These parameters govern the Difference-of-Gaussians (DoG) keypoint
 * detector and the SIFT descriptor computation.  Typical defaults are
 * shown in square brackets after each field.
 *
 * ── Detector thresholds ──────────────────────────────────────────────
 *
 *   thresh_        [~0.02-0.10]
 *       Contrast threshold applied to DoG extrema.  A candidate
 *       keypoint is accepted only when its absolute DoG response
 *       exceeds this value.  Higher values reject low-contrast
 *       features and produce fewer, more stable keypoints.
 *       Lower values retain weaker features at the cost of more
 *       noise.
 *
 *   lowest_scale_  [~0.0]
 *       Minimum feature scale (in pixels of the original image)
 *       that will be kept.  Keypoints whose estimated scale is
 *       below this cutoff are discarded.  Set to 0.0 to keep all
 *       detected scales.  Increasing this suppresses very
 *       fine-grained features.
 *
 *   edge_thresh_   [~10.0]
 *       Edge rejection threshold (ratio of principal curvatures).
 *       Candidates whose (trace²/determinant) of the 2×2 Hessian
 *       exceeds this limit are considered edge responses rather
 *       than corners and are discarded.  Lower values are stricter
 *       (reject more edges); higher values are more permissive.
 *       Lowe's original paper uses (r+1)²/r with r=10, giving
 *       a value of ~12.1.
 *
 * ── Scale-space construction ─────────────────────────────────────────
 *
 *   init_blur_     [~1.0]
 *       Assumed blur level (sigma) of the input image.  The
 *       library applies a low-pass filter so that the effective
 *       blur of the base image matches the first scale of the
 *       Gaussian pyramid. A value of 0.0
 *       means the input is essentially unblurred.
 *
 *   num_octaves_   [5]
 *       Number of octave levels in the scale-space pyramid.  Each
 *       successive octave halves the image resolution.  The
 *       library will silently clamp this to the maximum feasible
 *       value based on the image dimensions (approximately
 *       log2(min(width,height)) - 3), and emit a warning if the
 *       requested count exceeds that limit.  More octaves detect
 *       larger-scale features but increase computation.
 *
 * ── Capacity ─────────────────────────────────────────────────────────
 *
 *   max_keypoints_ [~8192]
 *       Maximum number of keypoints that will be returned.  This
 *       controls the size of the pre-allocated SiftData buffers
 *       on both host and device.  If the detector finds more
 *       candidates than this limit, the excess are silently
 *       dropped.  Set this high enough for your application to
 *       avoid losing valid features.
 */
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

    unsigned int seed_; // 0 = non-deterministic (random_device)
} FindHomographyOptions_t;

/**
 * @brief Initialize the CUDA SIFT library. Must be called before any other functions. All it does it find a valid device.
 * 
 */
CUSIFT_API void InitializeCudaSift();

/**
 * @brief Extract SIFT features from an image. The caller is responsible for freeing the SiftData using DeleteSiftData() when done.
 * 
 * @param image Pointer to the input image.
 * @param sift_data Pointer to the SiftData structure where the extracted features will be stored.
 * @param options Pointer to the ExtractSiftOptions_t structure containing extraction parameters.
 */
CUSIFT_API void ExtractSiftFromImage(const Image_t* image, SiftData* sift_data, const ExtractSiftOptions_t* options);

/**
 * @brief Match SIFT features between two SiftData structures. The match results are stored in the 'match', 'match_xpos', 'match_ypos', and 'match_error' fields of the SiftPoint structures in data1. The caller is responsible for ensuring that data1 and data2 are properly initialized and contain valid SIFT features before calling this function.
 * 
 * @param data1 Pointer to the first SiftData structure.
 * @param data2 Pointer to the second SiftData structure.
 */
CUSIFT_API void MatchSiftData(SiftData* data1, SiftData* data2);

/**
 * @brief Find a homography transformation between matched SIFT features in the given SiftData structure. The homography is returned as a 3x3 matrix in row-major order in the 'homography' output parameter. The number of matches used to compute the homography is returned in the 'num_matches' output parameter. The caller is responsible for ensuring that the SiftData structure contains valid matched SIFT features before calling this function.
 * 
 * @param data Pointer to the SiftData structure containing matched SIFT features.
 * @param homography Pointer to a 3x3 matrix in row-major order where the computed homography will be stored.
 * @param num_matches Pointer to an integer where the number of matches used to compute the homography will be stored.
 * @param options Pointer to the FindHomographyOptions_t structure containing homography computation parameters.
 */
CUSIFT_API void FindHomography(SiftData* data, float* homography, int* num_matches, const FindHomographyOptions_t* options);

/**
 * @brief Given the computed homography, warp the input images to align them. The warped images are returned in the 'warped_image1' and 'warped_image2' output parameters. The caller is responsible for ensuring that the input images and homography are valid before calling this function, and for freeing any resources associated with the warped images when done.
 * To free the warped images call the cstdlib free() function on the 'host_img_' field of the Image_t structures, and set the pointer to nullptr to avoid dangling pointers. We use malloc to allocate space for the warped images.
 * 
 * @param image1 Pointer to the first input image.
 * @param image2 Pointer to the second input image.
 * @param homography Pointer to a 3x3 matrix in row-major order representing the homography transformation.
 * @param warped_image1 Pointer to the Image_t structure where the warped first image will be stored.
 * @param warped_image2 Pointer to the Image_t structure where the warped second image will be stored.
 * @param useGPU Boolean flag indicating whether to use GPU acceleration for the warping operation.
 */
CUSIFT_API void WarpImages(const Image_t* image1, const Image_t* image2, const float* homography, Image_t* warped_image1, Image_t* warped_image2, bool useGPU);

/**
 * @brief Delete a SiftData structure and free all associated resources. After calling this function, the SiftData pointer should not be used again unless it is re-initialized. The caller is responsible for ensuring that the SiftData structure was properly initialized and contains valid data before calling this function.
 * 
 * @param sift_data Pointer to the SiftData structure to be deleted.
 */
CUSIFT_API void DeleteSiftData(SiftData* sift_data);

/**
 * @brief Save SIFT features from a SiftData structure to a json file.
 * 
 * @param filename Pointer to the name of the file where the SIFT features will be saved.
 * @param sift_data Pointer to the SiftData structure containing the SIFT features to be saved.
 */
CUSIFT_API void SaveSiftData(const char* filename, const SiftData* sift_data);

/**
 * @brief Extract Sift features from two images and match them. This is a convenience function that combines ExtractSiftFromImage() and MatchSiftData() into a single call. The caller is responsible for freeing the SiftData structures using DeleteSiftData() when done.
 * 
 * @param image1 Pointer to the first input image.
 * @param image2 Pointer to the second input image.
 * @param sift_data1 Pointer to the SiftData structure where the extracted features from the first image will be stored.
 * @param sift_data2 Pointer to the SiftData structure where the extracted features from the second image will be stored.
 * @param extract_options ExtractSiftOptions_t structure containing parameters for SIFT feature extraction. The same options will be used for both images.
 */
CUSIFT_API void ExtractAndMatchSift(const Image_t* image1, const Image_t* image2, SiftData* sift_data1, SiftData* sift_data2, const ExtractSiftOptions_t* extract_options);

/**
 * @brief Extract Sift features from two images, match them, and find a homography transformation between the matched features. This is a convenience function that combines ExtractSiftFromImage(), MatchSiftData(), and FindHomography() into a single call. The caller is responsible for freeing the SiftData structures using DeleteSiftData() when done.
 * 
 * @param image1 Pointer to the first input image.
 * @param image2 Pointer to the second input image.
 * @param sift_data1 Pointer to the SiftData structure where the extracted features from the first image will be stored.
 * @param sift_data2 Pointer to the SiftData structure where the extracted features from the second image will be stored.
 * @param homography Pointer to a 3x3 matrix in row-major order where the computed homography will be stored.
 * @param num_matches Pointer to an integer where the number of matches used to compute the homography will be stored.
 * @param extract_options ExtractSiftOptions_t structure containing parameters for SIFT feature extraction. The same options will be used for both images.
 * @param homography_options HomographyOptions_t structure containing parameters for homography computation.
 */
CUSIFT_API void ExtractAndMatchAndFindHomography(const Image_t* image1, const Image_t* image2, SiftData* sift_data1, SiftData* sift_data2, float* homography, int* num_matches, const ExtractSiftOptions_t* extract_options, const FindHomographyOptions_t* homography_options);


#ifdef __cplusplus
}
#endif

#endif /* CUSIFT_H */
