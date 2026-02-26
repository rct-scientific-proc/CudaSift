/**
 * @file cusift.h
 * @author rct-scientific-proc
 * @brief 
 * @version 0.1
 * @date 2026-02-25
 * 
 * @cite
 * M. Björkman, N. Bergström and D. Kragic, "Detecting, segmenting and tracking unknown objects using multi-label MRF inference", CVIU, 118, pp. 111-127, January 2014
 * 
 * MIT License
 *
 * Copyright (c) 2017 Mårten Björkman
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * 
 * Export Functions:
 * - InitializeCudaSift()   
 * - ExtractSiftFromImage()
 * - MatchSiftData()
 * - FindHomography()
 * - WarpImages()
 * - DeleteSiftData()
 * - CusiftGetLastErrorString()
 * - CusiftHadError()
 * - SaveSiftData()
 * - ExtractAndMatchSift()
 * - ExtractAndMatchAndFindHomography()
 * - ExtractAndMatchAndFindHomographyAndWarp()
 *
 */


#ifndef CUSIFT_H
#define CUSIFT_H

// -- Export / import macros ------------------------------
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
extern "C"
{
#endif

    /**
     * @brief Return the last error from the library, including file, line, and
     *        human-readable message.  All three output parameters are optional
     *        (pass NULL to skip).
     *
     * @param line_number  Receives the source line where the error originated.
     * @param filename     256-char buffer that receives the source filename.
     * @param error_message 256-char buffer that receives the error description.
     */
    CUSIFT_API void CusiftGetLastErrorString(int *line_number,
                                             char filename[256],
                                             char error_message[256]);

    /**
     * @brief Check whether the most recent library call encountered an error.
     *
     * Every public API function clears the error flag on entry, so this always
     * reflects the status of the *last* call.
     *
     * @return Non-zero if an error occurred, 0 otherwise.
     */
    CUSIFT_API int CusiftHadError(void);

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
        float *host_img_;
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
     * -- Detector thresholds ----------------------------------------------
     *
     *   thresh_        [~2.0-5.0]
     *       Contrast threshold applied to DoG extrema.  A candidate
     *       keypoint is accepted only when its absolute DoG response
     *       exceeds this value.  Higher values reject low-contrast
     *       features and produce fewer, more stable keypoints.
     *       Lower values retain weaker features at the cost of more
     *       noise. Think of a higher value as a bigger magnitude difference
     *       between the keypoint and its neighbors in the DoG scale-space.
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
     * -- Scale-space construction -----------------------------------------
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
     * -- Capacity ---------------------------------------------------------
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

    /**
     * @brief Options controlling RANSAC-based homography estimation and
     *        iterative refinement.
     *
     * The homography pipeline has two stages:
     *
     *   1. **RANSAC estimation** – randomly samples 4 matched point pairs,
     *      computes a candidate homography, and counts inliers.  Repeated
     *      for `num_loops_` iterations; the candidate with the most
     *      inliers wins.
     *
     *   2. **Iterative refinement** – takes the RANSAC result and
     *      re-estimates the homography using weighted least-squares
     *      (Cholesky solve) over all inlier correspondences.  Repeated
     *      for `improve_num_loops_` iterations to converge on a more
     *      accurate solution.
     *
     * -- RANSAC stage -----------------------------------------------------
     *
     *   num_loops_      [~1000]
     *       Number of RANSAC iterations.  Each iteration draws 4
     *       random correspondences, computes a homography, and counts
     *       how many other correspondences agree within `thresh_`
     *       pixels.  Internally rounded up to a multiple of 16 for
     *       GPU occupancy.  More iterations increase the chance of
     *       finding the best model but take longer.
     *
     *   min_score_      [~0.0]
     *       Minimum match score a correspondence must have to be
     *       eligible for RANSAC sampling.  The `score` field on each
     *       SiftPoint is set during matching.  Correspondences with
     *       `score < min_score_` are excluded from sampling and
     *       cannot be drawn as one of the 4 random points.  Raise
     *       this to restrict RANSAC to high-confidence matches.
     *
     *   max_ambiguity_  [~1.0]
     *       Maximum match ambiguity a correspondence may have and
     *       still be used for RANSAC sampling.  Ambiguity is the
     *       ratio between the best and second-best match distances;
     *       values closer to 1.0 are more ambiguous.
     *       Correspondences with `ambiguity > max_ambiguity_` are
     *       excluded.  Lower values enforce stricter uniqueness.
     *
     *   thresh_         [~5.0, pixels]
     *       Inlier distance threshold for RANSAC.  After projecting
     *       a point through a candidate homography, if the reprojection
     *       error (Euclidean pixel distance) is less than `thresh_`
     *       the correspondence is counted as an inlier.  Larger values
     *       tolerate more noise but risk accepting wrong models;
     *       smaller values are stricter.  Internally squared before
     *       comparison.
     *
     * -- Refinement stage -------------------------------------------------
     *
     *   improve_num_loops_      [~3-5]
     *       Number of iterative re-estimation rounds.  Each round
     *       recomputes the homography using weighted least-squares
     *       over all correspondences, applying binary inlier/outlier
     *       weights based on `improve_thresh_`.  A few iterations
     *       (3–5) typically suffice for convergence.
     *
     *   improve_min_score_      [~0.0]
     *       Minimum match score for a correspondence to participate
     *       in the refinement solve.  Same semantics as `min_score_`
     *       but applied independently; you may choose a different
     *       (often lower) threshold here to include more points in
     *       the least-squares fit.
     *
     *   improve_max_ambiguity_  [~1.0]
     *       Maximum ambiguity for refinement participation.  Same
     *       semantics as `max_ambiguity_` but applied independently
     *       during the refinement stage.
     *
     *   improve_thresh_         [~3.0, pixels]
     *       Inlier distance threshold for refinement.  During each
     *       re-estimation round, correspondences whose reprojection
     *       error exceeds this threshold receive zero weight (binary
     *       outlier rejection).  Typically set tighter than the RANSAC
     *       `thresh_` to sharpen the final model.  Internally squared
     *       before comparison.
     *
     * -- Reproducibility --------------------------------------------------
     *
     *   seed_           [0]
     *       Seed for the PRNG that generates random 4-point samples
     *       in RANSAC.  Set to a non-zero value for deterministic,
     *       reproducible results.  When 0, a hardware random device
     *       is used to seed the generator, making each run
     *       non-deterministic.
     */
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
    CUSIFT_API void ExtractSiftFromImage(const Image_t *image, SiftData *sift_data, const ExtractSiftOptions_t *options);

    /**
     * @brief Match SIFT features between two SiftData structures. The match results are stored in the 'match', 'match_xpos', 'match_ypos', and 'match_error' fields of the SiftPoint structures in data1. The caller is responsible for ensuring that data1 and data2 are properly initialized and contain valid SIFT features before calling this function.
     *
     * @param data1 Pointer to the first SiftData structure.
     * @param data2 Pointer to the second SiftData structure.
     */
    CUSIFT_API void MatchSiftData(SiftData *data1, SiftData *data2);

    /**
     * @brief Find a homography transformation between matched SIFT features in the given SiftData structure. The homography is returned as a 3x3 matrix in row-major order in the 'homography' output parameter. The number of matches used to compute the homography is returned in the 'num_matches' output parameter. The caller is responsible for ensuring that the SiftData structure contains valid matched SIFT features before calling this function.
     *
     * @param data Pointer to the SiftData structure containing matched SIFT features.
     * @param homography Pointer to a 3x3 matrix in row-major order where the computed homography will be stored.
     * @param num_matches Pointer to an integer where the number of matches used to compute the homography will be stored.
     * @param options Pointer to the FindHomographyOptions_t structure containing homography computation parameters.
     */
    CUSIFT_API void FindHomography(SiftData *data, float *homography, int *num_matches, const FindHomographyOptions_t *options);

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
    CUSIFT_API void WarpImages(const Image_t *image1, const Image_t *image2, const float *homography, Image_t *warped_image1, Image_t *warped_image2, bool useGPU);

    /**
     * @brief Delete a SiftData structure and free all associated resources. After calling this function, the SiftData pointer should not be used again unless it is re-initialized. The caller is responsible for ensuring that the SiftData structure was properly initialized and contains valid data before calling this function.
     *
     * @param sift_data Pointer to the SiftData structure to be deleted.
     */
    CUSIFT_API void DeleteSiftData(SiftData *sift_data);

    /**
     * @brief Free the pixel buffer owned by an Image_t structure.
     *
     * This is intended for images whose ``host_img_`` was allocated by the
     * library (e.g. the warped output images from WarpImages()).  After
     * this call ``image->host_img_`` is set to NULL and the dimensions
     * are zeroed.
     *
     * @param image Pointer to the Image_t whose pixel buffer should be freed.
     */
    CUSIFT_API void FreeImage(Image_t *image);

    /**
     * @brief Save SIFT features from a SiftData structure to a json file.
     *
     * @param filename Pointer to the name of the file where the SIFT features will be saved.
     * @param sift_data Pointer to the SiftData structure containing the SIFT features to be saved.
     */
    CUSIFT_API void SaveSiftData(const char *filename, const SiftData *sift_data);

    /**
     * @brief Extract Sift features from two images and match them. This is a convenience function that combines ExtractSiftFromImage() and MatchSiftData() into a single call. The caller is responsible for freeing the SiftData structures using DeleteSiftData() when done.
     *
     * @param image1 Pointer to the first input image.
     * @param image2 Pointer to the second input image.
     * @param sift_data1 Pointer to the SiftData structure where the extracted features from the first image will be stored.
     * @param sift_data2 Pointer to the SiftData structure where the extracted features from the second image will be stored.
     * @param extract_options ExtractSiftOptions_t structure containing parameters for SIFT feature extraction. The same options will be used for both images.
     */
    CUSIFT_API void ExtractAndMatchSift(const Image_t *image1, const Image_t *image2, SiftData *sift_data1, SiftData *sift_data2, const ExtractSiftOptions_t *extract_options);

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
    CUSIFT_API void ExtractAndMatchAndFindHomography(const Image_t *image1, const Image_t *image2, SiftData *sift_data1, SiftData *sift_data2, float *homography, int *num_matches, const ExtractSiftOptions_t *extract_options, const FindHomographyOptions_t *homography_options);

    /**
     * @brief Full pipeline: Extract Sift features from two images, match them, find a homography transformation between the matched features, and warp the input images to align them. This is a convenience function that combines ExtractSiftFromImage(), MatchSiftData(), FindHomography(), and WarpImages() into a single call. The caller is responsible for freeing the SiftData structures using DeleteSiftData() when done, and for freeing any resources associated with the warped images when done.
     * This function is useful for applications that require both feature matching and image alignment, such as panorama stitching or object recognition. It provides a streamlined interface for performing the entire workflow with a single function call.
     *
     * @param image1 Pointer to the first input image.
     * @param image2 Pointer to the second input image.
     * @param sift_data1 Pointer to the SiftData structure where the extracted features from the first image will be stored.
     * @param sift_data2 Pointer to the SiftData structure where the extracted features from the second image will be stored.
     * @param homography Pointer to a 3x3 matrix in row-major order where the computed homography will be stored.
     * @param num_matches Pointer to an integer where the number of matches used to compute the homography will be stored.
     * @param extract_options ExtractSiftOptions_t structure containing parameters for SIFT feature extraction. The same options will be used for both images.
     * @param homography_options FindHomographyOptions_t structure containing parameters for homography computation.
     * @param warped_image1 Pointer to the Image_t structure where the warped first image will be stored. The caller is responsible for freeing any resources associated with the warped images when done.
     * @param warped_image2 Pointer to the Image_t structure where the warped second image will be stored. The caller is responsible for freeing any resources associated with the warped images when done.
     */
    CUSIFT_API void ExtractAndMatchAndFindHomographyAndWarp(const Image_t *image1, const Image_t *image2, SiftData *sift_data1, SiftData *sift_data2, float *homography, int *num_matches, const ExtractSiftOptions_t *extract_options, const FindHomographyOptions_t *homography_options, Image_t *warped_image1, Image_t *warped_image2);

#ifdef __cplusplus
}
#endif

#endif /* CUSIFT_H */
