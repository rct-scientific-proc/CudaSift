/**
 * @file cusift.h
 * @brief Public C API of cusift: GPU SIFT extraction, matching, RANSAC
 *        homography estimation and warping onto a common canvas.
 *
 * Naming: every exported symbol is prefixed `cusift_` and written in
 * snake_case; types have no suffix; struct fields are plain snake_case.
 * Functions report failure through the thread-local error state (see
 * cusift_last_error()), never through return codes or exceptions.
 *
 * @par References
 * M. Björkman, N. Bergström and D. Kragic, "Detecting, segmenting and tracking unknown objects using multi-label MRF inference", CVIU, 118, pp. 111-127, January 2014
 * https://github.com/Celebrandil/CudaSift
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
 * Exported functions
 *   Version and device:   cusift_version, cusift_init, cusift_device_reset
 *   Errors:               cusift_last_error, cusift_set_error_log,
 *                         cusift_set_error_callback, cusift_error_log_dropped
 *   Defaults:             cusift_default_extract_options, cusift_default_homography_options
 *   Stages:               cusift_extract, cusift_match, cusift_estimate_homography,
 *                         cusift_compute_canvas, cusift_warp, cusift_warp_to_device
 *   Pipelines:            cusift_extract_and_match, cusift_align,
 *                         cusift_align_and_warp, cusift_align_and_warp_to_device
 *   Resources:            cusift_keypoints_free, cusift_image_free,
 *                         cusift_device_image_free, cusift_keypoints_save_json
 *   Device memory:        cusift_memory_extract, cusift_memory_match,
 *                         cusift_memory_homography, cusift_memory_warp, cusift_memory_pipeline
 */

#ifndef CUSIFT_H
#define CUSIFT_H

#include <stddef.h>  /* size_t */
#include <stdbool.h> /* bool */

#define CUSIFT_VERSION_MAJOR 3
#define CUSIFT_VERSION_MINOR 0
#define CUSIFT_VERSION_PATCH 0

/* -- Export / import macros ------------------------------ */
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

#ifdef __cplusplus
extern "C"
{
#endif

    /* ===================================================================== */
    /* Version                                                               */
    /* ===================================================================== */

    /**
     * @brief Version of the library actually loaded (the header's
     *        CUSIFT_VERSION_* macros give the version compiled against).
     *        Any output pointer may be NULL.
     */
    CUSIFT_API void cusift_version(int *major, int *minor, int *patch);

    /* ===================================================================== */
    /* Errors                                                                */
    /* ===================================================================== */

    /**
     * @brief One error or warning: what cusift_last_error() reports and what an
     *        error callback receives.
     *
     * severity is 0 for an error (the call failed) and 1 for a warning (the
     * call completed in a degraded way: the AVX2 refinement path refusing to
     * run on a CPU without AVX2, or a device free failing during cleanup).
     * In a callback the pointers are valid for the duration of the callback.
     * From cusift_last_error() they are valid until the next library call on
     * the same thread. Copy what you keep.
     */
    typedef struct
    {
        int severity;                 /**< 0 = error, 1 = warning. */
        const char *function;         /**< Public function involved, e.g. "cusift_extract". */
        const char *file;             /**< Source file where the failure originated, or "". */
        int line;                     /**< Source line, or 0. */
        int cuda_error;               /**< cudaError_t value; 0 when the failure was not a CUDA error. */
        const char *cuda_error_name;  /**< e.g. "cudaErrorIllegalAddress"; "" when cuda_error is 0. */
        const char *message;          /**< Human-readable description. */
        unsigned long long thread_id; /**< Hash of the thread's id. */
        long long timestamp_ms;       /**< UTC Unix time in milliseconds. */
    } cusift_error_record;

    typedef void (*cusift_error_callback)(const cusift_error_record *record, void *user_data);

    /**
     * @brief Did the most recent library call on this thread fail?
     *
     * Every function below except the resource-release functions and the
     * error-sink setters clears the thread-local error state on entry, so
     * this always describes the last such call. Argument validation failures
     * (NULL pointers, out-of-range options, oversized images, device pointers
     * passed as host images) are reported this way, never by crashing.
     *
     * @param record  If not NULL, receives the error; untouched when there is none.
     * @return 1 if the last call failed, 0 otherwise.
     */
    CUSIFT_API int cusift_last_error(cusift_error_record *record);

    /**
     * @brief Append every error and warning to a log file, in addition to the
     *        thread-local error state.
     *
     * Off by default. Pass NULL to turn it off, "" to log to
     * <temp dir>/cusift-errors.log (the first set one of TMPDIR, TMP, TEMP,
     * TEMPDIR, else /tmp, on POSIX; the first set one of TMP, TEMP,
     * USERPROFILE, else the current directory, on Windows), or an explicit
     * path. max_bytes caps the file;
     * 0 means 1 MiB. When a new line would push the file past the cap, the
     * file is renamed to <path>.1 (replacing any previous .1) and started
     * afresh, so disk use is bounded at twice the cap.
     *
     * Each line holds a UTC timestamp, the severity, the thread id, the public
     * function, the source file and line, the CUDA error code and name, and
     * the message. The file is opened, appended and closed per record.
     *
     * Logging never fails or alters the call being logged; records that could
     * not be written are counted by cusift_error_log_dropped(). Without a call
     * to this function the environment variable CUSIFT_ERROR_LOG is consulted
     * once, on the first error: its value is the path, or "default" for the
     * temporary-directory file; CUSIFT_ERROR_LOG_MAX_BYTES sets the cap. This
     * function does not touch the thread-local error state.
     */
    CUSIFT_API void cusift_set_error_log(const char *path, size_t max_bytes);

    /**
     * @brief Deliver every error and warning to a callback as well. NULL disables.
     *
     * The callback runs on the thread where the error occurred, after the
     * error state has been set and the log line (if any) written, with no
     * library lock held, so it may reconfigure the sinks. It must not throw.
     * Process-wide: one callback for all threads.
     */
    CUSIFT_API void cusift_set_error_callback(cusift_error_callback callback, void *user_data);

    /** @brief Records that could not be written to the error log since the process started. */
    CUSIFT_API unsigned long long cusift_error_log_dropped(void);

    /* ===================================================================== */
    /* Device                                                                */
    /* ===================================================================== */

    /**
     * @brief Select the CUDA device. Call once before anything else.
     *        Uses device 0; fails if no CUDA device is present.
     */
    CUSIFT_API void cusift_init(void);

    /**
     * @brief Recover from a sticky CUDA error by destroying and re-creating this
     *        process's CUDA context on the current device.
     *
     * Some device errors are sticky: once a kernel has faulted (for example
     * cudaErrorIllegalAddress, cudaErrorLaunchFailure, cudaErrorLaunchTimeout,
     * cudaErrorECCUncorrectable or cudaErrorHardwareStackError), every later
     * CUDA call in the process returns that same error until the context is
     * reset; cusift_last_error() reports the code so a caller can recognise it.
     *
     * The reset is process-local: it calls cudaDeviceReset(), which tears down
     * only the calling process's context, so other processes sharing the GPU
     * are not affected. Inside this process, however, every device allocation
     * is destroyed, including the device buffers of every cusift_keypoints and
     * every cusift_device_image. Release all of them first, or at least never
     * touch them afterwards, and make sure no other thread of this process is
     * inside a library call. The function re-creates the context before
     * returning and sets the error state if the device is still unusable.
     */
    CUSIFT_API void cusift_device_reset(void);

    /* ===================================================================== */
    /* Types                                                                 */
    /* ===================================================================== */

    /** @brief A SIFT keypoint with its 128-d descriptor and its match fields. */
    typedef struct
    {
        float x;             /**< Column, in pixels of the original image. */
        float y;             /**< Row, in pixels of the original image. */
        float scale;         /**< Detected scale (sigma). */
        float sharpness;     /**< Difference-of-Gaussians response at the keypoint. */
        float edgeness;      /**< Edge response (trace^2 / det of the Hessian). */
        float orientation;   /**< Dominant orientation in degrees. */
        float score;         /**< Best match similarity (cosine), set by cusift_match. */
        float ambiguity;     /**< Second-best / best match similarity, set by cusift_match. */
        int match;           /**< Index of the matched keypoint in the other set, or -1. */
        float match_x;       /**< Column of the matched keypoint. */
        float match_y;       /**< Row of the matched keypoint. */
        float match_error;   /**< Reprojection error under the refined homography, in pixels. */
        float subsampling;   /**< Octave scale factor the keypoint was found at. */
        float reserved[3];   /**< Padding; keep the record 16-byte aligned. */
        float descriptor[128]; /**< Unit-length SIFT descriptor. */
    } cusift_keypoint;

    /**
     * @brief A set of keypoints on host and device.
     *
     * Ownership contract: zero-initialise the struct before its first use
     * (`cusift_keypoints kp = {0};` in C, `cusift_keypoints kp{};` in C++).
     * Every function that fills it first releases any buffers it already
     * holds, so one struct can be re-used across calls without leaking; call
     * cusift_keypoints_free() when done. If a call fails the struct is left
     * safe to free. Passing a struct that was never zero-initialised is
     * undefined behaviour: its garbage pointers would be freed.
     */
    typedef struct
    {
        int count;               /**< Number of valid keypoints. */
        int capacity;            /**< Number of allocated keypoints. */
        cusift_keypoint *host;   /**< Host copy, or NULL. */
        cusift_keypoint *device; /**< Device copy, or NULL. */
    } cusift_keypoints;

    /**
     * @brief A grayscale float32 image in HOST memory, row-major, values in [0, 255].
     *
     * pixels must point to memory the CPU can read: ordinary pageable memory,
     * pinned memory from cudaHostAlloc / cudaMallocHost, or managed memory from
     * cudaMallocManaged. A pointer returned by cudaMalloc is not CPU-readable;
     * every entry point classifies the pointer and rejects it with the error
     * "pixels points to device memory". Both dimensions must be in [8, 65535]
     * pixels. The library never modifies an input image.
     */
    typedef struct
    {
        float *pixels; /**< Row-major float32 pixel data. */
        int width;     /**< Width in pixels. */
        int height;    /**< Height in pixels. */
    } cusift_image;

    /**
     * @brief A grayscale float32 image in DEVICE memory with a row pitch, as
     *        produced by cusift_warp_to_device() and cusift_align_and_warp_to_device().
     */
    typedef struct
    {
        float *device_pixels; /**< Pitched device buffer. */
        int width;            /**< Width in pixels. */
        int height;           /**< Height in pixels. */
        size_t pitch_bytes;   /**< Bytes between the starts of consecutive rows. */
    } cusift_device_image;

    /**
     * @brief Where the two inputs land on the common canvas produced by the warp
     *        functions.
     *
     * The canvas is image 1's coordinate frame, extended so that it also holds
     * image 2 after transformation through the inverse homography. Canvas pixel
     * (0, 0) is image-1 coordinate (origin_u, origin_v); both are integers and
     * never positive, because image 1 is always kept whole. Every warp entry
     * point sizes its output with exactly this geometry, so the values here are
     * bit-identical to what the warps use internally.
     *
     * Corners are listed in the order (0, 0), (w-1, 0), (w-1, h-1), (0, h-1) of
     * the source image, in canvas pixel coordinates, using the library's
     * pixel-centre convention (pixel (i, j) sits at point (i, j)). Image 1's
     * corners are therefore integers; image 2's are the sub-pixel positions of
     * its transformed corners. The warp samples the half-open range
     * [0, w-1) x [0, h-1) of each source, so the last column and row described
     * by these corners hold NaN in the warped output.
     */
    typedef struct
    {
        int origin_u;           /**< Image-1 x coordinate of canvas pixel (0, 0); <= 0. */
        int origin_v;           /**< Image-1 y coordinate of canvas pixel (0, 0); <= 0. */
        int canvas_width;       /**< Width of the warped images. */
        int canvas_height;      /**< Height of the warped images. */
        float corners1[4][2];   /**< Image-1 corners on the canvas, (x, y) each. */
        float corners2[4][2];   /**< Image-2 corners on the canvas, (x, y) each. */
    } cusift_canvas;

    /**
     * @brief Options for SIFT extraction. Fill with cusift_default_extract_options()
     *        and change what you need.
     *
     * contrast_threshold [2.0]: a Difference-of-Gaussians extremum is kept when
     *   its absolute response exceeds this, in intensity units of the [0, 255]
     *   input. Higher keeps fewer, more stable keypoints.
     * lowest_scale [0.0], highest_scale [+inf]: keep only keypoints whose scale,
     *   in pixels of the original image, lies in this range.
     * edge_threshold [10.0]: candidates whose Hessian trace^2 / determinant
     *   exceeds this are edge responses and are rejected. Lower is stricter.
     * initial_blur [1.0]: sigma of the Gaussian applied to the input before the
     *   scale-space pyramid is built. The pyramid's scales are relative to this
     *   base, so a keypoint reported at scale s has total blur
     *   sqrt(initial_blur^2 + s^2). 0 disables the pre-blur.
     * max_keypoints [32768]: capacity of the keypoint buffers; detections beyond
     *   it are dropped. Must be in [1, INT_MAX / sizeof(cusift_keypoint)].
     * num_octaves [5]: pyramid depth; clamped to [3, 7] and to what the image
     *   size supports (about log2(min(width, height)) - 3).
     * scale_suppression_radius [0.0]: when > 0, every keypoint removes any
     *   smaller-scale keypoint within radius * its_scale pixels, keeping one
     *   detection per feature. 6.0 is a good starting value; 0 disables.
     */
    typedef struct
    {
        float contrast_threshold;
        float lowest_scale;
        float highest_scale;
        float edge_threshold;
        float initial_blur;
        int max_keypoints;
        int num_octaves;
        float scale_suppression_radius;
    } cusift_extract_options;

    /** @brief Geometric model fitted by cusift_estimate_homography(). */
    typedef enum
    {
        CUSIFT_MODEL_HOMOGRAPHY = 0, /**< 8-DOF projective homography from 4-point samples. */
        CUSIFT_MODEL_SIMILARITY = 1  /**< 4-DOF rotation + uniform scale + translation from 2-point samples; returned in the same 3x3 layout. */
    } cusift_model;

    /** @brief How to pick among several estimation attempts (see cusift_homography_options::attempts). */
    typedef enum
    {
        CUSIFT_GOAL_MAX_INLIERS = 0,           /**< The attempt with the most RANSAC inliers. */
        CUSIFT_GOAL_MIN_IDENTITY_DISTANCE = 1  /**< The attempt whose upper-left 2x2 block is closest to the identity (least rotation, scale and shear). */
    } cusift_goal;

    /**
     * @brief Options for homography estimation. Fill with
     *        cusift_default_homography_options() and change what you need.
     *
     * ransac_iterations [10000]: 4-point (or 2-point) hypotheses to test; rounded
     *   up to a multiple of 16, at most 1 048 560.
     * min_score [0.0], max_ambiguity [0.8]: a match takes part in RANSAC when its
     *   score is above min_score and its ambiguity below max_ambiguity.
     * inlier_threshold [3.0]: reprojection distance, in pixels, under which a
     *   match counts as an inlier of a hypothesis. The inlier count reported by
     *   the estimator is measured with this threshold.
     * refine_iterations [5]: rounds of least-squares refinement over the inliers
     *   after RANSAC; 0 keeps the raw 4-point hypothesis.
     * refine_min_score [0.0], refine_max_ambiguity [0.8], refine_inlier_threshold [2.0]:
     *   the same three gates for the refinement. The refinement also writes
     *   match_error into every keypoint.
     * seed [0]: seed of the random sampler; 0 draws from std::random_device,
     *   anything else makes the estimate bit-for-bit repeatable.
     * model [CUSIFT_MODEL_HOMOGRAPHY]: a cusift_model value.
     * attempts [1]: run the whole estimate this many times with seed + k for
     *   attempt k (fresh random seeds when seed is 0) and keep one by `goal`.
     *   1 is a single estimate.
     * goal [CUSIFT_GOAL_MAX_INLIERS]: a cusift_goal value, used when attempts > 1.
     */
    typedef struct
    {
        int ransac_iterations;
        float min_score;
        float max_ambiguity;
        float inlier_threshold;
        int refine_iterations;
        float refine_min_score;
        float refine_max_ambiguity;
        float refine_inlier_threshold;
        unsigned int seed;
        int model;
        int attempts;
        int goal;
    } cusift_homography_options;

    /** @brief Fill *options with the documented defaults. */
    CUSIFT_API void cusift_default_extract_options(cusift_extract_options *options);

    /** @brief Fill *options with the documented defaults. */
    CUSIFT_API void cusift_default_homography_options(cusift_homography_options *options);

    /* ===================================================================== */
    /* Stages                                                                */
    /* ===================================================================== */

    /**
     * @brief Detect SIFT keypoints and compute their descriptors on the GPU.
     *
     * @param image     Input image (host memory).
     * @param keypoints Receives the keypoints; follows the ownership contract above.
     * @param options   Extraction options.
     */
    CUSIFT_API void cusift_extract(const cusift_image *image, cusift_keypoints *keypoints, const cusift_extract_options *options);

    /**
     * @brief Match every keypoint of `query` against all keypoints of `train`.
     *
     * Brute-force cosine similarity on the GPU. For each query keypoint the
     * best and second-best similarities become `score` and `ambiguity`
     * (second / best), and `match`, `match_x`, `match_y` identify the best
     * candidate. Every query keypoint receives its nearest neighbour; the
     * ratio test is applied later through the max_ambiguity options. `train`
     * is only read. A set may be matched against itself.
     */
    CUSIFT_API void cusift_match(cusift_keypoints *query, const cusift_keypoints *train);

    /**
     * @brief Estimate the transform that maps the keypoints of `matched` onto
     *        their matches: RANSAC followed by least-squares refinement.
     *
     * `matched` must have been the `query` of cusift_match(). The result is a
     * row-major 3x3 homography H with H[8] = 1 such that, for a keypoint at
     * (x, y), H · (x, y, 1) is proportional to (match_x, match_y, 1). With
     * CUSIFT_MODEL_SIMILARITY the matrix has the form [a -b tx; b a ty; 0 0 1].
     * `inliers` receives the RANSAC inlier count of the winning hypothesis,
     * measured with inlier_threshold. Fails with "Homography contains non-finite
     * values" when fewer than 8 (homography) or 4 (similarity) candidates exist.
     */
    CUSIFT_API void cusift_estimate_homography(cusift_keypoints *matched, float homography[9], int *inliers, const cusift_homography_options *options);

    /**
     * @brief Compute, without warping anything, the canvas that cusift_warp()
     *        and cusift_warp_to_device() would produce for a homography: its
     *        size, its origin, and where the corners of both inputs land.
     *
     * Host-only and constant-time (one 3x3 inverse and four point projections);
     * it runs the same code the warps run internally, so the result matches a
     * subsequent warp exactly. Typical uses: a pre-flight check of the output
     * size, the exact device memory for a warp instead of the 2x bound of
     * cusift_memory_warp(), and locating the overlap, seams or crop rectangles.
     * See cusift_canvas for the meaning of each field.
     *
     * Errors mirror the warps: a non-finite or singular homography, non-positive
     * dimensions, or a canvas larger than twice the larger input in either axis
     * sets the error state. For the canvas-size errors *canvas is still
     * filled so the rejected canvas can be inspected; otherwise it is zeroed.
     */
    CUSIFT_API void cusift_compute_canvas(const float homography[9], int width1, int height1, int width2, int height2, cusift_canvas *canvas);

    /**
     * @brief Warp both images onto the common canvas; outputs in host memory.
     *
     * Image 1 is copied into the canvas unchanged (integer shift); image 2 is
     * resampled through the homography with bilinear interpolation. Canvas
     * pixels no source covers are NaN. The outputs are allocated with malloc;
     * release them with cusift_image_free(). Fails with "warped image too
     * large" when the canvas would exceed twice the larger input in either axis.
     *
     * @param use_gpu  true: warp on the GPU (both images are uploaded); false: OpenMP on the host.
     */
    CUSIFT_API void cusift_warp(const cusift_image *image1, const cusift_image *image2, const float homography[9], cusift_image *warped1, cusift_image *warped2, bool use_gpu);

    /**
     * @brief As cusift_warp() on the GPU, but the two canvases stay in device
     *        memory as pitched buffers. Release them with cusift_device_image_free().
     */
    CUSIFT_API void cusift_warp_to_device(const cusift_image *image1, const cusift_image *image2, const float homography[9], cusift_device_image *warped1, cusift_device_image *warped2);

    /* ===================================================================== */
    /* Pipelines                                                             */
    /* ===================================================================== */

    /**
     * @brief cusift_extract() on both images (same options) followed by
     *        cusift_match(keypoints1, keypoints2).
     */
    CUSIFT_API void cusift_extract_and_match(const cusift_image *image1, const cusift_image *image2, cusift_keypoints *keypoints1, cusift_keypoints *keypoints2, const cusift_extract_options *options);

    /**
     * @brief Extract, match and estimate the homography from image 1 to image 2
     *        in one call. The device-resident images are reused across stages.
     *        Several attempts and the selection goal are controlled by the
     *        homography options.
     */
    CUSIFT_API void cusift_align(const cusift_image *image1, const cusift_image *image2, cusift_keypoints *keypoints1, cusift_keypoints *keypoints2, const cusift_extract_options *extract_options, const cusift_homography_options *homography_options, float homography[9], int *inliers);

    /**
     * @brief cusift_align() followed by the GPU warp, with the canvases
     *        returned in host memory. The images already on the device for
     *        extraction are reused as warp sources, so nothing is uploaded twice.
     */
    CUSIFT_API void cusift_align_and_warp(const cusift_image *image1, const cusift_image *image2, cusift_keypoints *keypoints1, cusift_keypoints *keypoints2, const cusift_extract_options *extract_options, const cusift_homography_options *homography_options, float homography[9], int *inliers, cusift_image *warped1, cusift_image *warped2);

    /** @brief As cusift_align_and_warp(), with the canvases left in device memory. */
    CUSIFT_API void cusift_align_and_warp_to_device(const cusift_image *image1, const cusift_image *image2, cusift_keypoints *keypoints1, cusift_keypoints *keypoints2, const cusift_extract_options *extract_options, const cusift_homography_options *homography_options, float homography[9], int *inliers, cusift_device_image *warped1, cusift_device_image *warped2);

    /* ===================================================================== */
    /* Resources                                                             */
    /* ===================================================================== */

    /** @brief Release both buffers of a keypoint set and zero it. Safe on an empty set. Leaves the error state alone. */
    CUSIFT_API void cusift_keypoints_free(cusift_keypoints *keypoints);

    /** @brief Release a host image allocated by the library (a warp output) and zero it. */
    CUSIFT_API void cusift_image_free(cusift_image *image);

    /** @brief Release a device image allocated by the library and zero it. */
    CUSIFT_API void cusift_device_image_free(cusift_device_image *image);

    /** @brief Write the host copy of a keypoint set, descriptors included, to a JSON file. */
    CUSIFT_API void cusift_keypoints_save_json(const char *path, const cusift_keypoints *keypoints);

    /* ===================================================================== */
    /* Device memory estimates                                               */
    /* ===================================================================== */

    /**
     * @brief Peak device memory, in bytes, that cusift_extract() needs for an
     *        image of this size with these options (keypoint buffer, image copy,
     *        scale-space pyramid, sort temporaries, context).
     */
    CUSIFT_API size_t cusift_memory_extract(int width, int height, const cusift_extract_options *options);

    /** @brief Device memory held by two keypoint sets of these capacities during matching (matching allocates nothing else). */
    CUSIFT_API size_t cusift_memory_match(int capacity1, int capacity2);

    /** @brief Device memory that cusift_estimate_homography() needs: the keypoint set plus the RANSAC temporaries. */
    CUSIFT_API size_t cusift_memory_homography(int capacity, const cusift_homography_options *options);

    /**
     * @brief Peak device memory of a GPU warp, assuming the worst-case canvas of
     *        twice the larger input in each axis. cusift_compute_canvas() gives
     *        the exact canvas once the homography is known.
     */
    CUSIFT_API size_t cusift_memory_warp(int width1, int height1, int width2, int height2);

    /** @brief Peak device memory across cusift_align_and_warp(): the maximum over its stages. */
    CUSIFT_API size_t cusift_memory_pipeline(int width1, int height1, int width2, int height2, const cusift_extract_options *extract_options, const cusift_homography_options *homography_options);

#ifdef __cplusplus
}
#endif

#endif /* CUSIFT_H */
