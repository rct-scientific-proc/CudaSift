/**
 * @file cusift_matlab.h
 * @brief Flat C surface of cusift for MATLAB's loadlibrary / calllib.
 *
 * The public cusift.h is written for C and ctypes and uses constructs that
 * MATLAB's loadlibrary cannot parse: a struct holding pointers to another
 * struct (cusift_keypoints), fixed-size array fields (descriptor[128],
 * corners[4][2]), a callback function pointer, and bool.  This header is a
 * second surface over the same library, built as its own shared library
 * (cusift_matlab.dll / libcusift_matlab.so) with everything linked in, that
 * keeps to what loadlibrary handles:
 *
 *   - keypoint sets and warped images stay inside the library behind opaque
 *     uint64 handles; the library owns them until cusiftml_keypoints_free(),
 *     cusiftml_image_free() or cusiftml_free_all();
 *   - every array crosses as a plain pointer to float or int, with its size
 *     given by other arguments or fixed by this header;
 *   - every function returns an int status, 0 on success and 1 on failure,
 *     with the message available from cusiftml_last_error().  The only
 *     exceptions are the cusiftml_memory_* estimates and
 *     cusiftml_error_log_dropped(), which return their value directly;
 *   - only int, unsigned int, float, char and unsigned long long appear; no
 *     bool, size_t, enum or function pointer, and no other header.
 *
 * A handle that was never created, was already freed, or is of the other
 * kind is reported as an error, never dereferenced.
 *
 * @par Layout conventions
 * Images are float32, values in [0, 255], width x height pixels, row-major.
 * MATLAB stores matrices column by column, so pass single(img).' with
 * width = size(img, 2) and height = size(img, 1); a warped image copied out
 * with cusiftml_image_copy() into a width x height single matrix wants the
 * same .' afterwards.
 *
 * Homographies are 9 floats in row-major order: H(1,1) H(1,2) H(1,3) H(2,1)
 * ... H(3,3).  In MATLAB, reshape(h, 3, 3).' is the matrix and
 * reshape(H.', 1, 9) is the array.
 *
 * A keypoint set copies out as 144 floats per keypoint, keypoint after
 * keypoint, which is a 144 x N single matrix in MATLAB with one keypoint per
 * column.  Rows:
 *    1 x               (column, pixels of the original image, 0-based)
 *    2 y               (row)
 *    3 scale
 *    4 sharpness
 *    5 edgeness
 *    6 orientation     (degrees)
 *    7 score           (best match similarity, after matching)
 *    8 ambiguity       (second-best / best similarity, after matching)
 *    9 match           (int32 bits: typecast(row, 'int32'); 0-based index of the
 *                       matched keypoint in the other set, or -1)
 *   10 match_x
 *   11 match_y
 *   12 match_error     (reprojection error after homography refinement)
 *   13 subsampling
 *   14-16 reserved
 *   17-144 descriptor  (unit-length SIFT descriptor)
 *
 * @par Threads
 * The error message is per thread.  Handles are shared by all threads and
 * guarded by one lock, which is held while the library works on a handle.
 *
 * MIT License, see cusift.h.
 */

#ifndef CUSIFT_MATLAB_H
#define CUSIFT_MATLAB_H

/* Export macro: dllexport while building the library, nothing otherwise so
 * that loadlibrary sees plain prototypes. */
#if defined(CUSIFT_MATLAB_EXPORTS)
#if defined(_WIN32)
#define CUSIFT_MATLAB_API __declspec(dllexport)
#elif defined(__GNUC__) && __GNUC__ >= 4
#define CUSIFT_MATLAB_API __attribute__((visibility("default")))
#else
#define CUSIFT_MATLAB_API
#endif
#else
#define CUSIFT_MATLAB_API
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    /* ===================================================================== */
    /* Options (field for field the structs of cusift.h, which documents them) */
    /* ===================================================================== */

    typedef struct
    {
        float contrast_threshold;       /* [2.0] */
        float lowest_scale;             /* [0.0] */
        float highest_scale;            /* [+inf] */
        float edge_threshold;           /* [10.0] */
        float initial_blur;             /* [1.0] */
        int max_keypoints;              /* [32768] */
        int num_octaves;                /* [5] */
        float scale_suppression_radius; /* [0.0] */
    } cusiftml_extract_options;

    typedef struct
    {
        int ransac_iterations;         /* [10000] */
        float min_score;               /* [0.0] */
        float max_ambiguity;           /* [0.8] */
        float inlier_threshold;        /* [3.0] */
        int refine_iterations;         /* [5] */
        float refine_min_score;        /* [0.0] */
        float refine_max_ambiguity;    /* [0.8] */
        float refine_inlier_threshold; /* [2.0] */
        unsigned int seed;             /* [0]: 0 draws a random seed */
        int model;                     /* [0]: 0 homography, 1 similarity */
        int attempts;                  /* [1] */
        int goal;                      /* [0]: 0 most inliers, 1 closest to identity */
    } cusiftml_homography_options;

    /** Fill the documented defaults. */
    CUSIFT_MATLAB_API int cusiftml_default_extract_options(cusiftml_extract_options *options);
    CUSIFT_MATLAB_API int cusiftml_default_homography_options(cusiftml_homography_options *options);

    /* ===================================================================== */
    /* Library, device and errors                                            */
    /* ===================================================================== */

    /** Version of the loaded library.  Any output may be NULL. */
    CUSIFT_MATLAB_API int cusiftml_version(int *major, int *minor, int *patch);

    /** Select the CUDA device.  Call once before anything else. */
    CUSIFT_MATLAB_API int cusiftml_init(void);

    /**
     * Recover from a sticky CUDA error (see cusift_device_reset in cusift.h).
     * Frees every handle first, because the reset destroys their device
     * memory: all handles are invalid afterwards.
     */
    CUSIFT_MATLAB_API int cusiftml_device_reset(void);

    /**
     * Message of the most recent failed call on this thread.
     *
     * Writes the message, NUL-terminated and truncated to buffer_size - 1
     * characters, and returns 1 if the most recent cusiftml_* call on this
     * thread failed, 0 if it succeeded (the buffer then receives "").  Does
     * not change the state, so it may be called more than once.  A NULL
     * buffer or buffer_size <= 0 only returns the status.
     */
    CUSIFT_MATLAB_API int cusiftml_last_error(char *buffer, int buffer_size);

    /**
     * Append every library error and warning to a file (cusift_set_error_log).
     * "" logs to <temp dir>/cusift-errors.log; max_bytes 0 means 1 MiB.
     * NULL turns the file off, as does cusiftml_clear_error_log().
     */
    CUSIFT_MATLAB_API int cusiftml_set_error_log(const char *path, unsigned long long max_bytes);
    CUSIFT_MATLAB_API int cusiftml_clear_error_log(void);

    /** Records that could not be written to the error log since the process started. */
    CUSIFT_MATLAB_API unsigned long long cusiftml_error_log_dropped(void);

    /* ===================================================================== */
    /* Keypoint sets                                                         */
    /* ===================================================================== */

    /**
     * Detect SIFT keypoints and descriptors on the GPU.
     * pixels: width * height floats, row-major.  options may be NULL for the
     * defaults.  On success *keypoints receives a new handle; on failure 0.
     */
    CUSIFT_MATLAB_API int cusiftml_extract(const float *pixels, int width, int height,
                                           const cusiftml_extract_options *options,
                                           unsigned long long *keypoints);

    /** Number of keypoints in the set and the capacity it was extracted with. */
    CUSIFT_MATLAB_API int cusiftml_keypoints_info(unsigned long long keypoints, int *count, int *capacity);

    /**
     * Copy the set into data, 144 floats per keypoint in the layout above.
     * data must hold room for `count` keypoints (144 * count floats), and
     * count must be at least the set's count; the set's count keypoints are
     * written and anything beyond is left untouched.
     */
    CUSIFT_MATLAB_API int cusiftml_keypoints_copy(unsigned long long keypoints, float *data, int count);

    /** Write the set, descriptors included, to a JSON file. */
    CUSIFT_MATLAB_API int cusiftml_keypoints_save_json(unsigned long long keypoints, const char *path);

    /** Release a set.  A handle that is not live is ignored. */
    CUSIFT_MATLAB_API int cusiftml_keypoints_free(unsigned long long keypoints);

    /**
     * Match every keypoint of query against all of train (cusift_match):
     * fills the score, ambiguity, match, match_x and match_y rows of query.
     * The two handles may be the same set.
     */
    CUSIFT_MATLAB_API int cusiftml_match(unsigned long long query, unsigned long long train);

    /**
     * RANSAC plus refinement on a matched query set (cusift_estimate_homography).
     * homography receives 9 floats, row-major; inliers the RANSAC inlier count.
     * options may be NULL for the defaults.  Also fills the match_error row.
     */
    CUSIFT_MATLAB_API int cusiftml_estimate_homography(unsigned long long matched,
                                                       const cusiftml_homography_options *options,
                                                       float *homography, int *inliers);

    /* ===================================================================== */
    /* Canvas and warps                                                      */
    /* ===================================================================== */

    /**
     * The canvas a warp would produce (cusift_compute_canvas), without warping.
     * canvas receives 4 ints: origin_u, origin_v, canvas_width, canvas_height.
     * corners receives 16 floats: the four (x, y) corners of image 1 on the
     * canvas, then the four of image 2, in the order (0,0), (w-1,0),
     * (w-1,h-1), (0,h-1).  On a canvas-size error both are still filled.
     */
    CUSIFT_MATLAB_API int cusiftml_compute_canvas(const float *homography,
                                                  int width1, int height1, int width2, int height2,
                                                  int *canvas, float *corners);

    /**
     * Warp both images onto the common canvas (cusift_warp).  use_gpu: 1 for
     * the GPU, 0 for OpenMP on the host.  The two canvases stay inside the
     * library; *warped1 and *warped2 receive their handles (0 on failure).
     * Read them with cusiftml_image_info() and cusiftml_image_copy(), then
     * release them with cusiftml_image_free().
     */
    CUSIFT_MATLAB_API int cusiftml_warp(const float *pixels1, int width1, int height1,
                                        const float *pixels2, int width2, int height2,
                                        const float *homography, int use_gpu,
                                        unsigned long long *warped1, unsigned long long *warped2);

    /** Size of a warped image held by the library. */
    CUSIFT_MATLAB_API int cusiftml_image_info(unsigned long long image, int *width, int *height);

    /**
     * Copy a warped image into pixels, row-major.  width and height must be
     * the image's own size (from cusiftml_image_info), so the buffer size is
     * checked.  Pixels no source covers are NaN.
     */
    CUSIFT_MATLAB_API int cusiftml_image_copy(unsigned long long image, float *pixels, int width, int height);

    /** Release a warped image.  A handle that is not live is ignored. */
    CUSIFT_MATLAB_API int cusiftml_image_free(unsigned long long image);

    /* ===================================================================== */
    /* Pipelines                                                             */
    /* ===================================================================== */

    /** cusift_extract_and_match: two new keypoint handles, the first already matched against the second. */
    CUSIFT_MATLAB_API int cusiftml_extract_and_match(const float *pixels1, int width1, int height1,
                                                     const float *pixels2, int width2, int height2,
                                                     const cusiftml_extract_options *options,
                                                     unsigned long long *keypoints1, unsigned long long *keypoints2);

    /** cusift_align: extract, match and estimate in one call.  Either options pointer may be NULL. */
    CUSIFT_MATLAB_API int cusiftml_align(const float *pixels1, int width1, int height1,
                                         const float *pixels2, int width2, int height2,
                                         const cusiftml_extract_options *extract_options,
                                         const cusiftml_homography_options *homography_options,
                                         unsigned long long *keypoints1, unsigned long long *keypoints2,
                                         float *homography, int *inliers);

    /** cusift_align_and_warp: cusiftml_align plus the GPU warp, with the canvases as image handles. */
    CUSIFT_MATLAB_API int cusiftml_align_and_warp(const float *pixels1, int width1, int height1,
                                                  const float *pixels2, int width2, int height2,
                                                  const cusiftml_extract_options *extract_options,
                                                  const cusiftml_homography_options *homography_options,
                                                  unsigned long long *keypoints1, unsigned long long *keypoints2,
                                                  float *homography, int *inliers,
                                                  unsigned long long *warped1, unsigned long long *warped2);

    /**
     * Release every keypoint set and warped image.  Call before
     * unloadlibrary: unloading does not free device memory by itself.
     */
    CUSIFT_MATLAB_API int cusiftml_free_all(void);

    /* ===================================================================== */
    /* Device memory estimates (bytes; host-only, allocate nothing)          */
    /* ===================================================================== */

    /**
     * These mirror cusift_memory_* and return the estimate directly.  An
     * argument outside the library's limits (image sides in [8, 65535],
     * capacities in [1, INT_MAX / 576]) returns 0 with the reason in
     * cusiftml_last_error(); a NULL options pointer means the defaults.
     */
    CUSIFT_MATLAB_API unsigned long long cusiftml_memory_extract(int width, int height, const cusiftml_extract_options *options);
    CUSIFT_MATLAB_API unsigned long long cusiftml_memory_match(int capacity1, int capacity2);
    CUSIFT_MATLAB_API unsigned long long cusiftml_memory_homography(int capacity, const cusiftml_homography_options *options);
    CUSIFT_MATLAB_API unsigned long long cusiftml_memory_warp(int width1, int height1, int width2, int height2);
    CUSIFT_MATLAB_API unsigned long long cusiftml_memory_pipeline(int width1, int height1, int width2, int height2,
                                                                  const cusiftml_extract_options *extract_options,
                                                                  const cusiftml_homography_options *homography_options);

#ifdef __cplusplus
}
#endif

#endif /* CUSIFT_MATLAB_H */
