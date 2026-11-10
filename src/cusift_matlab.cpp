/**
 * @file cusift_matlab.cpp
 * @brief Implementation of cusift_matlab.h, the flat handle-based surface of
 *        cusift for MATLAB's loadlibrary.
 *
 * Everything here is bookkeeping; the work is done by the public cusift_*
 * functions, which are linked into the same shared library.  The façade
 * keeps the keypoint sets and warped images the library allocates in tables
 * keyed by a uint64 handle, converts the loadlibrary-friendly option structs
 * into the library's, and turns the library's thread-local error state into
 * the message cusiftml_last_error() hands back.
 */

#include "cusift_matlab.h"
#include "cusift.h"

#include <climits>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

// The copy-out layout documented in cusift_matlab.h: 144 floats per keypoint,
// no padding, match in row 9, descriptor from row 17.
static_assert(sizeof(cusift_keypoint) == 144 * sizeof(float), "cusift_keypoint is documented as 144 floats");
static_assert(offsetof(cusift_keypoint, match) == 8 * sizeof(float), "match is row 9 of the copy-out matrix");
static_assert(offsetof(cusift_keypoint, descriptor) == 16 * sizeof(float), "descriptor starts at row 17");
// The option structs are field-for-field mirrors; the conversion below copies
// each field, the size check only documents the intent.
static_assert(sizeof(cusiftml_extract_options) == sizeof(cusift_extract_options), "extract options must mirror cusift.h");
static_assert(sizeof(cusiftml_homography_options) == sizeof(cusift_homography_options), "homography options must mirror cusift.h");

namespace
{

// -- Error state (per thread, like the library's) --------------------------

thread_local std::string t_error; // empty: the most recent façade call on this thread succeeded

int fail(const std::string &message)
{
    t_error = message;
    return 1;
}

int ok()
{
    t_error.clear();
    return 0;
}

// Reads the library's error state for the cusift_* call that just returned.
int check(const char *function)
{
    cusift_error_record rec{};
    if (!cusift_last_error(&rec))
        return ok();
    std::string m = function;
    m += ": ";
    m += (rec.message && *rec.message) ? rec.message : "unknown error";
    if (rec.cuda_error != 0 && rec.cuda_error_name && *rec.cuda_error_name)
    {
        m += " [";
        m += rec.cuda_error_name;
        m += "]";
    }
    if (rec.file && *rec.file)
    {
        m += " (";
        m += rec.file;
        m += ":";
        m += std::to_string(rec.line);
        m += ")";
    }
    return fail(m);
}

// -- Handle tables ---------------------------------------------------------

struct Tables
{
    std::mutex mutex;
    unsigned long long next = 1; // 0 is never a valid handle
    std::unordered_map<unsigned long long, cusift_keypoints> keypoints;
    std::unordered_map<unsigned long long, cusift_image> images;
};

// Function-local static: valid however early MATLAB calls in.  Its
// destructor frees nothing on purpose: it runs while the library is being
// unloaded, when CUDA may already be gone.  cusiftml_free_all() is the
// caller's job before unloadlibrary.
Tables &tables()
{
    static Tables t;
    return t;
}

unsigned long long store(Tables &t, const cusift_keypoints &kp)
{
    unsigned long long id = t.next++;
    t.keypoints.emplace(id, kp);
    return id;
}

unsigned long long store(Tables &t, const cusift_image &im)
{
    unsigned long long id = t.next++;
    t.images.emplace(id, im);
    return id;
}

std::string no_such(const char *kind, unsigned long long handle)
{
    return "handle " + std::to_string(handle) + " is not a live " + kind +
           " (never created, already freed, or a handle of the other kind)";
}

// Locked for the whole of every function that touches a table: MATLAB calls
// from one thread, and holding the lock across the library call is what
// keeps a concurrent free from pulling a set out from under it.
using Lock = std::lock_guard<std::mutex>;

// -- Conversions -----------------------------------------------------------

cusift_extract_options to_c(const cusiftml_extract_options *o)
{
    cusift_extract_options c;
    if (!o)
    {
        cusift_default_extract_options(&c);
        return c;
    }
    c.contrast_threshold       = o->contrast_threshold;
    c.lowest_scale             = o->lowest_scale;
    c.highest_scale            = o->highest_scale;
    c.edge_threshold           = o->edge_threshold;
    c.initial_blur             = o->initial_blur;
    c.max_keypoints            = o->max_keypoints;
    c.num_octaves              = o->num_octaves;
    c.scale_suppression_radius = o->scale_suppression_radius;
    return c;
}

cusift_homography_options to_c(const cusiftml_homography_options *o)
{
    cusift_homography_options c;
    if (!o)
    {
        cusift_default_homography_options(&c);
        return c;
    }
    c.ransac_iterations       = o->ransac_iterations;
    c.min_score               = o->min_score;
    c.max_ambiguity           = o->max_ambiguity;
    c.inlier_threshold        = o->inlier_threshold;
    c.refine_iterations       = o->refine_iterations;
    c.refine_min_score        = o->refine_min_score;
    c.refine_max_ambiguity    = o->refine_max_ambiguity;
    c.refine_inlier_threshold = o->refine_inlier_threshold;
    c.seed                    = o->seed;
    c.model                   = o->model;
    c.attempts                = o->attempts;
    c.goal                    = o->goal;
    return c;
}

cusift_image view(const float *pixels, int width, int height)
{
    cusift_image im;
    im.pixels = const_cast<float *>(pixels); // the library never writes an input image
    im.width  = width;
    im.height = height;
    return im;
}

void release(cusift_image &im)
{
    if (im.pixels)
        cusift_image_free(&im);
}

// The limits the library enforces, applied here for the cusiftml_memory_*
// estimates, which have no status channel and whose cusift_* originals do
// not validate their arguments.
const int kMinDim      = 8;
const int kMaxDim      = 65535;
const int kMaxCapacity = INT_MAX / (int)sizeof(cusift_keypoint);

bool dims_ok(const char *label, int width, int height)
{
    if (width < kMinDim || width > kMaxDim || height < kMinDim || height > kMaxDim)
    {
        fail(std::string(label) + ": width and height must be in [" + std::to_string(kMinDim) + ", " +
             std::to_string(kMaxDim) + "], got " + std::to_string(width) + "x" + std::to_string(height));
        return false;
    }
    return true;
}

bool capacity_ok(const char *label, int capacity)
{
    if (capacity < 1 || capacity > kMaxCapacity)
    {
        fail(std::string(label) + " must be in [1, " + std::to_string(kMaxCapacity) + "], got " +
             std::to_string(capacity));
        return false;
    }
    return true;
}

} // namespace

extern "C"
{

    /* -- Options ------------------------------------------------------------ */

    int cusiftml_default_extract_options(cusiftml_extract_options *options)
    {
        if (!options)
            return fail("cusiftml_default_extract_options: options is NULL");
        cusift_extract_options c;
        cusift_default_extract_options(&c);
        options->contrast_threshold       = c.contrast_threshold;
        options->lowest_scale             = c.lowest_scale;
        options->highest_scale            = c.highest_scale;
        options->edge_threshold           = c.edge_threshold;
        options->initial_blur             = c.initial_blur;
        options->max_keypoints            = c.max_keypoints;
        options->num_octaves              = c.num_octaves;
        options->scale_suppression_radius = c.scale_suppression_radius;
        return ok();
    }

    int cusiftml_default_homography_options(cusiftml_homography_options *options)
    {
        if (!options)
            return fail("cusiftml_default_homography_options: options is NULL");
        cusift_homography_options c;
        cusift_default_homography_options(&c);
        options->ransac_iterations       = c.ransac_iterations;
        options->min_score               = c.min_score;
        options->max_ambiguity           = c.max_ambiguity;
        options->inlier_threshold        = c.inlier_threshold;
        options->refine_iterations       = c.refine_iterations;
        options->refine_min_score        = c.refine_min_score;
        options->refine_max_ambiguity    = c.refine_max_ambiguity;
        options->refine_inlier_threshold = c.refine_inlier_threshold;
        options->seed                    = c.seed;
        options->model                   = c.model;
        options->attempts                = c.attempts;
        options->goal                    = c.goal;
        return ok();
    }

    /* -- Library, device and errors ----------------------------------------- */

    int cusiftml_version(int *major, int *minor, int *patch)
    {
        cusift_version(major, minor, patch);
        return ok();
    }

    int cusiftml_init(void)
    {
        cusift_init();
        return check("cusift_init");
    }

    int cusiftml_device_reset(void)
    {
        cusiftml_free_all();
        cusift_device_reset();
        return check("cusift_device_reset");
    }

    int cusiftml_last_error(char *buffer, int buffer_size)
    {
        const int failed = t_error.empty() ? 0 : 1;
        if (!buffer || buffer_size <= 0)
            return failed;
        size_t n = t_error.size();
        if (n > (size_t)(buffer_size - 1))
            n = (size_t)(buffer_size - 1);
        std::memcpy(buffer, t_error.data(), n);
        buffer[n] = '\0';
        return failed;
    }

    int cusiftml_set_error_log(const char *path, unsigned long long max_bytes)
    {
        cusift_set_error_log(path, (size_t)max_bytes);
        return ok();
    }

    int cusiftml_clear_error_log(void)
    {
        cusift_set_error_log(nullptr, 0);
        return ok();
    }

    unsigned long long cusiftml_error_log_dropped(void)
    {
        return cusift_error_log_dropped();
    }

    /* -- Keypoint sets ------------------------------------------------------ */

    int cusiftml_extract(const float *pixels, int width, int height,
                         const cusiftml_extract_options *options, unsigned long long *keypoints)
    {
        if (!keypoints)
            return fail("cusiftml_extract: keypoints (output handle) is NULL");
        *keypoints = 0;
        cusift_image image           = view(pixels, width, height);
        cusift_extract_options eo    = to_c(options);
        cusift_keypoints kp{};
        cusift_extract(&image, &kp, &eo);
        if (check("cusift_extract"))
        {
            cusift_keypoints_free(&kp);
            return 1;
        }
        Tables &t = tables();
        Lock lock(t.mutex);
        *keypoints = store(t, kp);
        return 0;
    }

    int cusiftml_keypoints_info(unsigned long long keypoints, int *count, int *capacity)
    {
        Tables &t = tables();
        Lock lock(t.mutex);
        auto it = t.keypoints.find(keypoints);
        if (it == t.keypoints.end())
            return fail("cusiftml_keypoints_info: " + no_such("keypoint set", keypoints));
        if (count)
            *count = it->second.count;
        if (capacity)
            *capacity = it->second.capacity;
        return ok();
    }

    int cusiftml_keypoints_copy(unsigned long long keypoints, float *data, int count)
    {
        if (!data)
            return fail("cusiftml_keypoints_copy: data is NULL");
        Tables &t = tables();
        Lock lock(t.mutex);
        auto it = t.keypoints.find(keypoints);
        if (it == t.keypoints.end())
            return fail("cusiftml_keypoints_copy: " + no_such("keypoint set", keypoints));
        const cusift_keypoints &kp = it->second;
        if (count < kp.count)
            return fail("cusiftml_keypoints_copy: data holds " + std::to_string(count) + " keypoints, the set has " +
                        std::to_string(kp.count));
        if (kp.count > 0)
        {
            if (!kp.host)
                return fail("cusiftml_keypoints_copy: the set has no host copy");
            std::memcpy(data, kp.host, (size_t)kp.count * sizeof(cusift_keypoint));
        }
        return ok();
    }

    int cusiftml_keypoints_save_json(unsigned long long keypoints, const char *path)
    {
        Tables &t = tables();
        Lock lock(t.mutex);
        auto it = t.keypoints.find(keypoints);
        if (it == t.keypoints.end())
            return fail("cusiftml_keypoints_save_json: " + no_such("keypoint set", keypoints));
        cusift_keypoints_save_json(path, &it->second);
        return check("cusift_keypoints_save_json");
    }

    int cusiftml_keypoints_free(unsigned long long keypoints)
    {
        Tables &t = tables();
        Lock lock(t.mutex);
        auto it = t.keypoints.find(keypoints);
        if (it != t.keypoints.end())
        {
            cusift_keypoints_free(&it->second);
            t.keypoints.erase(it);
        }
        return ok();
    }

    int cusiftml_match(unsigned long long query, unsigned long long train)
    {
        Tables &t = tables();
        Lock lock(t.mutex);
        auto q = t.keypoints.find(query);
        if (q == t.keypoints.end())
            return fail("cusiftml_match: query " + no_such("keypoint set", query));
        auto r = t.keypoints.find(train);
        if (r == t.keypoints.end())
            return fail("cusiftml_match: train " + no_such("keypoint set", train));
        cusift_match(&q->second, &r->second);
        return check("cusift_match");
    }

    int cusiftml_estimate_homography(unsigned long long matched, const cusiftml_homography_options *options,
                                     float *homography, int *inliers)
    {
        if (!homography || !inliers)
            return fail("cusiftml_estimate_homography: homography or inliers (outputs) is NULL");
        Tables &t = tables();
        Lock lock(t.mutex);
        auto it = t.keypoints.find(matched);
        if (it == t.keypoints.end())
            return fail("cusiftml_estimate_homography: " + no_such("keypoint set", matched));
        cusift_homography_options ho = to_c(options);
        cusift_estimate_homography(&it->second, homography, inliers, &ho);
        return check("cusift_estimate_homography");
    }

    /* -- Canvas and warps --------------------------------------------------- */

    int cusiftml_compute_canvas(const float *homography, int width1, int height1, int width2, int height2,
                                int *canvas, float *corners)
    {
        if (!canvas || !corners)
            return fail("cusiftml_compute_canvas: canvas or corners (outputs) is NULL");
        cusift_canvas c{};
        cusift_compute_canvas(homography, width1, height1, width2, height2, &c);
        // Filled even when the library rejects the canvas size, as cusift does.
        canvas[0] = c.origin_u;
        canvas[1] = c.origin_v;
        canvas[2] = c.canvas_width;
        canvas[3] = c.canvas_height;
        for (int i = 0; i < 4; ++i)
        {
            corners[2 * i]         = c.corners1[i][0];
            corners[2 * i + 1]     = c.corners1[i][1];
            corners[8 + 2 * i]     = c.corners2[i][0];
            corners[8 + 2 * i + 1] = c.corners2[i][1];
        }
        return check("cusift_compute_canvas");
    }

    int cusiftml_warp(const float *pixels1, int width1, int height1, const float *pixels2, int width2, int height2,
                      const float *homography, int use_gpu, unsigned long long *warped1, unsigned long long *warped2)
    {
        if (!warped1 || !warped2)
            return fail("cusiftml_warp: warped1 or warped2 (output handles) is NULL");
        *warped1 = *warped2 = 0;
        cusift_image image1 = view(pixels1, width1, height1);
        cusift_image image2 = view(pixels2, width2, height2);
        cusift_image out1{}, out2{};
        cusift_warp(&image1, &image2, homography, &out1, &out2, use_gpu != 0);
        if (check("cusift_warp"))
        {
            release(out1);
            release(out2);
            return 1;
        }
        Tables &t = tables();
        Lock lock(t.mutex);
        *warped1 = store(t, out1);
        *warped2 = store(t, out2);
        return 0;
    }

    int cusiftml_image_info(unsigned long long image, int *width, int *height)
    {
        Tables &t = tables();
        Lock lock(t.mutex);
        auto it = t.images.find(image);
        if (it == t.images.end())
            return fail("cusiftml_image_info: " + no_such("warped image", image));
        if (width)
            *width = it->second.width;
        if (height)
            *height = it->second.height;
        return ok();
    }

    int cusiftml_image_copy(unsigned long long image, float *pixels, int width, int height)
    {
        if (!pixels)
            return fail("cusiftml_image_copy: pixels is NULL");
        Tables &t = tables();
        Lock lock(t.mutex);
        auto it = t.images.find(image);
        if (it == t.images.end())
            return fail("cusiftml_image_copy: " + no_such("warped image", image));
        const cusift_image &im = it->second;
        if (width != im.width || height != im.height)
            return fail("cusiftml_image_copy: buffer is " + std::to_string(width) + "x" + std::to_string(height) +
                        ", the image is " + std::to_string(im.width) + "x" + std::to_string(im.height));
        if (!im.pixels)
            return fail("cusiftml_image_copy: the image has no pixels");
        std::memcpy(pixels, im.pixels, (size_t)im.width * (size_t)im.height * sizeof(float));
        return ok();
    }

    int cusiftml_image_free(unsigned long long image)
    {
        Tables &t = tables();
        Lock lock(t.mutex);
        auto it = t.images.find(image);
        if (it != t.images.end())
        {
            release(it->second);
            t.images.erase(it);
        }
        return ok();
    }

    /* -- Pipelines ---------------------------------------------------------- */

    int cusiftml_extract_and_match(const float *pixels1, int width1, int height1, const float *pixels2, int width2,
                                   int height2, const cusiftml_extract_options *options,
                                   unsigned long long *keypoints1, unsigned long long *keypoints2)
    {
        if (!keypoints1 || !keypoints2)
            return fail("cusiftml_extract_and_match: keypoints1 or keypoints2 (output handles) is NULL");
        *keypoints1 = *keypoints2 = 0;
        cusift_image image1       = view(pixels1, width1, height1);
        cusift_image image2       = view(pixels2, width2, height2);
        cusift_extract_options eo = to_c(options);
        cusift_keypoints kp1{}, kp2{};
        cusift_extract_and_match(&image1, &image2, &kp1, &kp2, &eo);
        if (check("cusift_extract_and_match"))
        {
            cusift_keypoints_free(&kp1);
            cusift_keypoints_free(&kp2);
            return 1;
        }
        Tables &t = tables();
        Lock lock(t.mutex);
        *keypoints1 = store(t, kp1);
        *keypoints2 = store(t, kp2);
        return 0;
    }

    int cusiftml_align(const float *pixels1, int width1, int height1, const float *pixels2, int width2, int height2,
                       const cusiftml_extract_options *extract_options,
                       const cusiftml_homography_options *homography_options, unsigned long long *keypoints1,
                       unsigned long long *keypoints2, float *homography, int *inliers)
    {
        if (!keypoints1 || !keypoints2 || !homography || !inliers)
            return fail("cusiftml_align: an output pointer is NULL");
        *keypoints1 = *keypoints2 = 0;
        cusift_image image1          = view(pixels1, width1, height1);
        cusift_image image2          = view(pixels2, width2, height2);
        cusift_extract_options eo    = to_c(extract_options);
        cusift_homography_options ho = to_c(homography_options);
        cusift_keypoints kp1{}, kp2{};
        cusift_align(&image1, &image2, &kp1, &kp2, &eo, &ho, homography, inliers);
        if (check("cusift_align"))
        {
            cusift_keypoints_free(&kp1);
            cusift_keypoints_free(&kp2);
            return 1;
        }
        Tables &t = tables();
        Lock lock(t.mutex);
        *keypoints1 = store(t, kp1);
        *keypoints2 = store(t, kp2);
        return 0;
    }

    int cusiftml_align_and_warp(const float *pixels1, int width1, int height1, const float *pixels2, int width2,
                                int height2, const cusiftml_extract_options *extract_options,
                                const cusiftml_homography_options *homography_options,
                                unsigned long long *keypoints1, unsigned long long *keypoints2, float *homography,
                                int *inliers, unsigned long long *warped1, unsigned long long *warped2)
    {
        if (!keypoints1 || !keypoints2 || !homography || !inliers || !warped1 || !warped2)
            return fail("cusiftml_align_and_warp: an output pointer is NULL");
        *keypoints1 = *keypoints2 = 0;
        *warped1 = *warped2 = 0;
        cusift_image image1          = view(pixels1, width1, height1);
        cusift_image image2          = view(pixels2, width2, height2);
        cusift_extract_options eo    = to_c(extract_options);
        cusift_homography_options ho = to_c(homography_options);
        cusift_keypoints kp1{}, kp2{};
        cusift_image out1{}, out2{};
        cusift_align_and_warp(&image1, &image2, &kp1, &kp2, &eo, &ho, homography, inliers, &out1, &out2);
        if (check("cusift_align_and_warp"))
        {
            cusift_keypoints_free(&kp1);
            cusift_keypoints_free(&kp2);
            release(out1);
            release(out2);
            return 1;
        }
        Tables &t = tables();
        Lock lock(t.mutex);
        *keypoints1 = store(t, kp1);
        *keypoints2 = store(t, kp2);
        *warped1    = store(t, out1);
        *warped2    = store(t, out2);
        return 0;
    }

    int cusiftml_free_all(void)
    {
        Tables &t = tables();
        Lock lock(t.mutex);
        for (auto &entry : t.keypoints)
            cusift_keypoints_free(&entry.second);
        t.keypoints.clear();
        for (auto &entry : t.images)
            release(entry.second);
        t.images.clear();
        return ok();
    }

    /* -- Device memory estimates -------------------------------------------- */

    unsigned long long cusiftml_memory_extract(int width, int height, const cusiftml_extract_options *options)
    {
        if (!dims_ok("cusiftml_memory_extract", width, height))
            return 0;
        cusift_extract_options eo = to_c(options);
        if (!capacity_ok("cusiftml_memory_extract: max_keypoints", eo.max_keypoints))
            return 0;
        ok();
        return cusift_memory_extract(width, height, &eo);
    }

    unsigned long long cusiftml_memory_match(int capacity1, int capacity2)
    {
        if (!capacity_ok("cusiftml_memory_match: capacity1", capacity1) ||
            !capacity_ok("cusiftml_memory_match: capacity2", capacity2))
            return 0;
        ok();
        return cusift_memory_match(capacity1, capacity2);
    }

    unsigned long long cusiftml_memory_homography(int capacity, const cusiftml_homography_options *options)
    {
        if (!capacity_ok("cusiftml_memory_homography: capacity", capacity))
            return 0;
        cusift_homography_options ho = to_c(options);
        ok();
        return cusift_memory_homography(capacity, &ho);
    }

    unsigned long long cusiftml_memory_warp(int width1, int height1, int width2, int height2)
    {
        if (!dims_ok("cusiftml_memory_warp: image1", width1, height1) ||
            !dims_ok("cusiftml_memory_warp: image2", width2, height2))
            return 0;
        ok();
        return cusift_memory_warp(width1, height1, width2, height2);
    }

    unsigned long long cusiftml_memory_pipeline(int width1, int height1, int width2, int height2,
                                                const cusiftml_extract_options *extract_options,
                                                const cusiftml_homography_options *homography_options)
    {
        if (!dims_ok("cusiftml_memory_pipeline: image1", width1, height1) ||
            !dims_ok("cusiftml_memory_pipeline: image2", width2, height2))
            return 0;
        cusift_extract_options eo    = to_c(extract_options);
        cusift_homography_options ho = to_c(homography_options);
        if (!capacity_ok("cusiftml_memory_pipeline: max_keypoints", eo.max_keypoints))
            return 0;
        ok();
        return cusift_memory_pipeline(width1, height1, width2, height2, &eo, &ho);
    }

} // extern "C"
