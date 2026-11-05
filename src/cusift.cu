#include "cusift.h"
#include "cudautils.h"
#include "cudaImage.h"
#include "cudaSift.h"
#include "cudaSiftH.h"
#include "geomFuncs.h"
#include "RAII_Guards.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cfloat>
#include <climits>
#include <string>
#include <random>
#include <omp.h>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <thread>
#include <memory>
#include <limits>

#define ERROR(msg) cusift_throw_error(__FILE__, __LINE__, msg)

static void cusift_throw_error(const char *file, int line, const char *msg)
{
    throw CusiftError(file, line, msg);
}

static int p_iAlignUp(int a, int b) { return (a % b != 0) ? (a - a % b + b) : a; }

// ── Argument validation ─────────────────────────────────
// Every public entry point validates its arguments before touching the GPU so
// that a bad call is reported through cusift_last_error() instead of crashing or
// silently corrupting memory.

static void RequireNonNull(const void *p, const char *what)
{
    if (!p)
        throw CusiftError(__FILE__, __LINE__, std::string("Required argument is NULL: ") + what);
}

// Largest max_keypoints whose byte count fits a 32-bit int; InitSiftData
// checks this too, but failing here is cheaper and gives a clearer message.
static const int kMaxKeypointsLimit = (int)(INT_MAX / sizeof(SiftPoint));
// TestHomographies / TestSimilarities put one 16-hypothesis block per grid.y.
static const int kMaxRansacLoops = 65535 * 16;

// Image dimension limits.  The minimum keeps every level of the three octaves
// the library always builds non-empty (8 -> 4 -> 2); below it the pyramid
// kernels failed with an opaque "invalid argument".  The maximum is set by the
// 16-bit candidate-position storage in FindPointsMultiNew (width) and by the
// one-block-per-row Laplace grid and the 2-D texture limit (height).
static const int kMinImageDim = 8;
static const int kMaxImageDim = 65535;

static void ValidateImage(const cusift_image *img, const char *what)
{
    RequireNonNull(img, what);
    if (!img->pixels)
        throw CusiftError(__FILE__, __LINE__, std::string(what) + ": pixels is NULL");
    if (img->width <= 0 || img->height <= 0)
        throw CusiftError(__FILE__, __LINE__, std::string(what) + ": width and height must be positive");
    if (img->width < kMinImageDim || img->height < kMinImageDim)
        throw CusiftError(__FILE__, __LINE__, std::string(what) + ": image must be at least " + std::to_string(kMinImageDim) + " pixels in each dimension");
    if (img->width > kMaxImageDim || img->height > kMaxImageDim)
        throw CusiftError(__FILE__, __LINE__, std::string(what) + ": image dimensions must not exceed " + std::to_string(kMaxImageDim) + " pixels");
    // Row offsets inside the kernels are 32-bit; one pitched plane must fit.
    if ((size_t)p_iAlignUp(img->width, 128) * (size_t)img->height > (size_t)INT_MAX)
        throw CusiftError(__FILE__, __LINE__, std::string(what) + ": image too large (pitch * height must fit in a 32-bit int)");
    // A cudaMalloc pointer is not CPU-readable: the CPU warp path would fault on
    // it and the host-to-device copies would fail.  Classify it and reject it
    // before anything dereferences it.  Pinned and managed memory are host
    // readable and pass.  Older runtimes report plain pageable memory through
    // the error return instead of cudaMemoryTypeUnregistered; that is host
    // memory, so clear the flag and accept it.
    cudaPointerAttributes attr;
    cudaError_t perr = cudaPointerGetAttributes(&attr, img->pixels);
    if (perr == cudaSuccess)
    {
        if (attr.type == cudaMemoryTypeDevice)
            throw CusiftError(__FILE__, __LINE__, std::string(what) + ": pixels points to device memory; pass host, pinned or managed memory");
    }
    else
    {
        (void)cudaGetLastError();
    }
}

static void ValidateExtractOptions(const cusift_extract_options *o)
{
    RequireNonNull(o, "extract options");
    if (o->max_keypoints <= 0 || o->max_keypoints > kMaxKeypointsLimit)
        throw CusiftError(__FILE__, __LINE__, "max_keypoints must be in [1, " + std::to_string(kMaxKeypointsLimit) + "]");
}

static void ValidateHomographyOptions(const cusift_homography_options *o)
{
    RequireNonNull(o, "homography options");
    if (o->ransac_iterations <= 0 || o->ransac_iterations > kMaxRansacLoops)
        throw CusiftError(__FILE__, __LINE__, "ransac_iterations must be in [1, " + std::to_string(kMaxRansacLoops) + "]");
    if (o->refine_iterations < 0)
        throw CusiftError(__FILE__, __LINE__, "refine_iterations must be >= 0");
    if (o->model != CUSIFT_MODEL_HOMOGRAPHY && o->model != CUSIFT_MODEL_SIMILARITY)
        throw CusiftError(__FILE__, __LINE__, "model must be CUSIFT_MODEL_HOMOGRAPHY or CUSIFT_MODEL_SIMILARITY");
    if (o->attempts < 1)
        throw CusiftError(__FILE__, __LINE__, "attempts must be >= 1");
    if (o->goal != CUSIFT_GOAL_MAX_INLIERS && o->goal != CUSIFT_GOAL_MIN_IDENTITY_DISTANCE)
        throw CusiftError(__FILE__, __LINE__, "goal must be CUSIFT_GOAL_MAX_INLIERS or CUSIFT_GOAL_MIN_IDENTITY_DISTANCE");
}

// Output SiftData contract (see cusift.h): the caller zero-initialises the
// struct once; after that it may hold buffers from a previous call, which are
// released here so re-using one SiftData across frames does not leak.  Leaves
// the struct empty (count = capacity = 0, NULL buffers) so cusift_keypoints_free is
// safe even if this call fails before InitSiftData runs.
static void PrepareSiftData(SiftData *sd, const char *what)
{
    RequireNonNull(sd, what);
    FreeSiftData(sd);
}

// Dispatch to either full homography or similarity RANSAC based on model
static void FindGeometricModel(SiftData *data, float *homography, int *numMatches,
                                const cusift_homography_options *opts)
{
    if (opts->model == CUSIFT_MODEL_SIMILARITY)
    {
        FindSimilarity_private(
            data, homography, numMatches,
            opts->ransac_iterations, opts->min_score, opts->max_ambiguity,
            opts->inlier_threshold, opts->seed);
    }
    else
    {
        FindHomography_private(
            data, homography, numMatches,
            opts->ransac_iterations, opts->min_score, opts->max_ambiguity,
            opts->inlier_threshold, opts->seed);
    }
}

static bool Invert3x3(const float *M, float *out);
static int clampOctaves(int requested, int width, int height);
static inline float SampleBilinear(const float *img, int w, int h, float x, float y);
static __global__ void WarpDualKernel(
    const float *__restrict__ src1, float *__restrict__ dst1, int src1W, int src1H, int src1Pitch,
    const float *__restrict__ src2, float *__restrict__ dst2, int src2W, int src2H, int src2Pitch,
    int dstW, int dstH, int dst1Pitch, int dst2Pitch, float originU, float originV,
    float h00, float h01, float h02, float h10, float h11, float h12, float h20, float h21, float h22);

// ── Thread-local error storage ──────────────────────────
struct CusiftErrorState
{
    bool        had_error  = false;
    int         line       = 0;
    cudaError_t cuda_error = cudaSuccess;
    std::string file;
    std::string message;
    std::string function;      // public function that failed
    long long   timestamp_ms = 0;

    void clear()
    {
        had_error    = false;
        line         = 0;
        cuda_error   = cudaSuccess;
        file.clear();
        message.clear();
        function.clear();
        timestamp_ms = 0;
    }
};

static thread_local CusiftErrorState s_error;

// ── Error sinks: log file and callback ──────────────────
// Every failure of a public function passes through cusift_api_guard, which
// hands it to cusift_report().  cusift_report() forwards the record to the
// optional callback and the optional log file.  Reporting never throws and
// never changes the outcome of the call being reported.
namespace
{
struct ErrorSinkConfig
{
    std::mutex              mutex;
    bool                    api_configured = false; // cusift_set_error_log() was called: the environment is ignored
    bool                    env_checked    = false;
    std::string             log_path;               // empty = file logging off
    size_t                  max_bytes      = 0;
    cusift_error_callback   callback       = nullptr;
    void                   *user_data      = nullptr;
    unsigned long long      dropped        = 0;
};

// Function-local static: constructed on first use, so it is valid however
// early a caller (or a static initialiser in the host program) reports.
ErrorSinkConfig &sinks()
{
    static ErrorSinkConfig c;
    return c;
}

const size_t kDefaultLogMaxBytes = (size_t)1 << 20;

std::string default_log_path()
{
    std::error_code ec;
    std::filesystem::path dir = std::filesystem::temp_directory_path(ec);
    if (ec)
        dir = ".";
    return (dir / "cusift-errors.log").string();
}

// Mutex held.
void apply_log_setting(ErrorSinkConfig &c, const char *path, size_t max_bytes)
{
    if (!path)
    {
        c.log_path.clear();
        c.max_bytes = 0;
        return;
    }
    c.log_path  = (*path == '\0') ? default_log_path() : std::string(path);
    c.max_bytes = max_bytes ? max_bytes : kDefaultLogMaxBytes;
}

// Mutex held.  CUSIFT_ERROR_LOG is read once, and only if the API was never used.
void read_env_once(ErrorSinkConfig &c)
{
    if (c.env_checked)
        return;
    c.env_checked = true;
    if (c.api_configured)
        return;
    const char *p = std::getenv("CUSIFT_ERROR_LOG");
    if (!p || !*p)
        return;
    size_t max_bytes = 0;
    if (const char *m = std::getenv("CUSIFT_ERROR_LOG_MAX_BYTES"))
        max_bytes = (size_t)std::strtoull(m, nullptr, 10);
    apply_log_setting(c, (std::strcmp(p, "default") == 0) ? "" : p, max_bytes);
}

std::string format_record(const cusift_error_record &r)
{
    std::time_t secs = (std::time_t)(r.timestamp_ms / 1000);
    int ms = (int)(r.timestamp_ms % 1000);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &secs);
#else
    gmtime_r(&secs, &tm);
#endif
    char stamp[40];
    std::snprintf(stamp, sizeof(stamp), "%04d-%02d-%02dT%02d:%02d:%02d.%03dZ",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, ms);
    std::ostringstream line;
    line << stamp << ' ' << (r.severity == 0 ? "error  " : "warning")
         << " tid=0x" << std::hex << r.thread_id << std::dec
         << ' ' << r.function;
    if (r.file && *r.file)
        line << ' ' << r.file << ':' << r.line;
    if (r.cuda_error != 0)
        line << " cuda=" << r.cuda_error << '(' << r.cuda_error_name << ')';
    line << ' ' << r.message << '\n';
    return line.str();
}

// Mutex held.  Rotate when the next line would push the file past the cap.
void write_line(ErrorSinkConfig &c, const std::string &line)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path p(c.log_path);
    auto size = fs::file_size(p, ec);
    if (!ec && size + line.size() > c.max_bytes)
    {
        fs::path old = p;
        old += ".1";
        fs::remove(old, ec);
        ec.clear();
        fs::rename(p, old, ec);
        if (ec)
            fs::remove(p, ec); // could not rotate: start over in place
    }
    std::ofstream f(p, std::ios::app);
    if (!f)
    {
        c.dropped++;
        return;
    }
    f << line;
    f.close();
    if (!f)
        c.dropped++;
}
} // namespace

// Build a record and hand it to the sinks.  noexcept by construction: it is
// called from error paths, from destructors during unwinding, and from C.
static void cusift_report(int severity, const char *function, const char *file, int line,
                          cudaError_t cuda_error, const char *message) noexcept
{
    try
    {
        cusift_error_record rec{};
        rec.severity        = severity;
        rec.function        = function ? function : "";
        rec.file            = file ? file : "";
        rec.line            = line;
        rec.cuda_error      = (int)cuda_error;
        rec.cuda_error_name = (cuda_error != cudaSuccess) ? cudaGetErrorName(cuda_error) : "";
        rec.message         = message ? message : "";
        rec.thread_id       = (unsigned long long)std::hash<std::thread::id>{}(std::this_thread::get_id());
        rec.timestamp_ms    = (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
                                  std::chrono::system_clock::now().time_since_epoch()).count();

        cusift_error_callback cb = nullptr;
        void *ud = nullptr;
        {
            ErrorSinkConfig &c = sinks();
            std::lock_guard<std::mutex> lock(c.mutex);
            read_env_once(c);
            if (!c.log_path.empty())
                write_line(c, format_record(rec));
            cb = c.callback;
            ud = c.user_data;
        }
        if (cb)
            cb(&rec, ud); // outside the lock: the callback may reconfigure the sinks
    }
    catch (...)
    {
        // Reporting must never throw.
    }
}

// C-callable warning entry used by geomFuncs.c and the cleanup paths.
void cusift_report_warning(const char *function, const char *message)
{
    cusift_report(1, function, "", 0, cudaSuccess, message);
}

// Record a failure in the thread-local state and hand it to the sinks.
static void cusift_record_failure(const char *api, const std::string &file, int line, cudaError_t cuda_error, const std::string &message)
{
    s_error.had_error    = true;
    s_error.function     = api ? api : "";
    s_error.file         = file;
    s_error.line         = line;
    s_error.cuda_error   = cuda_error;
    s_error.message      = message;
    s_error.timestamp_ms = (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::system_clock::now().time_since_epoch()).count();
    cusift_report(0, api, file.c_str(), line, cuda_error, message.c_str());
}

// Wrap an API body: clear error state, run fn(), catch and record any
// exception.  `api` is the public function's name.
template <typename F>
static void cusift_api_guard(const char *api, F &&fn)
{
    s_error.clear();
    try
    {
        fn();
    }
    catch (const CusiftError &e)
    {
        cusift_record_failure(api, e.file, e.line, e.cuda_error, e.what());
    }
    catch (const std::exception &e)
    {
        // STL/host exceptions (e.g. std::bad_alloc) lack a source location.
        cusift_record_failure(api, "", 0, cudaSuccess, e.what());
    }
    catch (...)
    {
        cusift_record_failure(api, "", 0, cudaSuccess, "Unknown exception");
    }
}

void cusift_version(int *major, int *minor, int *patch)
{
    if (major) *major = CUSIFT_VERSION_MAJOR;
    if (minor) *minor = CUSIFT_VERSION_MINOR;
    if (patch) *patch = CUSIFT_VERSION_PATCH;
}

int cusift_last_error(cusift_error_record *record)
{
    if (!s_error.had_error)
        return 0;
    if (record)
    {
        record->severity        = 0;
        record->function        = s_error.function.c_str();
        record->file            = s_error.file.c_str();
        record->line            = s_error.line;
        record->cuda_error      = (int)s_error.cuda_error;
        record->cuda_error_name = (s_error.cuda_error != cudaSuccess) ? cudaGetErrorName(s_error.cuda_error) : "";
        record->message         = s_error.message.c_str();
        record->thread_id       = (unsigned long long)std::hash<std::thread::id>{}(std::this_thread::get_id());
        record->timestamp_ms    = s_error.timestamp_ms;
    }
    return 1;
}

// The documented defaults (see cusift.h).  These do not touch the error state.
void cusift_default_extract_options(cusift_extract_options *o)
{
    if (!o)
        return;
    o->contrast_threshold       = 2.0f;
    o->lowest_scale             = 0.0f;
    o->highest_scale            = std::numeric_limits<float>::infinity();
    o->edge_threshold           = 10.0f;
    o->initial_blur             = 1.0f;
    o->max_keypoints            = 32768;
    o->num_octaves              = 5;
    o->scale_suppression_radius = 0.0f;
}

void cusift_default_homography_options(cusift_homography_options *o)
{
    if (!o)
        return;
    o->ransac_iterations       = 10000;
    o->min_score               = 0.0f;
    o->max_ambiguity           = 0.8f;
    o->inlier_threshold        = 3.0f;
    o->refine_iterations       = 5;
    o->refine_min_score        = 0.0f;
    o->refine_max_ambiguity    = 0.8f;
    o->refine_inlier_threshold = 2.0f;
    o->seed                    = 0;
    o->model                   = CUSIFT_MODEL_HOMOGRAPHY;
    o->attempts                = 1;
    o->goal                    = CUSIFT_GOAL_MAX_INLIERS;
}

void cusift_init()
{
    cusift_api_guard(__func__, [&]()
    {
        int nDevices;
        safeCall(cudaGetDeviceCount(&nDevices));
        if (!nDevices)
            ERROR("No CUDA devices available");
        int devNum = std::min(nDevices - 1, 0);
        safeCall(cudaSetDevice(devNum));
    });
}

void cusift_device_reset(void)
{
    cusift_api_guard(__func__, [&]()
    {
        // cudaDeviceReset tears down this process's context on the current
        // device only; other processes are unaffected.  Its own return value is
        // not a useful health signal (it may echo the sticky error it is
        // clearing), so drop it and let a fresh context creation decide.
        (void)cudaDeviceReset();
        (void)cudaGetLastError();
        safeCall(cudaFree(0)); // creates the new primary context; fails if the device is still unusable
    });
}

// The sink configuration functions never fail and deliberately leave the
// thread-local error state alone, like the cleanup functions.
void cusift_set_error_log(const char *path, size_t max_bytes)
{
    try
    {
        ErrorSinkConfig &c = sinks();
        std::lock_guard<std::mutex> lock(c.mutex);
        c.api_configured = true;
        c.env_checked    = true;
        apply_log_setting(c, path, max_bytes);
    }
    catch (...)
    {
    }
}

void cusift_set_error_callback(cusift_error_callback callback, void *user_data)
{
    try
    {
        ErrorSinkConfig &c = sinks();
        std::lock_guard<std::mutex> lock(c.mutex);
        c.callback  = callback;
        c.user_data = user_data;
    }
    catch (...)
    {
    }
}

unsigned long long cusift_error_log_dropped(void)
{
    try
    {
        ErrorSinkConfig &c = sinks();
        std::lock_guard<std::mutex> lock(c.mutex);
        return c.dropped;
    }
    catch (...)
    {
        return 0;
    }
}

void cusift_extract(const cusift_image *image, SiftData *sift_data, const cusift_extract_options *options)
{
    cusift_api_guard(__func__, [&]()
    {
        ValidateImage(image, "image");
        ValidateExtractOptions(options);
        PrepareSiftData(sift_data, "sift_data");

        CudaImageGuard cuda_image;

        InitSiftData(sift_data, options->max_keypoints, true, true);

        CudaImage_Allocate(
            cuda_image.get(),
            image->width,
            image->height,
            p_iAlignUp(image->width, 128),
            false,
            nullptr,
            image->pixels);

        CudaImage_Download(cuda_image.get());
        //CudaImage_Normalize(cuda_image.get());

        // Get the smallest dimension of the image to determine the maximum number of octaves
        int minDim = std::min(image->width, image->height);
        int maxOctaves = static_cast<int>(std::floor(std::log2(minDim))) - 3;
        int octaves = std::min(options->num_octaves, maxOctaves);
        // Limit num octaves to 7, quietly
        octaves = std::min(octaves, 7);
        octaves = std::max(octaves, 3); // Ensure at least 3 octaves

        SiftTempMemoryGuard tempMemory(AllocSiftTempMemory(image->width, image->height, octaves));

        ExtractSift(
            sift_data,
            cuda_image.get(),
            octaves,
            options->initial_blur,
            options->contrast_threshold,
            options->lowest_scale,
            options->highest_scale,
            options->edge_threshold,
            tempMemory.get());
        if (options->scale_suppression_radius > 0.0f)
            SuppressEmbeddedPoints(sift_data, options->scale_suppression_radius);
    });
}

void cusift_match(SiftData *query, const SiftData *train)
{
    cusift_api_guard(__func__, [&]()
    {
        RequireNonNull(query, "query");
        RequireNonNull(train, "train");
        // The matcher only reads the train set; the internal signature predates const.
        MatchSiftData_private(query, const_cast<SiftData *>(train));
    });
}

// Squared Frobenius distance between the upper-left 2x2 block and the
// identity: the CUSIFT_GOAL_MIN_IDENTITY_DISTANCE criterion.
static float EyeDiff2x2(const float *H)
{
    float d00 = H[0] - 1.0f;
    float d01 = H[1];
    float d10 = H[3];
    float d11 = H[4] - 1.0f;
    return d00 * d00 + d01 * d01 + d10 * d10 + d11 * d11;
}

// RANSAC plus refinement, run options->attempts times and reduced to one
// result by options->goal.  attempts == 1 is the plain single estimate and
// passes the caller's seed through unchanged (0 = random inside the sampler),
// so the result is bit-identical to the former single-path functions.  With
// several attempts, attempt k uses seed + k, or a fresh random seed per
// attempt when seed is 0.
static void EstimateHomographyImpl(SiftData *data, const cusift_homography_options *options, float *homography, int *inliers)
{
    const bool multi = options->attempts > 1;
    cusift_homography_options opts = *options;
    std::unique_ptr<std::random_device> rd;   // constructed only if a random seed is actually needed
    auto draw = [&]() -> unsigned int
    {
        if (!rd)
            rd.reset(new std::random_device());
        unsigned int v = (*rd)();
        return v ? v : 1u;
    };

    float bestH[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int   bestInliers = -1;
    float bestScore   = FLT_MAX;
    for (int attempt = 0; attempt < options->attempts; attempt++)
    {
        if (multi)
            opts.seed = (options->seed == 0) ? draw() : options->seed + (unsigned int)attempt;

        float H[9];
        int   n = 0;
        FindGeometricModel(data, H, &n, &opts);

        bool finite = true;
        for (int i = 0; i < 9; i++)
            finite = finite && std::isfinite(H[i]);
        if (!finite)
        {
            if (!multi)
                ERROR("Homography contains non-finite values");
            continue;
        }

        ImproveHomography(data, H, opts.refine_iterations, opts.refine_min_score, opts.refine_max_ambiguity, opts.refine_inlier_threshold);

        bool better;
        if (!multi)
            better = true;
        else if (options->goal == CUSIFT_GOAL_MIN_IDENTITY_DISTANCE)
        {
            float s = EyeDiff2x2(H);
            better = (bestInliers < 0 || s < bestScore);
            if (better)
                bestScore = s;
        }
        else
            better = (n > bestInliers);

        if (better)
        {
            std::memcpy(bestH, H, sizeof(bestH));
            bestInliers = n;
        }
    }
    if (bestInliers < 0)
        ERROR("Homography contains non-finite values in all attempts");
    std::memcpy(homography, bestH, sizeof(bestH));
    *inliers = bestInliers;
}

void cusift_estimate_homography(SiftData *matched, float *homography, int *inliers, const cusift_homography_options *options)
{
    cusift_api_guard(__func__, [&]()
    {
        RequireNonNull(matched, "matched");
        RequireNonNull(homography, "homography");
        RequireNonNull(inliers, "inliers");
        ValidateHomographyOptions(options);
        EstimateHomographyImpl(matched, options, homography, inliers);
    });
}

// Resource-cleanup functions (cusift_keypoints_free / cusift_image_free / cusift_device_image_free) do
// NOT pass through cusift_api_guard.  Cleanup is destructor-safe (uses
// cudaFree / free directly, never throws) and we deliberately leave the
// thread-local error state untouched so the typical "do work; then free; then
// check error" pattern still surfaces the original failure.
void cusift_keypoints_free(SiftData *sift_data)
{
    if (sift_data)
        FreeSiftData(sift_data);
}

void cusift_image_free(cusift_image *image)
{
    if (image && image->pixels)
    {
        free(image->pixels);
        image->pixels = nullptr;
        image->width = 0;
        image->height = 0;
    }
}

void cusift_device_image_free(cusift_device_image *image)
{
    if (image && image->device_pixels)
    {
        cudaError_t e = cudaFree(image->device_pixels);
        if (e != cudaSuccess)
        {
            (void)cudaGetLastError();
            char buf[160];
            std::snprintf(buf, sizeof(buf), "cudaFree of the warped image failed: %s", cudaGetErrorString(e));
            cusift_report(1, "cusift_device_image_free", "", 0, e, buf);
        }
        image->device_pixels = nullptr;
        image->width = 0;
        image->height = 0;
        image->pitch_bytes = 0;
    }
}

void cusift_keypoints_save_json(const char *filename, const SiftData *sift_data)
{
    cusift_api_guard(__func__, [&]()
    {
        RequireNonNull(filename, "filename");
        if (!sift_data || !sift_data->host || sift_data->count <= 0)
            ERROR("cusift_keypoints_save_json: no data to save");

        FILE *f = fopen(filename, "w");
        if (!f)
            ERROR("cusift_keypoints_save_json: could not open file for writing");

        fprintf(f, "{\n");
    fprintf(f, "  \"num_keypoints\": %d,\n", sift_data->count);
    fprintf(f, "  \"keypoints\": [\n");

    for (int i = 0; i < sift_data->count; i++)
    {
        const SiftPoint *pt = &sift_data->host[i];
        fprintf(f, "    {\n");
        fprintf(f, "      \"x\": %.6f,\n", pt->x);
        fprintf(f, "      \"y\": %.6f,\n", pt->y);
        fprintf(f, "      \"scale\": %.6f,\n", pt->scale);
        fprintf(f, "      \"sharpness\": %.6f,\n", pt->sharpness);
        fprintf(f, "      \"edgeness\": %.6f,\n", pt->edgeness);
        fprintf(f, "      \"orientation\": %.6f,\n", pt->orientation);
        fprintf(f, "      \"score\": %.6f,\n", pt->score);
        fprintf(f, "      \"ambiguity\": %.6f,\n", pt->ambiguity);
        fprintf(f, "      \"match\": %d,\n", pt->match);
        fprintf(f, "      \"match_x\": %.6f,\n", pt->match_x);
        fprintf(f, "      \"match_y\": %.6f,\n", pt->match_y);
        fprintf(f, "      \"match_error\": %.6f,\n", pt->match_error);
        fprintf(f, "      \"descriptor\": [");
        for (int j = 0; j < 128; j++)
        {
            fprintf(f, "%.6f", pt->descriptor[j]);
            if (j < 127)
                fprintf(f, ", ");
        }
        fprintf(f, "]\n");
        fprintf(f, "    }%s\n", (i < sift_data->count - 1) ? "," : "");
    }

    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    fclose(f);
    });
}

// ── Helper: Frobenius-like distance between 2x2 top-left sub-matrix and I ──
// ── Helper: 3x3 matrix inverse (row-major) ──────────────
bool Invert3x3(const float *M, float *out)
{
    float a = M[0], b = M[1], c = M[2];
    float d = M[3], e = M[4], f = M[5];
    float g = M[6], h = M[7], i = M[8];

    float det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (fabsf(det) < 1e-12f)
        return false;

    float inv_det = 1.0f / det;
    out[0] = (e * i - f * h) * inv_det;
    out[1] = (c * h - b * i) * inv_det;
    out[2] = (b * f - c * e) * inv_det;
    out[3] = (f * g - d * i) * inv_det;
    out[4] = (a * i - c * g) * inv_det;
    out[5] = (c * d - a * f) * inv_det;
    out[6] = (d * h - e * g) * inv_det;
    out[7] = (b * g - a * h) * inv_det;
    out[8] = (a * e - b * d) * inv_det;
    return true;
}

// ── Canvas geometry ─────────────────────────────────────
// The one place that decides where the two images land on the common canvas.
// Every warp entry point and the public cusift_compute_canvas() go through
// here, so the geometry a caller queries is bit-identical to the geometry a
// warp uses.
//
// The canvas is image 1's frame. Image 2's four corners are carried through
// inv(H) into that frame; the canvas is the integer bounding box of those
// corners together with image 1's own rectangle, rounded outward; canvas
// pixel (0, 0) is reference point (origin_u, origin_v).
//
// Fills *g before running the size guards, so a caller that catches the
// "too large" error still sees where the corners went. Throws (via ERROR) on
// a non-finite or singular H, on a canvas larger than twice the larger input,
// and on a non-positive canvas.
static void ComputeCanvasGeometry(const float *homography, int w1, int h1, int w2, int h2, cusift_canvas *g)
{
    std::memset(g, 0, sizeof(*g));

    for (int i = 0; i < 9; i++)
    {
        if (!std::isfinite(homography[i]))
            ERROR("Homography contains non-finite values");
    }
    float Hinv[9];
    if (!Invert3x3(homography, Hinv))
    {
        ERROR("Homography is too close to singular to invert");
    }

    // Corner order (0,0), (w-1,0), (w-1,h-1), (0,h-1); pixel-centre convention.
    const float c1[4][2] = {{0.0f, 0.0f}, {(float)(w1 - 1), 0.0f}, {(float)(w1 - 1), (float)(h1 - 1)}, {0.0f, (float)(h1 - 1)}};
    const float c2[4][2] = {{0.0f, 0.0f}, {(float)(w2 - 1), 0.0f}, {(float)(w2 - 1), (float)(h2 - 1)}, {0.0f, (float)(h2 - 1)}};

    // Project image 2's corners through inv(H) into image 1's frame.
    float cx[4], cy[4];
    for (int i = 0; i < 4; i++)
    {
        float x = c2[i][0], y = c2[i][1];
        float z = Hinv[6] * x + Hinv[7] * y + Hinv[8];
        cx[i] = (Hinv[0] * x + Hinv[1] * y + Hinv[2]) / z;
        cy[i] = (Hinv[3] * x + Hinv[4] * y + Hinv[5]) / z;
    }

    // Bounding box of image 1's rectangle and the projected corners, rounded outward.
    float uMin = 0.0f, uMax = (float)(w1 - 1);
    float vMin = 0.0f, vMax = (float)(h1 - 1);
    for (int i = 0; i < 4; i++)
    {
        uMin = std::min(uMin, cx[i]);
        uMax = std::max(uMax, cx[i]);
        vMin = std::min(vMin, cy[i]);
        vMax = std::max(vMax, cy[i]);
    }
    const int u0 = (int)floorf(uMin);
    const int u1 = (int)ceilf(uMax);
    const int v0 = (int)floorf(vMin);
    const int v1 = (int)ceilf(vMax);

    g->origin_u = u0;
    g->origin_v = v0;
    g->canvas_width = u1 - u0 + 1;
    g->canvas_height = v1 - v0 + 1;
    for (int i = 0; i < 4; i++)
    {
        g->corners1[i][0] = c1[i][0] - (float)u0;
        g->corners1[i][1] = c1[i][1] - (float)v0;
        g->corners2[i][0] = cx[i] - (float)u0;
        g->corners2[i][1] = cy[i] - (float)v0;
    }

    const int outW = g->canvas_width, outH = g->canvas_height;
    const int maxW_ = 2 * std::max(w1, w2);
    const int maxH_ = 2 * std::max(h1, h2);
    if (outW > maxW_ || outH > maxH_)
    {
        std::stringstream ss;
        ss << "Warping: warped image too large (" << outW << "x" << outH << "), not attempting warp";
        ERROR(ss.str().c_str());
    }
    if (outW <= 0 || outH <= 0)
    {
        std::stringstream ss;
        ss << "Warping: invalid output image size (" << outW << "x" << outH << ")";
        ERROR(ss.str().c_str());
    }
}

// ── Bilinear sample (CPU) ───────────────────────────────
inline float SampleBilinear(const float *img, int w, int h, float x, float y)
{
    if (x < 0.0f || x >= (float)(w - 1) || y < 0.0f || y >= (float)(h - 1))
        return std::nanf(""); // Out of bounds; caller should check before sampling but return nan just in case

    int x0 = (int)x;
    int y0 = (int)y;
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Clamp to be safe at exact boundaries
    x1 = x1 < w ? x1 : w - 1;
    y1 = y1 < h ? y1 : h - 1;

    float fx = x - (float)x0;
    float fy = y - (float)y0;

    float v00 = img[y0 * w + x0];
    float v10 = img[y0 * w + x1];
    float v01 = img[y1 * w + x0];
    float v11 = img[y1 * w + x1];

    return (1.0f - fx) * (1.0f - fy) * v00 + fx * (1.0f - fy) * v10 + (1.0f - fx) * fy * v01 + fx * fy * v11;
}

// ── GPU kernel ──────────────────────────────────────────
// Dual warp kernel: warps both images in a single launch.
// Image 1 is identity-warped (translated into canvas).
// Image 2 is warped through the supplied homography.
// srcPitch / dstPitch are row strides in floats (may differ from width
// when the allocation is pitch-aligned).  dst1 and dst2 come from separate
// cudaMallocPitch calls and each is written with its own stride.
__global__ void WarpDualKernel(
    const float *__restrict__ src1, float *__restrict__ dst1, int src1W, int src1H, int src1Pitch,
    const float *__restrict__ src2, float *__restrict__ dst2, int src2W, int src2H, int src2Pitch,
    int dstW, int dstH, int dst1Pitch, int dst2Pitch, float originU, float originV,
    float h00, float h01, float h02,
    float h10, float h11, float h12,
    float h20, float h21, float h22)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH)
        return;

    // Canvas coords
    float u = (float)x + originU;
    float v = (float)y + originV;

    // ── Image 1: identity warp (just translate into canvas) ──
    float val1 = std::nanf(""); // Padded pixels get nan so they show up as black in visualization and don't contribute to error metrics
    if (u >= 0.0f && u < (float)(src1W - 1) && v >= 0.0f && v < (float)(src1H - 1))
    {
        int x0 = (int)u;
        int y0 = (int)v;
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        float fx = u - (float)x0;
        float fy = v - (float)y0;
        val1 = (1.0f - fx) * (1.0f - fy) * src1[y0 * src1Pitch + x0] + fx * (1.0f - fy) * src1[y0 * src1Pitch + x1] + (1.0f - fx) * fy * src1[y1 * src1Pitch + x0] + fx * fy * src1[y1 * src1Pitch + x1];
    }
    dst1[y * dst1Pitch + x] = val1;

    // ── Image 2: homography warp ──
    float z = h20 * u + h21 * v + h22;
    float sx = (h00 * u + h01 * v + h02) / z;
    float sy = (h10 * u + h11 * v + h12) / z;

    float val2 = std::nanf(""); // Padded pixels get nan so they show up as black in visualization and don't contribute to error metrics
    if (sx >= 0.0f && sx < (float)(src2W - 1) && sy >= 0.0f && sy < (float)(src2H - 1))
    {
        int x0 = (int)sx;
        int y0 = (int)sy;
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        float fx = sx - (float)x0;
        float fy = sy - (float)y0;
        val2 = (1.0f - fx) * (1.0f - fy) * src2[y0 * src2Pitch + x0] + fx * (1.0f - fy) * src2[y0 * src2Pitch + x1] + (1.0f - fx) * fy * src2[y1 * src2Pitch + x0] + fx * fy * src2[y1 * src2Pitch + x1];
    }
    dst2[y * dst2Pitch + x] = val2;
}

// ── Main entry point ────────────────────────────────────
void cusift_warp(const cusift_image *image1, const cusift_image *image2, const float *homography,
                cusift_image *warped_image1, cusift_image *warped_image2, bool useGPU)
{
    cusift_api_guard(__func__, [&]()
    {
        ValidateImage(image1, "image1");
        ValidateImage(image2, "image2");
        RequireNonNull(homography, "homography");
        RequireNonNull(warped_image1, "warped_image1");
        RequireNonNull(warped_image2, "warped_image2");

        int w1 = image1->width, h1 = image1->height;
        int w2 = image2->width, h2 = image2->height;

        // ── Step 1: Compute inv(homography) and project image2 corners ──
        cusift_canvas geom;
        ComputeCanvasGeometry(homography, w1, h1, w2, h2, &geom);
        const int outW = geom.canvas_width;
        const int outH = geom.canvas_height;
        const float originU = (float)geom.origin_u;
        const float originV = (float)geom.origin_v;


        // ── Step 3: Allocate output images ──────────────────
        size_t nPixels = (size_t)outW * outH;
        HostPtrGuard<float> out1Guard((float *)malloc(sizeof(float) * nPixels));
        HostPtrGuard<float> out2Guard((float *)malloc(sizeof(float) * nPixels));
        if (!out1Guard.get() || !out2Guard.get())
        {
            ERROR("Warping: host return image allocation failed");
        }

        // Homography row-major elements
        float H00 = homography[0], H01 = homography[1], H02 = homography[2];
        float H10 = homography[3], H11 = homography[4], H12 = homography[5];
        float H20 = homography[6], H21 = homography[7], H22 = homography[8];

        if (useGPU)
        {
            // ── GPU path ────────────────────────────────────
            DevicePtrGuard<float> d_src1Guard, d_src2Guard;
            DevicePtrGuard<float> d_out1Guard, d_out2Guard;

            size_t src1Pitch = 0, src2Pitch = 0;
            size_t out1Pitch = 0, out2Pitch = 0;

            safeCall(cudaMallocPitch(&d_src1Guard.getRef(), &src1Pitch, w1 * sizeof(float), h1));
            safeCall(cudaMallocPitch(&d_src2Guard.getRef(), &src2Pitch, w2 * sizeof(float), h2));
            safeCall(cudaMallocPitch(&d_out1Guard.getRef(), &out1Pitch, outW * sizeof(float), outH));
            safeCall(cudaMallocPitch(&d_out2Guard.getRef(), &out2Pitch, outW * sizeof(float), outH));

            safeCall(cudaMemcpy2D(d_src1Guard.get(), src1Pitch, image1->pixels, w1 * sizeof(float), w1 * sizeof(float), h1, cudaMemcpyHostToDevice));
            safeCall(cudaMemcpy2D(d_src2Guard.get(), src2Pitch, image2->pixels, w2 * sizeof(float), w2 * sizeof(float), h2, cudaMemcpyHostToDevice));

            // Strides in elements (pixels) for the kernel
            int src1StrideElem = (int)(src1Pitch / sizeof(float));
            int src2StrideElem = (int)(src2Pitch / sizeof(float));
            int out1StrideElem = (int)(out1Pitch / sizeof(float));
            int out2StrideElem = (int)(out2Pitch / sizeof(float));
            // out2 stride same width as out1 since both are outW

            dim3 threads(16, 16);
            dim3 blocks((outW + threads.x - 1) / threads.x, (outH + threads.y - 1) / threads.y);

            // Warp both images in a single kernel launch
            WarpDualKernel<<<blocks, threads>>>(
                d_src1Guard.get(), d_out1Guard.get(), w1, h1, src1StrideElem,
                d_src2Guard.get(), d_out2Guard.get(), w2, h2, src2StrideElem,
                outW, outH, out1StrideElem, out2StrideElem, originU, originV,
                H00, H01, H02,
                H10, H11, H12,
                H20, H21, H22);
        checkMsg("WarpDualKernel() execution failed\n");

            safeCall(cudaDeviceSynchronize());

            safeCall(cudaMemcpy2D(out1Guard.get(), outW * sizeof(float), d_out1Guard.get(), out1Pitch, outW * sizeof(float), outH, cudaMemcpyDeviceToHost));
            safeCall(cudaMemcpy2D(out2Guard.get(), outW * sizeof(float), d_out2Guard.get(), out2Pitch, outW * sizeof(float), outH, cudaMemcpyDeviceToHost));

            // d_src1Guard, d_src2Guard, d_out1Guard, d_out2Guard freed automatically
        }
        else
        {
            float *out1 = out1Guard.get();
            float *out2 = out2Guard.get();
// ── CPU path ────────────────────────────────────
#pragma omp parallel for schedule(static)
            for (int y = 0; y < outH; y++)
            {
                float v = (float)y + originV;
                for (int x = 0; x < outW; x++)
                {
                    float u = (float)x + originU;

                    // Warped image1: identity (just sample at canvas coords)
                    out1[y * outW + x] = SampleBilinear(image1->pixels, w1, h1, u, v);

                    // Warped image2: apply homography to get source coords
                    float z = H20 * u + H21 * v + H22;
                    float su = (H00 * u + H01 * v + H02) / z;
                    float sv = (H10 * u + H11 * v + H12) / z;
                    out2[y * outW + x] = SampleBilinear(image2->pixels, w2, h2, su, sv);
                }
            }
        }

        // ── Step 4: Fill output structs, release ownership ──
        warped_image1->pixels = out1Guard.release();
        warped_image1->width = outW;
        warped_image1->height = outH;

        warped_image2->pixels = out2Guard.release();
        warped_image2->width = outW;
        warped_image2->height = outH;
    }); // end cusift_api_guard for cusift_warp
}

void cusift_warp_to_device(const cusift_image *image1, const cusift_image *image2, const float *homography,
                    cusift_device_image *warped_image1, cusift_device_image *warped_image2)
{
    cusift_api_guard(__func__, [&]()
    {
        ValidateImage(image1, "image1");
        ValidateImage(image2, "image2");
        RequireNonNull(homography, "homography");
        RequireNonNull(warped_image1, "warped_image1");
        RequireNonNull(warped_image2, "warped_image2");

        int w1 = image1->width, h1 = image1->height;
        int w2 = image2->width, h2 = image2->height;

        // ── Step 1: Compute inv(homography) and project image2 corners ──
        cusift_canvas geom;
        ComputeCanvasGeometry(homography, w1, h1, w2, h2, &geom);
        const int outW = geom.canvas_width;
        const int outH = geom.canvas_height;
        const float originU = (float)geom.origin_u;
        const float originV = (float)geom.origin_v;


        // ── Step 3: Allocate GPU images ─────────────────────
        DevicePtrGuard<float> d_src1Guard, d_src2Guard;
        DevicePtrGuard<float> d_out1Guard, d_out2Guard;

        size_t src1Pitch = 0, src2Pitch = 0;
        size_t out1Pitch = 0, out2Pitch = 0;

        safeCall(cudaMallocPitch(&d_src1Guard.getRef(), &src1Pitch, w1 * sizeof(float), h1));
        safeCall(cudaMallocPitch(&d_src2Guard.getRef(), &src2Pitch, w2 * sizeof(float), h2));
        safeCall(cudaMallocPitch(&d_out1Guard.getRef(), &out1Pitch, outW * sizeof(float), outH));
        safeCall(cudaMallocPitch(&d_out2Guard.getRef(), &out2Pitch, outW * sizeof(float), outH));

        safeCall(cudaMemcpy2D(d_src1Guard.get(), src1Pitch, image1->pixels, w1 * sizeof(float), w1 * sizeof(float), h1, cudaMemcpyHostToDevice));
        safeCall(cudaMemcpy2D(d_src2Guard.get(), src2Pitch, image2->pixels, w2 * sizeof(float), w2 * sizeof(float), h2, cudaMemcpyHostToDevice));

        int src1StrideElem = (int)(src1Pitch / sizeof(float));
        int src2StrideElem = (int)(src2Pitch / sizeof(float));
        int out1StrideElem = (int)(out1Pitch / sizeof(float));
        int out2StrideElem = (int)(out2Pitch / sizeof(float));

        float H00 = homography[0], H01 = homography[1], H02 = homography[2];
        float H10 = homography[3], H11 = homography[4], H12 = homography[5];
        float H20 = homography[6], H21 = homography[7], H22 = homography[8];

        dim3 threads(16, 16);
        dim3 blocks((outW + threads.x - 1) / threads.x, (outH + threads.y - 1) / threads.y);

        WarpDualKernel<<<blocks, threads>>>(
            d_src1Guard.get(), d_out1Guard.get(), w1, h1, src1StrideElem,
            d_src2Guard.get(), d_out2Guard.get(), w2, h2, src2StrideElem,
            outW, outH, out1StrideElem, out2StrideElem, originU, originV,
            H00, H01, H02,
            H10, H11, H12,
            H20, H21, H22);
        checkMsg("WarpDualKernel() execution failed\n");

        safeCall(cudaDeviceSynchronize());

        // ── Step 4: Fill output structs, release GPU ownership ──
        warped_image1->device_pixels = d_out1Guard.release();
        warped_image1->width = outW;
        warped_image1->height = outH;
        warped_image1->pitch_bytes = out1Pitch;

        warped_image2->device_pixels = d_out2Guard.release();
        warped_image2->width = outW;
        warped_image2->height = outH;
        warped_image2->pitch_bytes = out2Pitch;

        // d_src1Guard, d_src2Guard freed automatically
    }); // end cusift_api_guard for cusift_warp_to_device
}

void cusift_compute_canvas(const float *homography, int image_width1, int image_height1, int image_width2, int image_height2, cusift_canvas *geometry)
{
    cusift_api_guard(__func__, [&]()
    {
        RequireNonNull(geometry, "geometry");
        std::memset(geometry, 0, sizeof(*geometry));
        RequireNonNull(homography, "homography");
        if (image_width1 <= 0 || image_height1 <= 0 || image_width2 <= 0 || image_height2 <= 0)
            throw CusiftError(__FILE__, __LINE__, "image dimensions must be positive");
        ComputeCanvasGeometry(homography, image_width1, image_height1, image_width2, image_height2, geometry);
    });
}

// ── Pipelines ───────────────────────────────────────────
// One implementation behind cusift_extract_and_match, cusift_align,
// cusift_align_and_warp and cusift_align_and_warp_to_device.  Which stages run
// is decided by which outputs were requested: a homography pointer asks for
// the estimate, a warped-image pair asks for the warp (host or device output).
// The device copies uploaded for extraction stay resident and serve as the
// warp sources, so nothing is uploaded twice.

static void ExtractInto(SiftData *kp, CudaImageGuard &img, const cusift_image *image, const cusift_extract_options *eo, int octaves, float *tempMemory)
{
    InitSiftData(kp, eo->max_keypoints, true, true);
    CudaImage_Allocate(img.get(), image->width, image->height, p_iAlignUp(image->width, 128), false, nullptr, image->pixels);
    CudaImage_Download(img.get());
    ExtractSift(kp, img.get(), octaves, eo->initial_blur, eo->contrast_threshold,
                eo->lowest_scale, eo->highest_scale, eo->edge_threshold, tempMemory);
    if (eo->scale_suppression_radius > 0.0f)
        SuppressEmbeddedPoints(kp, eo->scale_suppression_radius);
}

static void RunPipeline(const cusift_image *image1, const cusift_image *image2,
                        SiftData *kp1, SiftData *kp2,
                        const cusift_extract_options *eo, const cusift_homography_options *ho,
                        float *homography, int *inliers,
                        cusift_image *warped_host1, cusift_image *warped_host2,
                        cusift_device_image *warped_device1, cusift_device_image *warped_device2)
{
    const bool estimate    = (homography != nullptr);
    const bool warp_host   = (warped_host1 != nullptr);
    const bool warp_device = (warped_device1 != nullptr);

    ValidateImage(image1, "image1");
    ValidateImage(image2, "image2");
    ValidateExtractOptions(eo);
    PrepareSiftData(kp1, "keypoints1");
    PrepareSiftData(kp2, "keypoints2");
    if (estimate)
    {
        RequireNonNull(inliers, "inliers");
        ValidateHomographyOptions(ho);
    }
    if (warp_host)
        RequireNonNull(warped_host2, "warped2");
    if (warp_device)
        RequireNonNull(warped_device2, "warped2");

    const int w1 = image1->width, h1 = image1->height;
    const int w2 = image2->width, h2 = image2->height;
    // One pyramid buffer sized for the larger image serves both extractions;
    // the octave count is what the smaller image supports.
    const int octaves = clampOctaves(eo->num_octaves, std::min(w1, w2), std::min(h1, h2));
    SiftTempMemoryGuard tempMemory(AllocSiftTempMemory(std::max(w1, w2), std::max(h1, h2), octaves));

    CudaImageGuard cuda_image1, cuda_image2;
    ExtractInto(kp1, cuda_image1, image1, eo, octaves, tempMemory.get());
    ExtractInto(kp2, cuda_image2, image2, eo, octaves, tempMemory.get());
    MatchSiftData_private(kp1, kp2);
    if (!estimate)
        return;

    EstimateHomographyImpl(kp1, ho, homography, inliers);
    if (!warp_host && !warp_device)
        return;

    cusift_canvas geom;
    ComputeCanvasGeometry(homography, w1, h1, w2, h2, &geom);
    const int   outW = geom.canvas_width, outH = geom.canvas_height;
    const float originU = (float)geom.origin_u, originV = (float)geom.origin_v;

    DevicePtrGuard<float> d_out1, d_out2;
    size_t out1Stride = 0, out2Stride = 0;
    safeCall(cudaMallocPitch(&d_out1.getRef(), &out1Stride, outW * sizeof(float), outH));
    safeCall(cudaMallocPitch(&d_out2.getRef(), &out2Stride, outW * sizeof(float), outH));

    const float *H = homography;
    dim3 threads(16, 16);
    dim3 blocks((outW + threads.x - 1) / threads.x, (outH + threads.y - 1) / threads.y);
    WarpDualKernel<<<blocks, threads>>>(
        cuda_image1.get()->d_data, d_out1.get(), w1, h1, cuda_image1.get()->pitch,
        cuda_image2.get()->d_data, d_out2.get(), w2, h2, cuda_image2.get()->pitch,
        outW, outH, (int)(out1Stride / sizeof(float)), (int)(out2Stride / sizeof(float)), originU, originV,
        H[0], H[1], H[2], H[3], H[4], H[5], H[6], H[7], H[8]);
    checkMsg("WarpDualKernel() execution failed\n");
    safeCall(cudaDeviceSynchronize());

    if (warp_device)
    {
        warped_device1->device_pixels = d_out1.release();
        warped_device1->width         = outW;
        warped_device1->height        = outH;
        warped_device1->pitch_bytes   = out1Stride;
        warped_device2->device_pixels = d_out2.release();
        warped_device2->width         = outW;
        warped_device2->height        = outH;
        warped_device2->pitch_bytes   = out2Stride;
        return;
    }

    const size_t nPixels = (size_t)outW * (size_t)outH;
    HostPtrGuard<float> out1Guard((float *)malloc(sizeof(float) * nPixels));
    HostPtrGuard<float> out2Guard((float *)malloc(sizeof(float) * nPixels));
    if (!out1Guard.get() || !out2Guard.get())
        ERROR("Warping: return image host allocation failed");
    safeCall(cudaMemcpy2D(out1Guard.get(), outW * sizeof(float), d_out1.get(), out1Stride, outW * sizeof(float), outH, cudaMemcpyDeviceToHost));
    safeCall(cudaMemcpy2D(out2Guard.get(), outW * sizeof(float), d_out2.get(), out2Stride, outW * sizeof(float), outH, cudaMemcpyDeviceToHost));
    warped_host1->pixels = out1Guard.release();
    warped_host1->width  = outW;
    warped_host1->height = outH;
    warped_host2->pixels = out2Guard.release();
    warped_host2->width  = outW;
    warped_host2->height = outH;
}

void cusift_extract_and_match(const cusift_image *image1, const cusift_image *image2, SiftData *keypoints1, SiftData *keypoints2, const cusift_extract_options *options)
{
    cusift_api_guard(__func__, [&]()
    {
        RunPipeline(image1, image2, keypoints1, keypoints2, options, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    });
}

void cusift_align(const cusift_image *image1, const cusift_image *image2, SiftData *keypoints1, SiftData *keypoints2, const cusift_extract_options *extract_options, const cusift_homography_options *homography_options, float *homography, int *inliers)
{
    cusift_api_guard(__func__, [&]()
    {
        RequireNonNull(homography, "homography");
        RunPipeline(image1, image2, keypoints1, keypoints2, extract_options, homography_options, homography, inliers, nullptr, nullptr, nullptr, nullptr);
    });
}

void cusift_align_and_warp(const cusift_image *image1, const cusift_image *image2, SiftData *keypoints1, SiftData *keypoints2, const cusift_extract_options *extract_options, const cusift_homography_options *homography_options, float *homography, int *inliers, cusift_image *warped1, cusift_image *warped2)
{
    cusift_api_guard(__func__, [&]()
    {
        RequireNonNull(homography, "homography");
        RequireNonNull(warped1, "warped1");
        RunPipeline(image1, image2, keypoints1, keypoints2, extract_options, homography_options, homography, inliers, warped1, warped2, nullptr, nullptr);
    });
}

void cusift_align_and_warp_to_device(const cusift_image *image1, const cusift_image *image2, SiftData *keypoints1, SiftData *keypoints2, const cusift_extract_options *extract_options, const cusift_homography_options *homography_options, float *homography, int *inliers, cusift_device_image *warped1, cusift_device_image *warped2)
{
    cusift_api_guard(__func__, [&]()
    {
        RequireNonNull(homography, "homography");
        RequireNonNull(warped1, "warped1");
        RunPipeline(image1, image2, keypoints1, keypoints2, extract_options, homography_options, homography, inliers, nullptr, nullptr, warped1, warped2);
    });
}


// ── VRAM estimation helpers ─────────────────────────────────────────────────

static size_t estimatePyramidBytes(int width, int height, int numOctaves)
{
    const int nd = 5 + 3; // NUM_SCALES + 3
    int w = width;
    int h = height;
    int p = p_iAlignUp(w, 128);
    long long size    = (long long)h * p;       // image buffers
    long long sizeTmp = (long long)nd * h * p;  // laplace buffers
    for (int i = 0; i < numOctaves; i++)
    {
        w /= 2;
        h /= 2;
        p = p_iAlignUp(w, 128);
        size    += (long long)h * p;
        sizeTmp += (long long)nd * h * p;
    }
    size += sizeTmp;
    // Actual allocation: cudaMallocPitch with width=4096, height=(size+4095)/4096
    long long rows = (size + 4095) / 4096;
    return (size_t)(rows * 4096 * sizeof(float));
}

static size_t estimateCudaImageBytes(int width, int height)
{
    // CudaImage_Allocate uses cudaMallocPitch; approximate pitch as iAlignUp(width, 128)
    int pitch = p_iAlignUp(width, 128);
    return (size_t)pitch * height * sizeof(float);
}

static size_t estimateSiftDataDeviceBytes(int maxKeypoints)
{
    return (size_t)sizeof(SiftPoint) * maxKeypoints;
}

static size_t estimateContextBytes()
{
    const size_t counterBytes   = (8 * 2 + 1) * sizeof(unsigned int);
    const size_t laplaceBytes   = 8 * 12 * 16 * sizeof(float);
    const size_t scaleDownBytes = 5 * sizeof(float);
    const size_t lowPassBytes   = (2 * 4 + 1) * sizeof(float); // LOWPASS_R = 4
    return counterBytes + laplaceBytes + scaleDownBytes + lowPassBytes;
}

static int clampOctaves(int requested, int width, int height)
{
    int minDim = std::min(width, height);
    int maxOctaves = (int)std::floor(std::log2((double)minDim)) - 3;
    int octaves = std::min(requested, maxOctaves);
    octaves = std::min(octaves, 7);
    octaves = std::max(octaves, 3);
    return octaves;
}

size_t cusift_memory_extract(int image_width, int image_height, const cusift_extract_options *options)
{
    if (!options)
        return 0;
    int octaves = clampOctaves(options->num_octaves, image_width, image_height);

    size_t siftBytes    = estimateSiftDataDeviceBytes(options->max_keypoints);
    // thrust::sort in ExtractSift allocates temporary storage of roughly one
    // extra copy of the keypoint array while it runs.
    size_t sortTmpBytes = siftBytes;
    size_t imageBytes   = estimateCudaImageBytes(image_width, image_height);
    size_t pyramidBytes = estimatePyramidBytes(image_width, image_height, octaves);
    size_t contextBytes = estimateContextBytes();

    return siftBytes + sortTmpBytes + imageBytes + pyramidBytes + contextBytes;
}

size_t cusift_memory_match(int max_keypoints1, int max_keypoints2)
{
    return estimateSiftDataDeviceBytes(max_keypoints1)
         + estimateSiftDataDeviceBytes(max_keypoints2);
}

size_t cusift_memory_homography(int max_keypoints, const cusift_homography_options *options)
{
    if (!options)
        return 0;
    // Existing SiftData that must be resident
    size_t siftBytes = estimateSiftDataDeviceBytes(max_keypoints);

    // Temporary RANSAC buffers (mirroring matching.cu)
    int numPtsUp = ((max_keypoints + 15) / 16) * 16;
    int numLoops = ((options->ransac_iterations + 15) / 16) * 16;

    size_t coordBytes    = 4 * sizeof(float) * numPtsUp;

    size_t randPtsBytes, modelBytes;
    if (options->model == CUSIFT_MODEL_SIMILARITY) {
        randPtsBytes = 2 * sizeof(int)   * numLoops;
        modelBytes   = 4 * sizeof(float) * numLoops;
    } else {
        randPtsBytes = 4 * sizeof(int)   * numLoops;
        modelBytes   = 8 * sizeof(float) * numLoops;
    }

    return siftBytes + coordBytes + randPtsBytes + modelBytes;
}

size_t cusift_memory_warp(int image_width1, int image_height1, int image_width2, int image_height2)
{
    // Source images on device (pitch-aligned)
    size_t src1 = estimateCudaImageBytes(image_width1, image_height1);
    size_t src2 = estimateCudaImageBytes(image_width2, image_height2);

    // Worst-case output canvas: 2x the larger dimension in each axis
    int outW = 2 * std::max(image_width1, image_width2);
    int outH = 2 * std::max(image_height1, image_height2);
    size_t out1 = estimateCudaImageBytes(outW, outH);
    size_t out2 = out1; // same canvas size

    return src1 + src2 + out1 + out2;
}

size_t cusift_memory_pipeline(int image_width1, int image_height1,
                                int image_width2, int image_height2,
                                const cusift_extract_options *extract_options,
                                const cusift_homography_options *homography_options)
{
    if (!extract_options || !homography_options)
        return 0;
    // SiftData for both images persists across all stages
    size_t siftBoth = estimateSiftDataDeviceBytes(extract_options->max_keypoints) * 2;

    // Peak during extraction (image 1 or 2, whichever is larger)
    int maxW = std::max(image_width1, image_width2);
    int maxH = std::max(image_height1, image_height2);
    int octaves = clampOctaves(extract_options->num_octaves, maxW, maxH);

    size_t extractPeak = siftBoth
                       + estimateSiftDataDeviceBytes(extract_options->max_keypoints) // thrust::sort temporaries
                       + estimateCudaImageBytes(maxW, maxH)
                       + estimatePyramidBytes(maxW, maxH, octaves)
                       + estimateContextBytes();

    // Peak during homography (SiftData + RANSAC temporaries)
    size_t homographyPeak = siftBoth
                          + cusift_memory_homography(extract_options->max_keypoints, homography_options)
                          - estimateSiftDataDeviceBytes(extract_options->max_keypoints); // avoid double-counting the one SiftData

    // Peak during warping (SiftData already freed in convenience funcs, but be conservative)
    size_t warpPeak = siftBoth
                    + cusift_memory_warp(image_width1, image_height1, image_width2, image_height2);

    return std::max({extractPeak, homographyPeak, warpPeak});
}