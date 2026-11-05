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

#define ERROR(msg) cusift_throw_error(__FILE__, __LINE__, msg)

static void cusift_throw_error(const char *file, int line, const char *msg)
{
    throw CusiftError(file, line, msg);
}

static int p_iAlignUp(int a, int b) { return (a % b != 0) ? (a - a % b + b) : a; }

// ── Argument validation ─────────────────────────────────
// Every public entry point validates its arguments before touching the GPU so
// that a bad call is reported through CusiftHadError() instead of crashing or
// silently corrupting memory.

static void RequireNonNull(const void *p, const char *what)
{
    if (!p)
        throw CusiftError(__FILE__, __LINE__, std::string("Required argument is NULL: ") + what);
}

// Largest max_keypoints_ whose byte count fits a 32-bit int; InitSiftData
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

static void ValidateImage(const Image_t *img, const char *what)
{
    RequireNonNull(img, what);
    if (!img->host_img_)
        throw CusiftError(__FILE__, __LINE__, std::string(what) + ": host_img_ is NULL");
    if (img->width_ <= 0 || img->height_ <= 0)
        throw CusiftError(__FILE__, __LINE__, std::string(what) + ": width and height must be positive");
    if (img->width_ < kMinImageDim || img->height_ < kMinImageDim)
        throw CusiftError(__FILE__, __LINE__, std::string(what) + ": image must be at least " + std::to_string(kMinImageDim) + " pixels in each dimension");
    if (img->width_ > kMaxImageDim || img->height_ > kMaxImageDim)
        throw CusiftError(__FILE__, __LINE__, std::string(what) + ": image dimensions must not exceed " + std::to_string(kMaxImageDim) + " pixels");
    // Row offsets inside the kernels are 32-bit; one pitched plane must fit.
    if ((size_t)p_iAlignUp(img->width_, 128) * (size_t)img->height_ > (size_t)INT_MAX)
        throw CusiftError(__FILE__, __LINE__, std::string(what) + ": image too large (pitch * height must fit in a 32-bit int)");
    // A cudaMalloc pointer is not CPU-readable: the CPU warp path would fault on
    // it and the host-to-device copies would fail.  Classify it and reject it
    // before anything dereferences it.  Pinned and managed memory are host
    // readable and pass.  Older runtimes report plain pageable memory through
    // the error return instead of cudaMemoryTypeUnregistered; that is host
    // memory, so clear the flag and accept it.
    cudaPointerAttributes attr;
    cudaError_t perr = cudaPointerGetAttributes(&attr, img->host_img_);
    if (perr == cudaSuccess)
    {
        if (attr.type == cudaMemoryTypeDevice)
            throw CusiftError(__FILE__, __LINE__, std::string(what) + ": host_img_ points to device memory; pass host, pinned or managed memory");
    }
    else
    {
        (void)cudaGetLastError();
    }
}

static void ValidateExtractOptions(const ExtractSiftOptions_t *o)
{
    RequireNonNull(o, "extract options");
    if (o->max_keypoints_ <= 0 || o->max_keypoints_ > kMaxKeypointsLimit)
        throw CusiftError(__FILE__, __LINE__, "max_keypoints_ must be in [1, " + std::to_string(kMaxKeypointsLimit) + "]");
}

static void ValidateHomographyOptions(const FindHomographyOptions_t *o)
{
    RequireNonNull(o, "homography options");
    if (o->num_loops_ <= 0 || o->num_loops_ > kMaxRansacLoops)
        throw CusiftError(__FILE__, __LINE__, "num_loops_ must be in [1, " + std::to_string(kMaxRansacLoops) + "]");
    if (o->improve_num_loops_ < 0)
        throw CusiftError(__FILE__, __LINE__, "improve_num_loops_ must be >= 0");
}

// Output SiftData contract (see cusift.h): the caller zero-initialises the
// struct once; after that it may hold buffers from a previous call, which are
// released here so re-using one SiftData across frames does not leak.  Leaves
// the struct empty (numPts = maxPts = 0, NULL buffers) so DeleteSiftData is
// safe even if this call fails before InitSiftData runs.
static void PrepareSiftData(SiftData *sd, const char *what)
{
    RequireNonNull(sd, what);
    FreeSiftData(sd);
}

// Dispatch to either full homography or similarity RANSAC based on model_type_
static void FindGeometricModel(SiftData *data, float *homography, int *numMatches,
                                const FindHomographyOptions_t *opts)
{
    if (opts->model_type_ == CUSIFT_MODEL_SIMILARITY)
    {
        FindSimilarity_private(
            data, homography, numMatches,
            opts->num_loops_, opts->min_score_, opts->max_ambiguity_,
            opts->thresh_, opts->seed_);
    }
    else
    {
        FindHomography_private(
            data, homography, numMatches,
            opts->num_loops_, opts->min_score_, opts->max_ambiguity_,
            opts->thresh_, opts->seed_);
    }
}

static bool Invert3x3(const float *M, float *out);
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

    void clear()
    {
        had_error  = false;
        line       = 0;
        cuda_error = cudaSuccess;
        file.clear();
        message.clear();
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
    bool                    api_configured = false; // CusiftSetErrorLog() was called: the environment is ignored
    bool                    env_checked    = false;
    std::string             log_path;               // empty = file logging off
    size_t                  max_bytes      = 0;
    CusiftErrorCallback_t   callback       = nullptr;
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

std::string format_record(const CusiftErrorRecord_t &r)
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
        CusiftErrorRecord_t rec{};
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

        CusiftErrorCallback_t cb = nullptr;
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

// Wrap an API body: clear error state, run fn(), catch and store any
// exception, and report it to the sinks.  `api` is the public function's name.
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
        s_error.had_error  = true;
        s_error.file       = e.file;
        s_error.line       = e.line;
        s_error.message    = e.what();
        s_error.cuda_error = e.cuda_error;
        cusift_report(0, api, e.file.c_str(), e.line, e.cuda_error, e.what());
    }
    catch (const std::exception &e)
    {
        // STL/host exceptions (e.g. std::bad_alloc) lack source location.
        s_error.had_error = true;
        s_error.line      = 0;
        s_error.file.clear();
        s_error.message   = e.what();
        cusift_report(0, api, "", 0, cudaSuccess, e.what());
    }
    catch (...)
    {
        s_error.had_error = true;
        s_error.line      = 0;
        s_error.file.clear();
        s_error.message   = "Unknown exception";
        cusift_report(0, api, "", 0, cudaSuccess, "Unknown exception");
    }
}

void CusiftGetLastErrorString(int *line_number,
                              char *filename, size_t filename_size,
                              char *error_message, size_t error_message_size)
{
    if (line_number)
        *line_number = s_error.line;
    if (filename && filename_size > 0)
    {
        const std::string &src = s_error.file;
        size_t n = src.size();
        if (n >= filename_size) n = filename_size - 1;
        memcpy(filename, src.data(), n);
        filename[n] = '\0';
    }
    if (error_message && error_message_size > 0)
    {
        const std::string &src = s_error.message;
        size_t n = src.size();
        if (n >= error_message_size) n = error_message_size - 1;
        memcpy(error_message, src.data(), n);
        error_message[n] = '\0';
    }
}

int CusiftHadError()
{
    return s_error.had_error ? 1 : 0;
}

int CusiftGetLastCudaError()
{
    return static_cast<int>(s_error.cuda_error);
}

void InitializeCudaSift()
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

void CusiftResetDevice(void)
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
void CusiftSetErrorLog(const char *path, size_t max_bytes)
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

void CusiftSetErrorCallback(CusiftErrorCallback_t callback, void *user_data)
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

unsigned long long CusiftErrorLogDroppedCount(void)
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

void ExtractSiftFromImage(const Image_t *image, SiftData *sift_data, const ExtractSiftOptions_t *options)
{
    cusift_api_guard(__func__, [&]()
    {
        ValidateImage(image, "image");
        ValidateExtractOptions(options);
        PrepareSiftData(sift_data, "sift_data");

        CudaImageGuard cuda_image;

        InitSiftData(sift_data, options->max_keypoints_, true, true);

        CudaImage_Allocate(
            cuda_image.get(),
            image->width_,
            image->height_,
            p_iAlignUp(image->width_, 128),
            false,
            nullptr,
            image->host_img_);

        CudaImage_Download(cuda_image.get());
        //CudaImage_Normalize(cuda_image.get());

        // Get the smallest dimension of the image to determine the maximum number of octaves
        int minDim = std::min(image->width_, image->height_);
        int maxOctaves = static_cast<int>(std::floor(std::log2(minDim))) - 3;
        int octaves = std::min(options->num_octaves_, maxOctaves);
        // Limit num octaves to 7, quietly
        octaves = std::min(octaves, 7);
        octaves = std::max(octaves, 3); // Ensure at least 3 octaves

        SiftTempMemoryGuard tempMemory(AllocSiftTempMemory(image->width_, image->height_, octaves));

        ExtractSift(
            sift_data,
            cuda_image.get(),
            octaves,
            options->init_blur_,
            options->thresh_,
            options->lowest_scale_,
            options->highest_scale_,
            options->edge_thresh_,
            tempMemory.get());
        if (options->scale_suppression_radius_ > 0.0f)
            SuppressEmbeddedPoints(sift_data, options->scale_suppression_radius_);
    });
}

void MatchSiftData(SiftData *data1, SiftData *data2)
{
    cusift_api_guard(__func__, [&]()
    {
        RequireNonNull(data1, "data1");
        RequireNonNull(data2, "data2");
        MatchSiftData_private(data1, data2);
    });
}

void FindHomography(SiftData *data, float *homography, int *num_matches, const FindHomographyOptions_t *options)
{
    cusift_api_guard(__func__, [&]()
    {
        RequireNonNull(data, "data");
        RequireNonNull(homography, "homography");
        RequireNonNull(num_matches, "num_matches");
        ValidateHomographyOptions(options);

        FindGeometricModel(
            data,
            homography,
            num_matches,
            options);
        
        // If any homography values are NaN or +/-inf, we should throw an error rather than proceeding to the warp which will produce garbage output and may cause CUDA errors.
        for (int i = 0; i < 8; i++)
        {
            if (!std::isfinite(homography[i]))
            {
                ERROR("Homography contains non-finite values");
            }
        }

        ImproveHomography(
            data,
            homography,
            options->improve_num_loops_,
            options->improve_min_score_,
            options->improve_max_ambiguity_,
            options->improve_thresh_);
    });
}

// Resource-cleanup functions (DeleteSiftData / FreeImage / FreeImage_GPU) do
// NOT pass through cusift_api_guard.  Cleanup is destructor-safe (uses
// cudaFree / free directly, never throws) and we deliberately leave the
// thread-local error state untouched so the typical "do work; then free; then
// check error" pattern still surfaces the original failure.
void DeleteSiftData(SiftData *sift_data)
{
    if (sift_data)
        FreeSiftData(sift_data);
}

void FreeImage(Image_t *image)
{
    if (image && image->host_img_)
    {
        free(image->host_img_);
        image->host_img_ = nullptr;
        image->width_ = 0;
        image->height_ = 0;
    }
}

void FreeImage_GPU(ImageStrided_t *image)
{
    if (image && image->strided_img_)
    {
        cudaError_t e = cudaFree(image->strided_img_);
        if (e != cudaSuccess)
        {
            (void)cudaGetLastError();
            char buf[160];
            std::snprintf(buf, sizeof(buf), "cudaFree of the warped image failed: %s", cudaGetErrorString(e));
            cusift_report(1, "FreeImage_GPU", "", 0, e, buf);
        }
        image->strided_img_ = nullptr;
        image->width_ = 0;
        image->height_ = 0;
        image->stride_ = 0;
    }
}

void SaveSiftData(const char *filename, const SiftData *sift_data)
{
    cusift_api_guard(__func__, [&]()
    {
        RequireNonNull(filename, "filename");
        if (!sift_data || !sift_data->h_data || sift_data->numPts <= 0)
            ERROR("SaveSiftData: no data to save");

        FILE *f = fopen(filename, "w");
        if (!f)
            ERROR("SaveSiftData: could not open file for writing");

        fprintf(f, "{\n");
    fprintf(f, "  \"num_keypoints\": %d,\n", sift_data->numPts);
    fprintf(f, "  \"keypoints\": [\n");

    for (int i = 0; i < sift_data->numPts; i++)
    {
        const SiftPoint *pt = &sift_data->h_data[i];
        fprintf(f, "    {\n");
        fprintf(f, "      \"x\": %.6f,\n", pt->xpos);
        fprintf(f, "      \"y\": %.6f,\n", pt->ypos);
        fprintf(f, "      \"scale\": %.6f,\n", pt->scale);
        fprintf(f, "      \"sharpness\": %.6f,\n", pt->sharpness);
        fprintf(f, "      \"edgeness\": %.6f,\n", pt->edgeness);
        fprintf(f, "      \"orientation\": %.6f,\n", pt->orientation);
        fprintf(f, "      \"score\": %.6f,\n", pt->score);
        fprintf(f, "      \"ambiguity\": %.6f,\n", pt->ambiguity);
        fprintf(f, "      \"match\": %d,\n", pt->match);
        fprintf(f, "      \"match_x\": %.6f,\n", pt->match_xpos);
        fprintf(f, "      \"match_y\": %.6f,\n", pt->match_ypos);
        fprintf(f, "      \"match_error\": %.6f,\n", pt->match_error);
        fprintf(f, "      \"descriptor\": [");
        for (int j = 0; j < 128; j++)
        {
            fprintf(f, "%.6f", pt->data[j]);
            if (j < 127)
                fprintf(f, ", ");
        }
        fprintf(f, "]\n");
        fprintf(f, "    }%s\n", (i < sift_data->numPts - 1) ? "," : "");
    }

    fprintf(f, "  ]\n");
    fprintf(f, "}\n");
    fclose(f);
    });
}

void ExtractAndMatchSift(const Image_t *image1, const Image_t *image2, SiftData *sift_data1, SiftData *sift_data2, const ExtractSiftOptions_t *extract_options)
{
    cusift_api_guard(__func__, [&]()
    {
        ValidateImage(image1, "image1");
        ValidateImage(image2, "image2");
        ValidateExtractOptions(extract_options);
        PrepareSiftData(sift_data1, "sift_data1");
        PrepareSiftData(sift_data2, "sift_data2");

        int maxW = std::max(image1->width_, image2->width_);
        int maxH = std::max(image1->height_, image2->height_);

        CudaImageGuard cuda_image1;
        CudaImageGuard cuda_image2;

        // Clamp octaves to what the smallest image dimension supports
        int minDim = std::min({image1->width_, image1->height_, image2->width_, image2->height_});
        int maxOctaves = static_cast<int>(std::floor(std::log2(minDim))) - 3;
        int octaves = std::min(extract_options->num_octaves_, maxOctaves);
        // Limit num octaves to 7, quietly
        octaves = std::min(octaves, 7);
        octaves = std::max(octaves, 3); // Ensure at least 3 octaves

        // Only allocate a single temporary buffer for both images since they won't be processed at the same time
        SiftTempMemoryGuard tempMemory(AllocSiftTempMemory(maxW, maxH, octaves));

        // Extract from image 1
        InitSiftData(sift_data1, extract_options->max_keypoints_, true, true);
        CudaImage_Allocate(cuda_image1.get(), image1->width_, image1->height_,
                           p_iAlignUp(image1->width_, 128), false, nullptr, image1->host_img_);
        CudaImage_Download(cuda_image1.get());
        //CudaImage_Normalize(cuda_image1.get());
        ExtractSift(sift_data1, cuda_image1.get(), octaves,
                    extract_options->init_blur_, extract_options->thresh_,
                    extract_options->lowest_scale_, extract_options->highest_scale_, extract_options->edge_thresh_,
                    tempMemory.get());
        if (extract_options->scale_suppression_radius_ > 0.0f)
            SuppressEmbeddedPoints(sift_data1, extract_options->scale_suppression_radius_);

        // Extract from image 2
        InitSiftData(sift_data2, extract_options->max_keypoints_, true, true);
        CudaImage_Allocate(cuda_image2.get(), image2->width_, image2->height_,
                           p_iAlignUp(image2->width_, 128), false, nullptr, image2->host_img_);
        CudaImage_Download(cuda_image2.get());
        //CudaImage_Normalize(cuda_image2.get());
        ExtractSift(sift_data2, cuda_image2.get(), octaves,
                    extract_options->init_blur_, extract_options->thresh_,
                    extract_options->lowest_scale_, extract_options->highest_scale_, extract_options->edge_thresh_,
                    tempMemory.get());
        if (extract_options->scale_suppression_radius_ > 0.0f)
            SuppressEmbeddedPoints(sift_data2, extract_options->scale_suppression_radius_);

        // Match
        MatchSiftData_private(sift_data1, sift_data2);
    });
}

void ExtractAndMatchAndFindHomography(const Image_t *image1, const Image_t *image2, SiftData *sift_data1, SiftData *sift_data2, float *homography, int *num_matches, const ExtractSiftOptions_t *extract_options, const FindHomographyOptions_t *homography_options)
{
    cusift_api_guard(__func__, [&]()
    {
        ValidateImage(image1, "image1");
        ValidateImage(image2, "image2");
        ValidateExtractOptions(extract_options);
        PrepareSiftData(sift_data1, "sift_data1");
        PrepareSiftData(sift_data2, "sift_data2");
        RequireNonNull(homography, "homography");
        RequireNonNull(num_matches, "num_matches");
        ValidateHomographyOptions(homography_options);

        int maxW = std::max(image1->width_, image2->width_);
        int maxH = std::max(image1->height_, image2->height_);

        CudaImageGuard cuda_image1;
        CudaImageGuard cuda_image2;

        // Clamp octaves to what the smallest image dimension supports
        int minDim = std::min({image1->width_, image1->height_, image2->width_, image2->height_});
        int maxOctaves = static_cast<int>(std::floor(std::log2(minDim))) - 3;
        int octaves = std::min(extract_options->num_octaves_, maxOctaves);
        // Limit num octaves to 7, quietly
        octaves = std::min(octaves, 7);
        octaves = std::max(octaves, 3); // Ensure at least 3 octaves

        // Only allocate a single temporary buffer for both images since they won't be processed at the same time
        SiftTempMemoryGuard tempMemory(AllocSiftTempMemory(maxW, maxH, octaves));

        // Extract from image 1
        InitSiftData(sift_data1, extract_options->max_keypoints_, true, true);
        CudaImage_Allocate(cuda_image1.get(), image1->width_, image1->height_,
                           p_iAlignUp(image1->width_, 128), false, nullptr, image1->host_img_);
        CudaImage_Download(cuda_image1.get());
        //CudaImage_Normalize(cuda_image1.get());
        ExtractSift(sift_data1, cuda_image1.get(), octaves,
                    extract_options->init_blur_, extract_options->thresh_,
                    extract_options->lowest_scale_, extract_options->highest_scale_, extract_options->edge_thresh_,
                    tempMemory.get());
        if (extract_options->scale_suppression_radius_ > 0.0f)
            SuppressEmbeddedPoints(sift_data1, extract_options->scale_suppression_radius_);

        // Extract from image 2
        InitSiftData(sift_data2, extract_options->max_keypoints_, true, true);
        CudaImage_Allocate(cuda_image2.get(), image2->width_, image2->height_,
                           p_iAlignUp(image2->width_, 128), false, nullptr, image2->host_img_);
        CudaImage_Download(cuda_image2.get());
        //CudaImage_Normalize(cuda_image2.get());
        ExtractSift(sift_data2, cuda_image2.get(), octaves,
                    extract_options->init_blur_, extract_options->thresh_,
                    extract_options->lowest_scale_, extract_options->highest_scale_, extract_options->edge_thresh_,
                    tempMemory.get());
        if (extract_options->scale_suppression_radius_ > 0.0f)
            SuppressEmbeddedPoints(sift_data2, extract_options->scale_suppression_radius_);

        // Match
        MatchSiftData_private(sift_data1, sift_data2);

        FindGeometricModel(
            sift_data1, homography, num_matches,
            homography_options);
        
        // If any homography values are NaN or +/-inf, we should throw an error rather than proceeding to the warp which will produce garbage output and may cause CUDA errors.
        for (int i = 0; i < 8; i++)
        {
            if (!std::isfinite(homography[i]))
            {
                ERROR("Homography contains non-finite values");
            }
        }

        ImproveHomography(
            sift_data1, homography,
            homography_options->improve_num_loops_,
            homography_options->improve_min_score_,
            homography_options->improve_max_ambiguity_,
            homography_options->improve_thresh_);
    });
}

// ── Helper: Frobenius-like distance between 2x2 top-left sub-matrix and I ──
static float EyeDiff2x2(const float *H)
{
    // H is row-major 3x3. Top-left 2x2: H[0],H[1],H[3],H[4]
    float d00 = H[0] - 1.0f;
    float d01 = H[1];
    float d10 = H[3];
    float d11 = H[4] - 1.0f;
    return d00 * d00 + d01 * d01 + d10 * d10 + d11 * d11;
}

void ExtractAndMatchAndFindHomography_Multi(const Image_t *image1, const Image_t *image2, SiftData *sift_data1, SiftData *sift_data2, float *homography, int *num_matches, const ExtractSiftOptions_t *extract_options, const FindHomographyOptions_t *homography_options, int num_homography_attempts, int homography_goal)
{
    cusift_api_guard(__func__, [&]()
    {
        ValidateImage(image1, "image1");
        ValidateImage(image2, "image2");
        ValidateExtractOptions(extract_options);
        PrepareSiftData(sift_data1, "sift_data1");
        PrepareSiftData(sift_data2, "sift_data2");
        RequireNonNull(homography, "homography");
        RequireNonNull(num_matches, "num_matches");
        ValidateHomographyOptions(homography_options);

        int maxW = std::max(image1->width_, image2->width_);
        int maxH = std::max(image1->height_, image2->height_);

        CudaImageGuard cuda_image1;
        CudaImageGuard cuda_image2;

        // Clamp octaves to what the smallest image dimension supports
        int minDim = std::min({image1->width_, image1->height_, image2->width_, image2->height_});
        int maxOctaves = static_cast<int>(std::floor(std::log2(minDim))) - 3;
        int octaves = std::min(extract_options->num_octaves_, maxOctaves);
        // Limit num octaves to 7, quietly
        octaves = std::min(octaves, 7);
        octaves = std::max(octaves, 3); // Ensure at least 3 octaves

        // Only allocate a single temporary buffer for both images since they won't be processed at the same time
        SiftTempMemoryGuard tempMemory(AllocSiftTempMemory(maxW, maxH, octaves));

        // Extract from image 1
        InitSiftData(sift_data1, extract_options->max_keypoints_, true, true);
        CudaImage_Allocate(cuda_image1.get(), image1->width_, image1->height_,
                           p_iAlignUp(image1->width_, 128), false, nullptr, image1->host_img_);
        CudaImage_Download(cuda_image1.get());
        ExtractSift(sift_data1, cuda_image1.get(), octaves,
                    extract_options->init_blur_, extract_options->thresh_,
                    extract_options->lowest_scale_, extract_options->highest_scale_, extract_options->edge_thresh_,
                    tempMemory.get());
        if (extract_options->scale_suppression_radius_ > 0.0f)
            SuppressEmbeddedPoints(sift_data1, extract_options->scale_suppression_radius_);

        // Extract from image 2
        InitSiftData(sift_data2, extract_options->max_keypoints_, true, true);
        CudaImage_Allocate(cuda_image2.get(), image2->width_, image2->height_,
                           p_iAlignUp(image2->width_, 128), false, nullptr, image2->host_img_);
        CudaImage_Download(cuda_image2.get());
        ExtractSift(sift_data2, cuda_image2.get(), octaves,
                    extract_options->init_blur_, extract_options->thresh_,
                    extract_options->lowest_scale_, extract_options->highest_scale_, extract_options->edge_thresh_,
                    tempMemory.get());
        if (extract_options->scale_suppression_radius_ > 0.0f)
            SuppressEmbeddedPoints(sift_data2, extract_options->scale_suppression_radius_);

        // Match
        MatchSiftData_private(sift_data1, sift_data2);

        // Clamp attempts to at least 1
        int attempts = std::max(num_homography_attempts, 1);

        // Build a mutable copy of the homography options so we can vary the seed
        FindHomographyOptions_t opts = *homography_options;

        // Seed source: if the caller specified seed=0 (non-deterministic), use
        // std::random_device; otherwise increment deterministically from the
        // given seed each attempt.
        std::random_device rd;

        float bestH[9] = {};
        int   bestInliers = -1;
        float bestScore   = FLT_MAX; // lower is better for EyeDiff

        for (int attempt = 0; attempt < attempts; attempt++)
        {
            float candidateH[9];
            int   candidateInliers = 0;

            // Vary seed per attempt
            if (homography_options->seed_ == 0)
                opts.seed_ = rd();
            else
                opts.seed_ = homography_options->seed_ + (unsigned int)attempt;

            FindGeometricModel(
                sift_data1, candidateH, &candidateInliers,
                &opts);

            // Skip non-finite results
            bool finite = true;
            for (int i = 0; i < 9; i++)
            {
                if (!std::isfinite(candidateH[i]))
                {
                    finite = false;
                    break;
                }
            }
            if (!finite)
                continue;

            ImproveHomography(
                sift_data1, candidateH,
                opts.improve_num_loops_,
                opts.improve_min_score_,
                opts.improve_max_ambiguity_,
                opts.improve_thresh_);

            // Evaluate candidate
            bool isBetter = false;
            if (homography_goal == CUSIFT_HOMOGRAPHY_GOAL_MIN_EYE_DIFF)
            {
                float score = EyeDiff2x2(candidateH);
                if (bestInliers < 0 || score < bestScore)
                {
                    bestScore = score;
                    isBetter = true;
                }
            }
            else // CUSIFT_HOMOGRAPHY_GOAL_MAX_INLIERS (default)
            {
                if (candidateInliers > bestInliers)
                    isBetter = true;
            }

            if (isBetter)
            {
                std::memcpy(bestH, candidateH, sizeof(bestH));
                bestInliers = candidateInliers;
            }
        }

        if (bestInliers < 0)
        {
            ERROR("Homography contains non-finite values in all attempts");
        }

        std::memcpy(homography, bestH, 9 * sizeof(float));
        *num_matches = bestInliers;
    });
}

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
// Every warp entry point and the public ComputeWarpGeometry() go through
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
static void ComputeCanvasGeometry(const float *homography, int w1, int h1, int w2, int h2, WarpGeometry_t *g)
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
void WarpImages(const Image_t *image1, const Image_t *image2, const float *homography,
                Image_t *warped_image1, Image_t *warped_image2, bool useGPU)
{
    cusift_api_guard(__func__, [&]()
    {
        ValidateImage(image1, "image1");
        ValidateImage(image2, "image2");
        RequireNonNull(homography, "homography");
        RequireNonNull(warped_image1, "warped_image1");
        RequireNonNull(warped_image2, "warped_image2");

        int w1 = image1->width_, h1 = image1->height_;
        int w2 = image2->width_, h2 = image2->height_;

        // ── Step 1: Compute inv(homography) and project image2 corners ──
        WarpGeometry_t geom;
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

            safeCall(cudaMemcpy2D(d_src1Guard.get(), src1Pitch, image1->host_img_, w1 * sizeof(float), w1 * sizeof(float), h1, cudaMemcpyHostToDevice));
            safeCall(cudaMemcpy2D(d_src2Guard.get(), src2Pitch, image2->host_img_, w2 * sizeof(float), w2 * sizeof(float), h2, cudaMemcpyHostToDevice));

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
                    out1[y * outW + x] = SampleBilinear(image1->host_img_, w1, h1, u, v);

                    // Warped image2: apply homography to get source coords
                    float z = H20 * u + H21 * v + H22;
                    float su = (H00 * u + H01 * v + H02) / z;
                    float sv = (H10 * u + H11 * v + H12) / z;
                    out2[y * outW + x] = SampleBilinear(image2->host_img_, w2, h2, su, sv);
                }
            }
        }

        // ── Step 4: Fill output structs, release ownership ──
        warped_image1->host_img_ = out1Guard.release();
        warped_image1->width_ = outW;
        warped_image1->height_ = outH;

        warped_image2->host_img_ = out2Guard.release();
        warped_image2->width_ = outW;
        warped_image2->height_ = outH;
    }); // end cusift_api_guard for WarpImages
}

void WarpImages_GPU(const Image_t *image1, const Image_t *image2, const float *homography,
                    ImageStrided_t *warped_image1, ImageStrided_t *warped_image2)
{
    cusift_api_guard(__func__, [&]()
    {
        ValidateImage(image1, "image1");
        ValidateImage(image2, "image2");
        RequireNonNull(homography, "homography");
        RequireNonNull(warped_image1, "warped_image1");
        RequireNonNull(warped_image2, "warped_image2");

        int w1 = image1->width_, h1 = image1->height_;
        int w2 = image2->width_, h2 = image2->height_;

        // ── Step 1: Compute inv(homography) and project image2 corners ──
        WarpGeometry_t geom;
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

        safeCall(cudaMemcpy2D(d_src1Guard.get(), src1Pitch, image1->host_img_, w1 * sizeof(float), w1 * sizeof(float), h1, cudaMemcpyHostToDevice));
        safeCall(cudaMemcpy2D(d_src2Guard.get(), src2Pitch, image2->host_img_, w2 * sizeof(float), w2 * sizeof(float), h2, cudaMemcpyHostToDevice));

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
        warped_image1->strided_img_ = d_out1Guard.release();
        warped_image1->width_ = outW;
        warped_image1->height_ = outH;
        warped_image1->stride_ = out1Pitch;

        warped_image2->strided_img_ = d_out2Guard.release();
        warped_image2->width_ = outW;
        warped_image2->height_ = outH;
        warped_image2->stride_ = out2Pitch;

        // d_src1Guard, d_src2Guard freed automatically
    }); // end cusift_api_guard for WarpImages_GPU
}

void ComputeWarpGeometry(const float *homography, int image_width1, int image_height1, int image_width2, int image_height2, WarpGeometry_t *geometry)
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

void ExtractAndMatchAndFindHomographyAndWarp(const Image_t *image1, const Image_t *image2, SiftData *sift_data1, SiftData *sift_data2, float *homography, int *num_matches, const ExtractSiftOptions_t *extract_options, const FindHomographyOptions_t *homography_options, Image_t *warped_image1, Image_t *warped_image2)
{
    cusift_api_guard(__func__, [&]()
    {
        ValidateImage(image1, "image1");
        ValidateImage(image2, "image2");
        ValidateExtractOptions(extract_options);
        PrepareSiftData(sift_data1, "sift_data1");
        PrepareSiftData(sift_data2, "sift_data2");
        RequireNonNull(homography, "homography");
        RequireNonNull(num_matches, "num_matches");
        ValidateHomographyOptions(homography_options);
        RequireNonNull(warped_image1, "warped_image1");
        RequireNonNull(warped_image2, "warped_image2");

        int w1 = image1->width_, h1 = image1->height_;
        int w2 = image2->width_, h2 = image2->height_;
        int maxW = std::max(w1, w2);
        int maxH = std::max(h1, h2);

        // ── Extract SIFT features (images stay on device via CudaImageGuard) ──
        CudaImageGuard cuda_image1;
        CudaImageGuard cuda_image2;

        int minDim = std::min({w1, h1, w2, h2});
        int maxOctaves = static_cast<int>(std::floor(std::log2(minDim))) - 3;
        int octaves = std::min(extract_options->num_octaves_, maxOctaves);
        // Limit num octaves to 7, quietly
        octaves = std::min(octaves, 7);
        octaves = std::max(octaves, 3); // Ensure at least 3 octaves

        SiftTempMemoryGuard tempMemory(AllocSiftTempMemory(maxW, maxH, octaves));

        // Image 1
        InitSiftData(sift_data1, extract_options->max_keypoints_, true, true);
        CudaImage_Allocate(cuda_image1.get(), w1, h1,
                           p_iAlignUp(w1, 128), false, nullptr, image1->host_img_);
        CudaImage_Download(cuda_image1.get());
        ExtractSift(sift_data1, cuda_image1.get(), octaves,
                    extract_options->init_blur_, extract_options->thresh_,
                    extract_options->lowest_scale_, extract_options->highest_scale_, extract_options->edge_thresh_,
                    tempMemory.get());
        if (extract_options->scale_suppression_radius_ > 0.0f)
            SuppressEmbeddedPoints(sift_data1, extract_options->scale_suppression_radius_);

        // Image 2
        InitSiftData(sift_data2, extract_options->max_keypoints_, true, true);
        CudaImage_Allocate(cuda_image2.get(), w2, h2,
                           p_iAlignUp(w2, 128), false, nullptr, image2->host_img_);
        CudaImage_Download(cuda_image2.get());
        ExtractSift(sift_data2, cuda_image2.get(), octaves,
                    extract_options->init_blur_, extract_options->thresh_,
                    extract_options->lowest_scale_, extract_options->highest_scale_, extract_options->edge_thresh_,
                    tempMemory.get());
        if (extract_options->scale_suppression_radius_ > 0.0f)
            SuppressEmbeddedPoints(sift_data2, extract_options->scale_suppression_radius_);

        // ── Match ────────────────────────────────────────────────────────────
        MatchSiftData_private(sift_data1, sift_data2);

        // ── Find homography ─────────────────────────────────────────────────
        FindGeometricModel(
            sift_data1, homography, num_matches,
            homography_options);
        
        // If any homography values are NaN or +/-inf, we should throw an error rather than proceeding to the warp which will produce garbage output and may cause CUDA errors.
        for (int i = 0; i < 8; i++)
        {
            if (!std::isfinite(homography[i]))
            {
                ERROR("Homography contains non-finite values");
            }
        }

        ImproveHomography(
            sift_data1, homography,
            homography_options->improve_num_loops_,
            homography_options->improve_min_score_,
            homography_options->improve_max_ambiguity_,
            homography_options->improve_thresh_);

        // ── Compute canvas bounding box ─────────────────────────────────────
        WarpGeometry_t geom;
        ComputeCanvasGeometry(homography, w1, h1, w2, h2, &geom);
        const int outW = geom.canvas_width;
        const int outH = geom.canvas_height;
        const float originU = (float)geom.origin_u;
        const float originV = (float)geom.origin_v;

        // ── Warp on GPU — source data is already device-resident ────────────
        // Reuse the CudaImageGuard device pointers (no extra upload needed).
        float *d_src1 = cuda_image1.get()->d_data;
        float *d_src2 = cuda_image2.get()->d_data;
        int src1Pitch = cuda_image1.get()->pitch;
        int src2Pitch = cuda_image2.get()->pitch;

        DevicePtrGuard<float> d_out1Guard, d_out2Guard;

        size_t out1Stride = 0;
        size_t out2Stride = 0;

        safeCall(cudaMallocPitch(&d_out1Guard.getRef(), &out1Stride, outW * sizeof(float), outH));
        safeCall(cudaMallocPitch(&d_out2Guard.getRef(), &out2Stride, outW * sizeof(float), outH));

        float H00 = homography[0], H01 = homography[1], H02 = homography[2];
        float H10 = homography[3], H11 = homography[4], H12 = homography[5];
        float H20 = homography[6], H21 = homography[7], H22 = homography[8];

        dim3 threads(16, 16);
        dim3 blocks((outW + threads.x - 1) / threads.x,
                    (outH + threads.y - 1) / threads.y);

        // Warp both images in a single kernel launch
        WarpDualKernel<<<blocks, threads>>>(
            d_src1, d_out1Guard.get(), w1, h1, src1Pitch,
            d_src2, d_out2Guard.get(), w2, h2, src2Pitch,
            outW, outH, static_cast<int>(out1Stride / sizeof(float)),
            static_cast<int>(out2Stride / sizeof(float)), originU, originV,
            H00, H01, H02,
            H10, H11, H12,
            H20, H21, H22);
        checkMsg("WarpDualKernel() execution failed\n");

        safeCall(cudaDeviceSynchronize());

        // ── Copy results to host and fill output structs ────────────────────
        size_t nPixels = (size_t)outW * outH;
        HostPtrGuard<float> out1Guard((float *)malloc(sizeof(float) * nPixels));
        HostPtrGuard<float> out2Guard((float *)malloc(sizeof(float) * nPixels));
        if (!out1Guard.get() || !out2Guard.get())
        {
            ERROR("Warping: return image host allocation failed");
        }

        safeCall(cudaMemcpy2D(out1Guard.get(), outW * sizeof(float), d_out1Guard.get(), out1Stride, outW * sizeof(float), outH, cudaMemcpyDeviceToHost));
        safeCall(cudaMemcpy2D(out2Guard.get(), outW * sizeof(float), d_out2Guard.get(), out2Stride, outW * sizeof(float), outH, cudaMemcpyDeviceToHost));

        warped_image1->host_img_ = out1Guard.release();
        warped_image1->width_ = outW;
        warped_image1->height_ = outH;

        warped_image2->host_img_ = out2Guard.release();
        warped_image2->width_ = outW;
        warped_image2->height_ = outH;
    }); // end cusift_api_guard for ExtractAndMatchAndFindHomographyAndWarp
}

void ExtractAndMatchAndFindHomographyAndWarp_GPU(const Image_t *image1, const Image_t *image2, SiftData *sift_data1, SiftData *sift_data2, float *homography, int *num_matches, const ExtractSiftOptions_t *extract_options, const FindHomographyOptions_t *homography_options, ImageStrided_t *warped_image1, ImageStrided_t *warped_image2)
{
    cusift_api_guard(__func__, [&]()
    {
        ValidateImage(image1, "image1");
        ValidateImage(image2, "image2");
        ValidateExtractOptions(extract_options);
        PrepareSiftData(sift_data1, "sift_data1");
        PrepareSiftData(sift_data2, "sift_data2");
        RequireNonNull(homography, "homography");
        RequireNonNull(num_matches, "num_matches");
        ValidateHomographyOptions(homography_options);
        RequireNonNull(warped_image1, "warped_image1");
        RequireNonNull(warped_image2, "warped_image2");

        int w1 = image1->width_, h1 = image1->height_;
        int w2 = image2->width_, h2 = image2->height_;
        int maxW = std::max(w1, w2);
        int maxH = std::max(h1, h2);

        // ── Extract SIFT features (images stay on device via CudaImageGuard) ──
        CudaImageGuard cuda_image1;
        CudaImageGuard cuda_image2;

        int minDim = std::min({w1, h1, w2, h2});
        int maxOctaves = static_cast<int>(std::floor(std::log2(minDim))) - 3;
        int octaves = std::min(extract_options->num_octaves_, maxOctaves);
        octaves = std::min(octaves, 7);
        octaves = std::max(octaves, 3);

        SiftTempMemoryGuard tempMemory(AllocSiftTempMemory(maxW, maxH, octaves));

        // Image 1
        InitSiftData(sift_data1, extract_options->max_keypoints_, true, true);
        CudaImage_Allocate(cuda_image1.get(), w1, h1,
                           p_iAlignUp(w1, 128), false, nullptr, image1->host_img_);
        CudaImage_Download(cuda_image1.get());
        ExtractSift(sift_data1, cuda_image1.get(), octaves,
                    extract_options->init_blur_, extract_options->thresh_,
                    extract_options->lowest_scale_, extract_options->highest_scale_, extract_options->edge_thresh_,
                    tempMemory.get());
        if (extract_options->scale_suppression_radius_ > 0.0f)
            SuppressEmbeddedPoints(sift_data1, extract_options->scale_suppression_radius_);

        // Image 2
        InitSiftData(sift_data2, extract_options->max_keypoints_, true, true);
        CudaImage_Allocate(cuda_image2.get(), w2, h2,
                           p_iAlignUp(w2, 128), false, nullptr, image2->host_img_);
        CudaImage_Download(cuda_image2.get());
        ExtractSift(sift_data2, cuda_image2.get(), octaves,
                    extract_options->init_blur_, extract_options->thresh_,
                    extract_options->lowest_scale_, extract_options->highest_scale_, extract_options->edge_thresh_,
                    tempMemory.get());
        if (extract_options->scale_suppression_radius_ > 0.0f)
            SuppressEmbeddedPoints(sift_data2, extract_options->scale_suppression_radius_);

        // ── Match ────────────────────────────────────────────────────────────
        MatchSiftData_private(sift_data1, sift_data2);

        // ── Find homography ─────────────────────────────────────────────────
        FindGeometricModel(
            sift_data1, homography, num_matches,
            homography_options);
        
        for (int i = 0; i < 8; i++)
        {
            if (!std::isfinite(homography[i]))
            {
                ERROR("Homography contains non-finite values");
            }
        }

        ImproveHomography(
            sift_data1, homography,
            homography_options->improve_num_loops_,
            homography_options->improve_min_score_,
            homography_options->improve_max_ambiguity_,
            homography_options->improve_thresh_);

        // ── Compute canvas bounding box ─────────────────────────────────────
        WarpGeometry_t geom;
        ComputeCanvasGeometry(homography, w1, h1, w2, h2, &geom);
        const int outW = geom.canvas_width;
        const int outH = geom.canvas_height;
        const float originU = (float)geom.origin_u;
        const float originV = (float)geom.origin_v;

        // ── Warp on GPU — source data is already device-resident ────────────
        float *d_src1 = cuda_image1.get()->d_data;
        float *d_src2 = cuda_image2.get()->d_data;
        int src1Pitch = cuda_image1.get()->pitch;
        int src2Pitch = cuda_image2.get()->pitch;

        DevicePtrGuard<float> d_out1Guard, d_out2Guard;

        size_t out1Stride = 0;
        size_t out2Stride = 0;

        safeCall(cudaMallocPitch(&d_out1Guard.getRef(), &out1Stride, outW * sizeof(float), outH));
        safeCall(cudaMallocPitch(&d_out2Guard.getRef(), &out2Stride, outW * sizeof(float), outH));

        float H00 = homography[0], H01 = homography[1], H02 = homography[2];
        float H10 = homography[3], H11 = homography[4], H12 = homography[5];
        float H20 = homography[6], H21 = homography[7], H22 = homography[8];

        dim3 threads(16, 16);
        dim3 blocks((outW + threads.x - 1) / threads.x,
                    (outH + threads.y - 1) / threads.y);

        WarpDualKernel<<<blocks, threads>>>(
            d_src1, d_out1Guard.get(), w1, h1, src1Pitch,
            d_src2, d_out2Guard.get(), w2, h2, src2Pitch,
            outW, outH, static_cast<int>(out1Stride / sizeof(float)),
            static_cast<int>(out2Stride / sizeof(float)), originU, originV,
            H00, H01, H02,
            H10, H11, H12,
            H20, H21, H22);
        checkMsg("WarpDualKernel() execution failed\n");

        safeCall(cudaDeviceSynchronize());

        // ── Fill output structs, release GPU ownership ──────────────────────
        warped_image1->strided_img_ = d_out1Guard.release();
        warped_image1->width_ = outW;
        warped_image1->height_ = outH;
        warped_image1->stride_ = out1Stride;

        warped_image2->strided_img_ = d_out2Guard.release();
        warped_image2->width_ = outW;
        warped_image2->height_ = outH;
        warped_image2->stride_ = out2Stride;
    }); // end cusift_api_guard for ExtractAndMatchAndFindHomographyAndWarp_GPU
}

void ExtractAndMatchAndFindHomography_Multi_AndWarp(const Image_t *image1, const Image_t *image2, SiftData *sift_data1, SiftData *sift_data2, float *homography, int *num_matches, const ExtractSiftOptions_t *extract_options, const FindHomographyOptions_t *homography_options, Image_t *warped_image1, Image_t *warped_image2, int num_homography_attempts, int homography_goal)
{
    cusift_api_guard(__func__, [&]()
    {
        ValidateImage(image1, "image1");
        ValidateImage(image2, "image2");
        ValidateExtractOptions(extract_options);
        PrepareSiftData(sift_data1, "sift_data1");
        PrepareSiftData(sift_data2, "sift_data2");
        RequireNonNull(homography, "homography");
        RequireNonNull(num_matches, "num_matches");
        ValidateHomographyOptions(homography_options);
        RequireNonNull(warped_image1, "warped_image1");
        RequireNonNull(warped_image2, "warped_image2");

        int w1 = image1->width_, h1 = image1->height_;
        int w2 = image2->width_, h2 = image2->height_;
        int maxW = std::max(w1, w2);
        int maxH = std::max(h1, h2);

        CudaImageGuard cuda_image1;
        CudaImageGuard cuda_image2;

        int minDim = std::min({w1, h1, w2, h2});
        int maxOctaves = static_cast<int>(std::floor(std::log2(minDim))) - 3;
        int octaves = std::min(extract_options->num_octaves_, maxOctaves);
        octaves = std::min(octaves, 7);
        octaves = std::max(octaves, 3);

        SiftTempMemoryGuard tempMemory(AllocSiftTempMemory(maxW, maxH, octaves));

        InitSiftData(sift_data1, extract_options->max_keypoints_, true, true);
        CudaImage_Allocate(cuda_image1.get(), w1, h1,
                           p_iAlignUp(w1, 128), false, nullptr, image1->host_img_);
        CudaImage_Download(cuda_image1.get());
        ExtractSift(sift_data1, cuda_image1.get(), octaves,
                    extract_options->init_blur_, extract_options->thresh_,
                    extract_options->lowest_scale_, extract_options->highest_scale_, extract_options->edge_thresh_,
                    tempMemory.get());
        if (extract_options->scale_suppression_radius_ > 0.0f)
            SuppressEmbeddedPoints(sift_data1, extract_options->scale_suppression_radius_);

        InitSiftData(sift_data2, extract_options->max_keypoints_, true, true);
        CudaImage_Allocate(cuda_image2.get(), w2, h2,
                           p_iAlignUp(w2, 128), false, nullptr, image2->host_img_);
        CudaImage_Download(cuda_image2.get());
        ExtractSift(sift_data2, cuda_image2.get(), octaves,
                    extract_options->init_blur_, extract_options->thresh_,
                    extract_options->lowest_scale_, extract_options->highest_scale_, extract_options->edge_thresh_,
                    tempMemory.get());
        if (extract_options->scale_suppression_radius_ > 0.0f)
            SuppressEmbeddedPoints(sift_data2, extract_options->scale_suppression_radius_);

        MatchSiftData_private(sift_data1, sift_data2);

        int attempts = std::max(num_homography_attempts, 1);
        FindHomographyOptions_t opts = *homography_options;
        std::random_device rd;

        float bestH[9] = {};
        int bestInliers = -1;
        float bestScore = FLT_MAX;

        for (int attempt = 0; attempt < attempts; attempt++)
        {
            float candidateH[9];
            int candidateInliers = 0;

            if (homography_options->seed_ == 0)
                opts.seed_ = rd();
            else
                opts.seed_ = homography_options->seed_ + (unsigned int)attempt;

            FindGeometricModel(
                sift_data1, candidateH, &candidateInliers,
                &opts);

            bool finite = true;
            for (int i = 0; i < 9; i++)
            {
                if (!std::isfinite(candidateH[i]))
                {
                    finite = false;
                    break;
                }
            }
            if (!finite)
                continue;

            ImproveHomography(
                sift_data1, candidateH,
                opts.improve_num_loops_,
                opts.improve_min_score_,
                opts.improve_max_ambiguity_,
                opts.improve_thresh_);

            bool isBetter = false;
            if (homography_goal == CUSIFT_HOMOGRAPHY_GOAL_MIN_EYE_DIFF)
            {
                float score = EyeDiff2x2(candidateH);
                if (bestInliers < 0 || score < bestScore)
                {
                    bestScore = score;
                    isBetter = true;
                }
            }
            else
            {
                if (candidateInliers > bestInliers)
                    isBetter = true;
            }

            if (isBetter)
            {
                std::memcpy(bestH, candidateH, sizeof(bestH));
                bestInliers = candidateInliers;
            }
        }

        if (bestInliers < 0)
        {
            ERROR("Homography contains non-finite values in all attempts");
        }

        std::memcpy(homography, bestH, 9 * sizeof(float));
        *num_matches = bestInliers;

        WarpGeometry_t geom;
        ComputeCanvasGeometry(homography, w1, h1, w2, h2, &geom);
        const int outW = geom.canvas_width;
        const int outH = geom.canvas_height;
        const float originU = (float)geom.origin_u;
        const float originV = (float)geom.origin_v;

        float *d_src1 = cuda_image1.get()->d_data;
        float *d_src2 = cuda_image2.get()->d_data;
        int src1Pitch = cuda_image1.get()->pitch;
        int src2Pitch = cuda_image2.get()->pitch;

        DevicePtrGuard<float> d_out1Guard, d_out2Guard;

        size_t out1Stride = 0;
        size_t out2Stride = 0;

        safeCall(cudaMallocPitch(&d_out1Guard.getRef(), &out1Stride, outW * sizeof(float), outH));
        safeCall(cudaMallocPitch(&d_out2Guard.getRef(), &out2Stride, outW * sizeof(float), outH));

        float H00 = homography[0], H01 = homography[1], H02 = homography[2];
        float H10 = homography[3], H11 = homography[4], H12 = homography[5];
        float H20 = homography[6], H21 = homography[7], H22 = homography[8];

        dim3 threads(16, 16);
        dim3 blocks((outW + threads.x - 1) / threads.x,
                    (outH + threads.y - 1) / threads.y);

        WarpDualKernel<<<blocks, threads>>>(
            d_src1, d_out1Guard.get(), w1, h1, src1Pitch,
            d_src2, d_out2Guard.get(), w2, h2, src2Pitch,
            outW, outH, static_cast<int>(out1Stride / sizeof(float)),
            static_cast<int>(out2Stride / sizeof(float)), originU, originV,
            H00, H01, H02,
            H10, H11, H12,
            H20, H21, H22);
        checkMsg("WarpDualKernel() execution failed\n");

        safeCall(cudaDeviceSynchronize());

        size_t nPixels = (size_t)outW * outH;
        HostPtrGuard<float> out1Guard((float *)malloc(sizeof(float) * nPixels));
        HostPtrGuard<float> out2Guard((float *)malloc(sizeof(float) * nPixels));
        if (!out1Guard.get() || !out2Guard.get())
        {
            ERROR("Warping: return image host allocation failed");
        }

        safeCall(cudaMemcpy2D(out1Guard.get(), outW * sizeof(float), d_out1Guard.get(), out1Stride, outW * sizeof(float), outH, cudaMemcpyDeviceToHost));
        safeCall(cudaMemcpy2D(out2Guard.get(), outW * sizeof(float), d_out2Guard.get(), out2Stride, outW * sizeof(float), outH, cudaMemcpyDeviceToHost));

        warped_image1->host_img_ = out1Guard.release();
        warped_image1->width_ = outW;
        warped_image1->height_ = outH;

        warped_image2->host_img_ = out2Guard.release();
        warped_image2->width_ = outW;
        warped_image2->height_ = outH;
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

size_t EstimateVramExtractSift(int image_width, int image_height, const ExtractSiftOptions_t *options)
{
    if (!options)
        return 0;
    int octaves = clampOctaves(options->num_octaves_, image_width, image_height);

    size_t siftBytes    = estimateSiftDataDeviceBytes(options->max_keypoints_);
    // thrust::sort in ExtractSift allocates temporary storage of roughly one
    // extra copy of the keypoint array while it runs.
    size_t sortTmpBytes = siftBytes;
    size_t imageBytes   = estimateCudaImageBytes(image_width, image_height);
    size_t pyramidBytes = estimatePyramidBytes(image_width, image_height, octaves);
    size_t contextBytes = estimateContextBytes();

    return siftBytes + sortTmpBytes + imageBytes + pyramidBytes + contextBytes;
}

size_t EstimateVramMatchSift(int max_keypoints1, int max_keypoints2)
{
    return estimateSiftDataDeviceBytes(max_keypoints1)
         + estimateSiftDataDeviceBytes(max_keypoints2);
}

size_t EstimateVramFindHomography(int max_keypoints, const FindHomographyOptions_t *options)
{
    if (!options)
        return 0;
    // Existing SiftData that must be resident
    size_t siftBytes = estimateSiftDataDeviceBytes(max_keypoints);

    // Temporary RANSAC buffers (mirroring matching.cu)
    int numPtsUp = ((max_keypoints + 15) / 16) * 16;
    int numLoops = ((options->num_loops_ + 15) / 16) * 16;

    size_t coordBytes    = 4 * sizeof(float) * numPtsUp;

    size_t randPtsBytes, modelBytes;
    if (options->model_type_ == CUSIFT_MODEL_SIMILARITY) {
        randPtsBytes = 2 * sizeof(int)   * numLoops;
        modelBytes   = 4 * sizeof(float) * numLoops;
    } else {
        randPtsBytes = 4 * sizeof(int)   * numLoops;
        modelBytes   = 8 * sizeof(float) * numLoops;
    }

    return siftBytes + coordBytes + randPtsBytes + modelBytes;
}

size_t EstimateVramWarpImages(int image_width1, int image_height1, int image_width2, int image_height2)
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

size_t EstimateVramFullPipeline(int image_width1, int image_height1,
                                int image_width2, int image_height2,
                                const ExtractSiftOptions_t *extract_options,
                                const FindHomographyOptions_t *homography_options)
{
    if (!extract_options || !homography_options)
        return 0;
    // SiftData for both images persists across all stages
    size_t siftBoth = estimateSiftDataDeviceBytes(extract_options->max_keypoints_) * 2;

    // Peak during extraction (image 1 or 2, whichever is larger)
    int maxW = std::max(image_width1, image_width2);
    int maxH = std::max(image_height1, image_height2);
    int octaves = clampOctaves(extract_options->num_octaves_, maxW, maxH);

    size_t extractPeak = siftBoth
                       + estimateSiftDataDeviceBytes(extract_options->max_keypoints_) // thrust::sort temporaries
                       + estimateCudaImageBytes(maxW, maxH)
                       + estimatePyramidBytes(maxW, maxH, octaves)
                       + estimateContextBytes();

    // Peak during homography (SiftData + RANSAC temporaries)
    size_t homographyPeak = siftBoth
                          + EstimateVramFindHomography(extract_options->max_keypoints_, homography_options)
                          - estimateSiftDataDeviceBytes(extract_options->max_keypoints_); // avoid double-counting the one SiftData

    // Peak during warping (SiftData already freed in convenience funcs, but be conservative)
    size_t warpPeak = siftBoth
                    + EstimateVramWarpImages(image_width1, image_height1, image_width2, image_height2);

    return std::max({extractPeak, homographyPeak, warpPeak});
}