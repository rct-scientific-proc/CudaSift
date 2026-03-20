#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "cusift.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <cstdio>
#include <chrono>

// -- Helpers ---------------------------------------------

static int load_image_to_grayscale_float(const char* filename, std::vector<float>& image, int& width, int& height)
{
    int channels;
    unsigned char* data = stbi_load(filename, &width, &height, &channels, 1);
    if (data == nullptr)
    {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return 1;
    }

    image.resize(width * height);
    for (int i = 0; i < width * height; ++i)
    {
        image[i] = static_cast<float>(data[i]);
    }

    stbi_image_free(data);
    return 0;
}

static void upscale(std::vector<float>& upImage, int& upwidth, int& upheight, const std::vector<float>& image, int width, int height, float scale)
{
    int upWidth = static_cast<int>(width * scale);
    int upHeight = static_cast<int>(height * scale);
    upImage.resize(upWidth * upHeight);
    upwidth = upWidth;
    upheight = upHeight;

    for (int y = 0; y < upHeight; ++y)
    {
        float srcY = y / scale;
        int y0 = static_cast<int>(std::floor(srcY));
        int y1 = std::min(y0 + 1, height - 1);
        float yLerp = srcY - y0;

        for (int x = 0; x < upWidth; ++x)
        {
            float srcX = x / scale;
            int x0 = static_cast<int>(std::floor(srcX));
            int x1 = std::min(x0 + 1, width - 1);
            float xLerp = srcX - x0;

            float topLeft = image[y0 * width + x0];
            float topRight = image[y0 * width + x1];
            float bottomLeft = image[y1 * width + x0];
            float bottomRight = image[y1 * width + x1];

            float top = topLeft + (topRight - topLeft) * xLerp;
            float bottom = bottomLeft + (bottomRight - bottomLeft) * xLerp;
            upImage[y * upWidth + x] = top + (bottom - top) * yLerp;
        }
    }
}

static bool check_error(const char* label)
{
    if (CusiftHadError())
    {
        char err[256], file[256];
        int line;
        CusiftGetLastErrorString(&line, file, err);
        std::cerr << "  ERROR in " << label << ": " << err << std::endl;
        return true;
    }
    return false;
}

static ExtractSiftOptions_t default_extract_options()
{
    ExtractSiftOptions_t opts;
    opts.thresh_ = 2.0f;
    opts.lowest_scale_ = 0.0f;
    opts.highest_scale_ = std::numeric_limits<float>::infinity();
    opts.edge_thresh_ = 10.0f;
    opts.init_blur_ = 1.0f;
    opts.max_keypoints_ = 65536;
    opts.num_octaves_ = 7;
    opts.scale_suppression_radius_ = 0.0f;
    return opts;
}

static FindHomographyOptions_t default_homography_options()
{
    FindHomographyOptions_t opts;
    opts.num_loops_ = 100000;
    opts.min_score_ = 0.0f;
    opts.max_ambiguity_ = 1.0f;
    opts.thresh_ = 3.0f;
    opts.improve_num_loops_ = 5;
    opts.improve_min_score_ = 0.0f;
    opts.improve_max_ambiguity_ = 1.0f;
    opts.improve_thresh_ = 2.0f;
    opts.seed_ = 42;
    opts.model_type_ = CUSIFT_MODEL_HOMOGRAPHY;
    return opts;
}

static std::string fmt_bytes(size_t bytes)
{
    if (bytes >= 1024ULL * 1024 * 1024)
    {
        double gb = bytes / (1024.0 * 1024.0 * 1024.0);
        char buf[32];
        snprintf(buf, sizeof(buf), "%.2f GB", gb);
        return buf;
    }
    double mb = bytes / (1024.0 * 1024.0);
    char buf[32];
    snprintf(buf, sizeof(buf), "%.2f MB", mb);
    return buf;
}

static std::string fmt_ms(double ms)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "%.2f ms", ms);
    return buf;
}

static std::string fmt_resolution(int w, int h)
{
    return std::to_string(w) + "x" + std::to_string(h);
}

static std::string fmt_ratio(double ratio)
{
    char buf[32];
    snprintf(buf, sizeof(buf), "%.2fx", ratio);
    return buf;
}

static std::string fmt_throughput(double gb_per_sec)
{
    char buf[32];
    if (gb_per_sec >= 1.0)
        snprintf(buf, sizeof(buf), "%.2f GB/s", gb_per_sec);
    else
        snprintf(buf, sizeof(buf), "%.2f MB/s", gb_per_sec * 1024.0);
    return buf;
}

// -- Per-resolution benchmark result ---------------------

struct BenchResult
{
    std::string label;
    int model_type;
    int w1, h1, w2, h2;

    double extract_ms;
    int keypoints1, keypoints2;

    double match_ms;
    int matched;

    double homography_ms;
    int inliers;

    double warp_gpu_ms;
    int warp_w, warp_h;

    double full_pipeline_ms;

    size_t vram_extract;
    size_t vram_match;
    size_t vram_homography;
    size_t vram_warp;
    size_t vram_full;

    bool ok;
};

static BenchResult benchmark(const char* label, const Image_t& im1, const Image_t& im2, int model_type = CUSIFT_MODEL_HOMOGRAPHY)
{
    using Clock = std::chrono::high_resolution_clock;

    BenchResult r{};
    r.label = label;
    r.model_type = model_type;
    r.w1 = im1.width_;  r.h1 = im1.height_;
    r.w2 = im2.width_;  r.h2 = im2.height_;
    r.ok = true;

    ExtractSiftOptions_t eo = default_extract_options();
    FindHomographyOptions_t ho = default_homography_options();
    ho.model_type_ = model_type;

    // -- VRAM estimates (no GPU calls) -------------------
    r.vram_extract    = EstimateVramExtractSift(im1.width_, im1.height_, &eo);
    r.vram_match      = EstimateVramMatchSift(eo.max_keypoints_, eo.max_keypoints_);
    r.vram_homography = EstimateVramFindHomography(eo.max_keypoints_, &ho);
    r.vram_warp       = EstimateVramWarpImages(im1.width_, im1.height_, im2.width_, im2.height_);
    r.vram_full       = EstimateVramFullPipeline(im1.width_, im1.height_, im2.width_, im2.height_, &eo, &ho);

    // -- ExtractSiftFromImage ----------------------------
    {
        SiftData sd1{}, sd2{};
        auto t0 = Clock::now();
        ExtractSiftFromImage(&im1, &sd1, &eo);
        ExtractSiftFromImage(&im2, &sd2, &eo);
        auto t1 = Clock::now();
        r.extract_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        r.keypoints1 = sd1.numPts;
        r.keypoints2 = sd2.numPts;
        if (check_error("ExtractSiftFromImage")) r.ok = false;
        DeleteSiftData(&sd1);
        DeleteSiftData(&sd2);
    }

    // -- MatchSiftData -----------------------------------
    {
        SiftData sd1{}, sd2{};
        ExtractSiftFromImage(&im1, &sd1, &eo);
        ExtractSiftFromImage(&im2, &sd2, &eo);

        auto t0 = Clock::now();
        MatchSiftData(&sd1, &sd2);
        auto t1 = Clock::now();
        r.match_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        int m = 0;
        for (int i = 0; i < sd1.numPts; i++)
            if (sd1.h_data[i].match >= 0) m++;
        r.matched = m;
        if (check_error("MatchSiftData")) r.ok = false;
        DeleteSiftData(&sd1);
        DeleteSiftData(&sd2);
    }

    // -- FindHomography ----------------------------------
    {
        SiftData sd1{}, sd2{};
        ExtractSiftFromImage(&im1, &sd1, &eo);
        ExtractSiftFromImage(&im2, &sd2, &eo);
        MatchSiftData(&sd1, &sd2);

        float H[9]{};
        int nm = 0;
        auto t0 = Clock::now();
        FindHomography(&sd1, H, &nm, &ho);
        auto t1 = Clock::now();
        r.homography_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        r.inliers = nm;
        if (check_error("FindHomography")) r.ok = false;
        DeleteSiftData(&sd1);
        DeleteSiftData(&sd2);
    }

    // -- WarpImages (GPU path) ---------------------------
    {
        SiftData sd1{}, sd2{};
        ExtractSiftFromImage(&im1, &sd1, &eo);
        ExtractSiftFromImage(&im2, &sd2, &eo);
        MatchSiftData(&sd1, &sd2);
        float H[9]{};
        int nm = 0;
        FindHomography(&sd1, H, &nm, &ho);

        bool valid = nm > 0;
        for (int i = 0; i < 9 && valid; i++)
            if (!std::isfinite(H[i])) valid = false;

        if (valid)
        {
            Image_t w1{}, w2{};
            auto t0 = Clock::now();
            WarpImages(&im1, &im2, H, &w1, &w2, true);
            auto t1 = Clock::now();
            r.warp_gpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            r.warp_w = w1.width_;
            r.warp_h = w1.height_;
            if (check_error("WarpImages")) r.ok = false;
            FreeImage(&w1);
            FreeImage(&w2);
        }

        DeleteSiftData(&sd1);
        DeleteSiftData(&sd2);
    }

    // -- Full pipeline (convenience function) ------------
    {
        SiftData sd1{}, sd2{};
        float H[9]{};
        int nm = 0;
        Image_t w1{}, w2{};

        auto t0 = Clock::now();
        ExtractAndMatchAndFindHomographyAndWarp(&im1, &im2, &sd1, &sd2, H, &nm, &eo, &ho, &w1, &w2);
        auto t1 = Clock::now();
        r.full_pipeline_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (check_error("FullPipeline")) r.ok = false;
        FreeImage(&w1);
        FreeImage(&w2);
        DeleteSiftData(&sd1);
        DeleteSiftData(&sd2);
    }

    return r;
}

// -- Pretty-print results table --------------------------

static void print_separator(int total_width)
{
    std::cout << std::string(total_width, '-') << std::endl;
}

static void print_results(const std::vector<BenchResult>& results)
{
    // Column widths
    const int cLabel = 14;
    const int cRes   = 14;
    const int cKP    = 16;
    const int cTime  = 14;
    const int cVram  = 14;

    int total = cLabel + cRes + cKP + 4 * cTime + cVram + 9; // 9 separators

    std::cout << "\n";
    print_separator(total);
    std::cout << std::left
              << std::setw(cLabel) << "Scale"
              << " | " << std::setw(cRes)   << "Resolution"
              << " | " << std::setw(cKP)    << "Keypoints"
              << " | " << std::setw(cTime)  << "Extract"
              << " | " << std::setw(cTime)  << "Match"
              << " | " << std::setw(cTime)  << "Homography"
              << " | " << std::setw(cTime)  << "Warp (GPU)"
              << " | " << std::setw(cVram)  << "Est. VRAM"
              << std::endl;
    print_separator(total);

    for (const auto& r : results)
    {
        std::string res = fmt_resolution(r.w1, r.h1);
        std::string kps = std::to_string(r.keypoints1) + " / " + std::to_string(r.keypoints2);

        std::cout << std::left
                  << std::setw(cLabel) << r.label
                  << " | " << std::setw(cRes)   << res
                  << " | " << std::setw(cKP)    << kps
                  << " | " << std::setw(cTime)  << fmt_ms(r.extract_ms)
                  << " | " << std::setw(cTime)  << fmt_ms(r.match_ms)
                  << " | " << std::setw(cTime)  << fmt_ms(r.homography_ms)
                  << " | " << std::setw(cTime)  << (r.warp_gpu_ms > 0 ? fmt_ms(r.warp_gpu_ms) : "N/A")
                  << " | " << std::setw(cVram)  << fmt_bytes(r.vram_full)
                  << std::endl;
    }

    print_separator(total);

    // Detailed VRAM breakdown
    std::cout << "\n";
    int vTotal = cLabel + 5 * 16 + 6;
    print_separator(vTotal);
    std::cout << std::left
              << std::setw(cLabel) << "Scale"
              << " | " << std::setw(16) << "VRAM Extract"
              << " | " << std::setw(16) << "VRAM Match"
              << " | " << std::setw(16) << "VRAM Homog."
              << " | " << std::setw(16) << "VRAM Warp"
              << " | " << std::setw(16) << "VRAM Peak"
              << std::endl;
    print_separator(vTotal);

    for (const auto& r : results)
    {
        std::cout << std::left
                  << std::setw(cLabel) << r.label
                  << " | " << std::setw(16) << fmt_bytes(r.vram_extract)
                  << " | " << std::setw(16) << fmt_bytes(r.vram_match)
                  << " | " << std::setw(16) << fmt_bytes(r.vram_homography)
                  << " | " << std::setw(16) << fmt_bytes(r.vram_warp)
                  << " | " << std::setw(16) << fmt_bytes(r.vram_full)
                  << std::endl;
    }

    print_separator(vTotal);

    // Detailed timing breakdown
    std::cout << "\n";
    int tTotal = cLabel + 14 + 12 + 14 + 16 + 8;
    print_separator(tTotal);
    std::cout << std::left
              << std::setw(cLabel) << "Scale"
              << " | " << std::setw(14) << "Full Pipeline"
              << " | " << std::setw(12) << "Matches"
              << " | " << std::setw(14) << "Inliers"
              << " | " << std::setw(16) << "Warp Output"
              << std::endl;
    print_separator(tTotal);

    for (const auto& r : results)
    {
        std::string warpRes = (r.warp_w > 0) ? fmt_resolution(r.warp_w, r.warp_h) : "N/A";
        std::cout << std::left
                  << std::setw(cLabel) << r.label
                  << " | " << std::setw(14) << fmt_ms(r.full_pipeline_ms)
                  << " | " << std::setw(12) << r.matched
                  << " | " << std::setw(14) << r.inliers
                  << " | " << std::setw(16) << warpRes
                  << std::endl;
    }

    print_separator(tTotal);

    // Pipeline ratio: sequential (sum of individual stages) vs full pipeline
    std::cout << "\n";
    const int cRLabel = 14;
    const int cRSeq   = 16;
    const int cRFull  = 16;
    const int cRRatio = 12;
    int rTotal = cRLabel + cRSeq + cRFull + cRRatio + 9;
    print_separator(rTotal);
    std::cout << std::left
              << std::setw(cRLabel) << "Scale"
              << " | " << std::setw(cRSeq)   << "Sequential"
              << " | " << std::setw(cRFull)  << "Full Pipeline"
              << " | " << std::setw(cRRatio) << "Speedup"
              << std::endl;
    print_separator(rTotal);

    for (const auto& r : results)
    {
        double seq_ms = r.extract_ms + r.match_ms + r.homography_ms + r.warp_gpu_ms;
        double ratio = (r.full_pipeline_ms > 0) ? seq_ms / r.full_pipeline_ms : 0.0;

        std::cout << std::left
                  << std::setw(cRLabel) << r.label
                  << " | " << std::setw(cRSeq)  << fmt_ms(seq_ms)
                  << " | " << std::setw(cRFull) << fmt_ms(r.full_pipeline_ms)
                  << " | " << std::setw(cRRatio) << fmt_ratio(ratio)
                  << std::endl;
    }

    print_separator(rTotal);

    // Memory bandwidth / throughput estimation
    // Estimated as: peak VRAM touched / full pipeline time
    // This gives an effective throughput figure (how fast we move through GPU memory)
    std::cout << "\n";
    const int cBLabel = 14;
    const int cBData  = 16;
    const int cBTime  = 16;
    const int cBBW    = 16;
    const int cBImg   = 16;
    int bTotal = cBLabel + cBData + cBTime + cBBW + cBImg + 12;
    print_separator(bTotal);
    std::cout << std::left
              << std::setw(cBLabel) << "Scale"
              << " | " << std::setw(cBData) << "Est. VRAM Peak"
              << " | " << std::setw(cBTime) << "Pipeline Time"
              << " | " << std::setw(cBBW)   << "Eff. Throughput"
              << " | " << std::setw(cBImg)  << "Image Pixels/ms"
              << std::endl;
    print_separator(bTotal);

    for (const auto& r : results)
    {
        // Effective throughput: estimated VRAM peak / pipeline time
        double gb_per_sec = 0.0;
        if (r.full_pipeline_ms > 0)
            gb_per_sec = (r.vram_full / (1024.0 * 1024.0 * 1024.0)) / (r.full_pipeline_ms / 1000.0);

        // Image throughput: total input pixels processed per ms
        double total_pixels = (double)r.w1 * r.h1 + (double)r.w2 * r.h2;
        double mpix_per_ms = (r.full_pipeline_ms > 0) ? (total_pixels / 1e6) / r.full_pipeline_ms : 0.0;

        char mpix_buf[32];
        snprintf(mpix_buf, sizeof(mpix_buf), "%.3f Mpix/ms", mpix_per_ms);

        std::cout << std::left
                  << std::setw(cBLabel) << r.label
                  << " | " << std::setw(cBData) << fmt_bytes(r.vram_full)
                  << " | " << std::setw(cBTime) << fmt_ms(r.full_pipeline_ms)
                  << " | " << std::setw(cBBW)   << fmt_throughput(gb_per_sec)
                  << " | " << std::setw(cBImg)  << mpix_buf
                  << std::endl;
    }

    print_separator(bTotal);
}

static void print_model_comparison(const std::vector<BenchResult>& results)
{
    // Group results by label: find pairs (HOMOGRAPHY, SIMILARITY)
    struct Pair { const BenchResult* homo; const BenchResult* sim; };
    std::vector<Pair> pairs;
    for (size_t i = 0; i < results.size(); i++)
    {
        if (results[i].model_type != CUSIFT_MODEL_HOMOGRAPHY) continue;
        Pair p{ &results[i], nullptr };
        for (size_t j = 0; j < results.size(); j++)
        {
            if (results[j].model_type == CUSIFT_MODEL_SIMILARITY
                && results[j].label == results[i].label)
            {
                p.sim = &results[j];
                break;
            }
        }
        if (p.sim) pairs.push_back(p);
    }
    if (pairs.empty()) return;

    const int cL = 14, cH = 16, cS = 16, cD = 14, cR = 12;
    const int cIH = 12, cIS = 12;
    int tw = cL + cH + cS + cD + cR + cIH + cIS + 18;

    std::cout << "\nModel Type Comparison: Homography Stage\n";
    print_separator(tw);
    std::cout << std::left
              << std::setw(cL)  << "Scale"
              << " | " << std::setw(cH)  << "Homography (ms)"
              << " | " << std::setw(cS)  << "Similarity (ms)"
              << " | " << std::setw(cD)  << "Delta (ms)"
              << " | " << std::setw(cR)  << "Speedup"
              << " | " << std::setw(cIH) << "Inliers (H)"
              << " | " << std::setw(cIS) << "Inliers (S)"
              << std::endl;
    print_separator(tw);

    for (const auto& p : pairs)
    {
        double delta = p.homo->homography_ms - p.sim->homography_ms;
        double speedup = (p.sim->homography_ms > 0) ? p.homo->homography_ms / p.sim->homography_ms : 0.0;

        std::cout << std::left
                  << std::setw(cL) << p.homo->label
                  << " | " << std::setw(cH) << fmt_ms(p.homo->homography_ms)
                  << " | " << std::setw(cS) << fmt_ms(p.sim->homography_ms)
                  << " | " << std::setw(cD) << fmt_ms(delta)
                  << " | " << std::setw(cR) << fmt_ratio(speedup)
                  << " | " << std::setw(cIH) << p.homo->inliers
                  << " | " << std::setw(cIS) << p.sim->inliers
                  << std::endl;
    }

    print_separator(tw);

    // Full pipeline comparison
    const int cFH = 18, cFS = 18, cFD = 14, cFR = 12;
    int fw = cL + cFH + cFS + cFD + cFR + 12;

    std::cout << "\nModel Type Comparison: Full Pipeline\n";
    print_separator(fw);
    std::cout << std::left
              << std::setw(cL)  << "Scale"
              << " | " << std::setw(cFH) << "Pipeline H (ms)"
              << " | " << std::setw(cFS) << "Pipeline S (ms)"
              << " | " << std::setw(cFD) << "Delta (ms)"
              << " | " << std::setw(cFR) << "Speedup"
              << std::endl;
    print_separator(fw);

    for (const auto& p : pairs)
    {
        double delta = p.homo->full_pipeline_ms - p.sim->full_pipeline_ms;
        double speedup = (p.sim->full_pipeline_ms > 0) ? p.homo->full_pipeline_ms / p.sim->full_pipeline_ms : 0.0;

        std::cout << std::left
                  << std::setw(cL) << p.homo->label
                  << " | " << std::setw(cFH) << fmt_ms(p.homo->full_pipeline_ms)
                  << " | " << std::setw(cFS) << fmt_ms(p.sim->full_pipeline_ms)
                  << " | " << std::setw(cFD) << fmt_ms(delta)
                  << " | " << std::setw(cFR) << fmt_ratio(speedup)
                  << std::endl;
    }

    print_separator(fw);
}

// -- Main ------------------------------------------------

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <image1> <image2>" << std::endl;
        return 1;
    }

    const char* image1_path = argv[1];
    const char* image2_path = argv[2];

    std::vector<float> image1, image2;
    int width1, height1, width2, height2;
    if (load_image_to_grayscale_float(image1_path, image1, width1, height1) != 0)
        return 1;
    if (load_image_to_grayscale_float(image2_path, image2, width2, height2) != 0)
        return 1;

    std::cout << "Image 1: " << image1_path << " (" << width1 << "x" << height1 << ")" << std::endl;
    std::cout << "Image 2: " << image2_path << " (" << width2 << "x" << height2 << ")" << std::endl;
    std::cout << std::endl;

    InitializeCudaSift();
    if (check_error("InitializeCudaSift")) return 1;

    // Prepare upscaled variants
    std::vector<float> up1_2x, up2_2x;
    int uw1_2, uh1_2, uw2_2, uh2_2;
    upscale(up1_2x, uw1_2, uh1_2, image1, width1, height1, 2.0f);
    upscale(up2_2x, uw2_2, uh2_2, image2, width2, height2, 2.0f);

    std::vector<float> up1_4x, up2_4x;
    int uw1_4, uh1_4, uw2_4, uh2_4;
    upscale(up1_4x, uw1_4, uh1_4, up1_2x, uw1_2, uh1_2, 2.0f);
    upscale(up2_4x, uw2_4, uh2_4, up2_2x, uw2_2, uh2_2, 2.0f);

    std::vector<float> up1_8x, up2_8x;
    int uw1_8, uh1_8, uw2_8, uh2_8;
    upscale(up1_8x, uw1_8, uh1_8, up1_4x, uw1_4, uh1_4, 2.0f);
    upscale(up2_8x, uw2_8, uh2_8, up2_4x, uw2_4, uh2_4, 2.0f);

    Image_t im1   = { image1.data(),  width1, height1 };
    Image_t im2   = { image2.data(),  width2, height2 };
    Image_t im1_2 = { up1_2x.data(),  uw1_2,  uh1_2 };
    Image_t im2_2 = { up2_2x.data(),  uw2_2,  uh2_2 };
    Image_t im1_4 = { up1_4x.data(),  uw1_4,  uh1_4 };
    Image_t im2_4 = { up2_4x.data(),  uw2_4,  uh2_4 };
    Image_t im1_8 = { up1_8x.data(),  uw1_8,  uh1_8 };
    Image_t im2_8 = { up2_8x.data(),  uw2_8,  uh2_8 };

    std::cout << "Running benchmarks..." << std::endl;

    struct ScaleEntry { const char* label; const Image_t* a; const Image_t* b; };
    ScaleEntry scales[] = {
        { "1x (original)", &im1,   &im2 },
        { "2x upscale",   &im1_2, &im2_2 },
        { "4x upscale",   &im1_4, &im2_4 },
        { "8x upscale",   &im1_8, &im2_8 },
    };

    std::vector<BenchResult> results;
    for (const auto& s : scales)
    {
        results.push_back(benchmark(s.label, *s.a, *s.b, CUSIFT_MODEL_HOMOGRAPHY));
        results.push_back(benchmark(s.label, *s.a, *s.b, CUSIFT_MODEL_SIMILARITY));
    }

    // Print the standard tables using only the homography results
    std::vector<BenchResult> homo_results;
    for (const auto& r : results)
        if (r.model_type == CUSIFT_MODEL_HOMOGRAPHY)
            homo_results.push_back(r);
    print_results(homo_results);

    // Print model comparison tables
    print_model_comparison(results);

    return 0;
}

