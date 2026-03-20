#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "cusift.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdio>

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
        image[i] = static_cast<float>(data[i]); // Keep [0, 255] range for DoG thresholds
    }

    stbi_image_free(data);
    return 0;
}

static bool check_error(const char* test_name)
{
    if (CusiftHadError())
    {
        char error_str[256], filename[256];
        int line_number;
        CusiftGetLastErrorString(&line_number, filename, error_str);
        std::cerr << "  [FAIL] " << test_name << ": " << error_str << std::endl;
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
    opts.max_keypoints_ = 10000;
    opts.num_octaves_ = 5;
    opts.scale_suppression_radius_ = 0.0f;
    return opts;
}

static FindHomographyOptions_t default_homography_options()
{
    FindHomographyOptions_t opts;
    opts.num_loops_ = 10000;
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

static void print_homography(const float* H)
{
    for (int r = 0; r < 3; r++)
        std::cout << "    [" << H[r*3+0] << "  " << H[r*3+1] << "  " << H[r*3+2] << "]" << std::endl;
}

static bool homography_is_valid(const float* H)
{
    for (int i = 0; i < 9; i++)
        if (!std::isfinite(H[i])) return false;
    return true;
}

// -- Test: ExtractSiftFromImage --------------------------

static bool test_extract(const Image_t& im1, const Image_t& im2)
{
    std::cout << "[TEST] ExtractSiftFromImage" << std::endl;

    SiftData sift_data1, sift_data2;
    ExtractSiftOptions_t options = default_extract_options();

    ExtractSiftFromImage(&im1, &sift_data1, &options);
    if (check_error("ExtractSiftFromImage (image1)")) return false;

    ExtractSiftFromImage(&im2, &sift_data2, &options);
    if (check_error("ExtractSiftFromImage (image2)")) return false;

    std::cout << "  Image 1: " << sift_data1.numPts << " keypoints" << std::endl;
    std::cout << "  Image 2: " << sift_data2.numPts << " keypoints" << std::endl;

    bool pass = sift_data1.numPts > 0 && sift_data2.numPts > 0;
    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;

    DeleteSiftData(&sift_data1);
    DeleteSiftData(&sift_data2);
    return pass;
}

// -- Test: MatchSiftData ---------------------------------

static bool test_match(const Image_t& im1, const Image_t& im2)
{
    std::cout << "[TEST] MatchSiftData" << std::endl;

    SiftData sift_data1, sift_data2;
    ExtractSiftOptions_t options = default_extract_options();

    ExtractSiftFromImage(&im1, &sift_data1, &options);
    ExtractSiftFromImage(&im2, &sift_data2, &options);
    if (check_error("Extract (setup)")) { DeleteSiftData(&sift_data1); DeleteSiftData(&sift_data2); return false; }

    MatchSiftData(&sift_data1, &sift_data2);
    if (check_error("MatchSiftData")) { DeleteSiftData(&sift_data1); DeleteSiftData(&sift_data2); return false; }

    int matched = 0;
    for (int i = 0; i < sift_data1.numPts; i++)
        if (sift_data1.h_data[i].match >= 0)
            matched++;

    std::cout << "  Matched " << matched << " / " << sift_data1.numPts << " keypoints" << std::endl;

    bool pass = matched > 0;
    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;

    DeleteSiftData(&sift_data1);
    DeleteSiftData(&sift_data2);
    return pass;
}

// -- Test: FindHomography --------------------------------

static bool test_find_homography(const Image_t& im1, const Image_t& im2)
{
    std::cout << "[TEST] FindHomography" << std::endl;

    SiftData sift_data1, sift_data2;
    ExtractSiftOptions_t eo = default_extract_options();
    FindHomographyOptions_t ho = default_homography_options();

    ExtractSiftFromImage(&im1, &sift_data1, &eo);
    ExtractSiftFromImage(&im2, &sift_data2, &eo);
    MatchSiftData(&sift_data1, &sift_data2);
    if (check_error("Extract+Match (setup)")) { DeleteSiftData(&sift_data1); DeleteSiftData(&sift_data2); return false; }

    float homography[9];
    int num_matches = 0;
    FindHomography(&sift_data1, homography, &num_matches, &ho);
    if (check_error("FindHomography")) { DeleteSiftData(&sift_data1); DeleteSiftData(&sift_data2); return false; }

    std::cout << "  Inliers: " << num_matches << std::endl;
    print_homography(homography);

    bool pass = num_matches > 0 && homography_is_valid(homography);
    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;

    DeleteSiftData(&sift_data1);
    DeleteSiftData(&sift_data2);
    return pass;
}

// -- Test: WarpImages (CPU path) -------------------------

static bool test_warp_images_cpu(const Image_t& im1, const Image_t& im2)
{
    std::cout << "[TEST] WarpImages (CPU)" << std::endl;

    SiftData sd1, sd2;
    ExtractSiftOptions_t eo = default_extract_options();
    FindHomographyOptions_t ho = default_homography_options();

    ExtractSiftFromImage(&im1, &sd1, &eo);
    ExtractSiftFromImage(&im2, &sd2, &eo);
    MatchSiftData(&sd1, &sd2);

    float H[9];
    int nm = 0;
    FindHomography(&sd1, H, &nm, &ho);
    if (check_error("setup") || !homography_is_valid(H))
    {
        std::cout << "  [SKIP] no valid homography" << std::endl;
        DeleteSiftData(&sd1); DeleteSiftData(&sd2);
        return false;
    }

    Image_t w1 = {}, w2 = {};
    WarpImages(&im1, &im2, H, &w1, &w2, false);
    if (check_error("WarpImages (CPU)")) { DeleteSiftData(&sd1); DeleteSiftData(&sd2); return false; }

    std::cout << "  Warped size: " << w1.width_ << "x" << w1.height_ << std::endl;

    bool pass = w1.host_img_ != nullptr && w2.host_img_ != nullptr && w1.width_ > 0 && w1.height_ > 0;
    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;

    FreeImage(&w1);
    FreeImage(&w2);
    DeleteSiftData(&sd1);
    DeleteSiftData(&sd2);
    return pass;
}

// -- Test: WarpImages (GPU path) -------------------------

static bool test_warp_images_gpu(const Image_t& im1, const Image_t& im2)
{
    std::cout << "[TEST] WarpImages (GPU)" << std::endl;

    SiftData sd1, sd2;
    ExtractSiftOptions_t eo = default_extract_options();
    FindHomographyOptions_t ho = default_homography_options();

    ExtractSiftFromImage(&im1, &sd1, &eo);
    ExtractSiftFromImage(&im2, &sd2, &eo);
    MatchSiftData(&sd1, &sd2);

    float H[9];
    int nm = 0;
    FindHomography(&sd1, H, &nm, &ho);
    if (check_error("setup") || !homography_is_valid(H))
    {
        std::cout << "  [SKIP] no valid homography" << std::endl;
        DeleteSiftData(&sd1); DeleteSiftData(&sd2);
        return false;
    }

    Image_t w1 = {}, w2 = {};
    WarpImages(&im1, &im2, H, &w1, &w2, true);
    if (check_error("WarpImages (GPU)")) { DeleteSiftData(&sd1); DeleteSiftData(&sd2); return false; }

    std::cout << "  Warped size: " << w1.width_ << "x" << w1.height_ << std::endl;

    bool pass = w1.host_img_ != nullptr && w2.host_img_ != nullptr && w1.width_ > 0 && w1.height_ > 0;
    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;

    FreeImage(&w1);
    FreeImage(&w2);
    DeleteSiftData(&sd1);
    DeleteSiftData(&sd2);
    return pass;
}

// -- Test: WarpImages_GPU (returns device memory) --------

static bool test_warp_images_gpu_strided(const Image_t& im1, const Image_t& im2)
{
    std::cout << "[TEST] WarpImages_GPU" << std::endl;

    SiftData sd1, sd2;
    ExtractSiftOptions_t eo = default_extract_options();
    FindHomographyOptions_t ho = default_homography_options();

    ExtractSiftFromImage(&im1, &sd1, &eo);
    ExtractSiftFromImage(&im2, &sd2, &eo);
    MatchSiftData(&sd1, &sd2);

    float H[9];
    int nm = 0;
    FindHomography(&sd1, H, &nm, &ho);
    if (check_error("setup") || !homography_is_valid(H))
    {
        std::cout << "  [SKIP] no valid homography" << std::endl;
        DeleteSiftData(&sd1); DeleteSiftData(&sd2);
        return false;
    }

    ImageStrided_t w1 = {}, w2 = {};
    WarpImages_GPU(&im1, &im2, H, &w1, &w2);
    if (check_error("WarpImages_GPU")) { DeleteSiftData(&sd1); DeleteSiftData(&sd2); return false; }

    std::cout << "  Warped size: " << w1.width_ << "x" << w1.height_ << ", stride=" << w1.stride_ << std::endl;

    bool pass = w1.strided_img_ != nullptr && w2.strided_img_ != nullptr
             && w1.width_ > 0 && w1.height_ > 0 && w1.stride_ > 0;
    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;

    FreeImage_GPU(&w1);
    FreeImage_GPU(&w2);
    DeleteSiftData(&sd1);
    DeleteSiftData(&sd2);
    return pass;
}

// -- Test: SaveSiftData ----------------------------------

static bool test_save_sift_data(const Image_t& im1)
{
    std::cout << "[TEST] SaveSiftData" << std::endl;

    SiftData sd;
    ExtractSiftOptions_t eo = default_extract_options();

    ExtractSiftFromImage(&im1, &sd, &eo);
    if (check_error("Extract (setup)")) { DeleteSiftData(&sd); return false; }

    const char* tmp_file = "test_sift_output.json";
    SaveSiftData(tmp_file, &sd);

    FILE* f = fopen(tmp_file, "r");
    bool pass = (f != nullptr);
    if (f)
    {
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fclose(f);
        std::cout << "  Wrote " << sz << " bytes to " << tmp_file << std::endl;
        pass = sz > 0;
        remove(tmp_file);
    }
    else
    {
        std::cerr << "  File was not created" << std::endl;
    }

    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;

    DeleteSiftData(&sd);
    return pass;
}

// -- Test: ExtractAndMatchSift ---------------------------

static bool test_extract_and_match(const Image_t& im1, const Image_t& im2)
{
    std::cout << "[TEST] ExtractAndMatchSift" << std::endl;

    SiftData sd1, sd2;
    ExtractSiftOptions_t eo = default_extract_options();

    ExtractAndMatchSift(&im1, &im2, &sd1, &sd2, &eo);
    if (check_error("ExtractAndMatchSift")) { DeleteSiftData(&sd1); DeleteSiftData(&sd2); return false; }

    int matched = 0;
    for (int i = 0; i < sd1.numPts; i++)
        if (sd1.h_data[i].match >= 0)
            matched++;

    std::cout << "  Image 1: " << sd1.numPts << " keypoints, Image 2: " << sd2.numPts << " keypoints" << std::endl;
    std::cout << "  Matched: " << matched << std::endl;

    bool pass = sd1.numPts > 0 && sd2.numPts > 0 && matched > 0;
    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;

    DeleteSiftData(&sd1);
    DeleteSiftData(&sd2);
    return pass;
}

// -- Test: ExtractAndMatchAndFindHomography ---------------

static bool test_extract_match_homography(const Image_t& im1, const Image_t& im2)
{
    std::cout << "[TEST] ExtractAndMatchAndFindHomography" << std::endl;

    SiftData sd1, sd2;
    ExtractSiftOptions_t eo = default_extract_options();
    FindHomographyOptions_t ho = default_homography_options();

    float H[9];
    int nm = 0;

    ExtractAndMatchAndFindHomography(&im1, &im2, &sd1, &sd2, H, &nm, &eo, &ho);
    if (check_error("ExtractAndMatchAndFindHomography")) { DeleteSiftData(&sd1); DeleteSiftData(&sd2); return false; }

    std::cout << "  Inliers: " << nm << std::endl;
    print_homography(H);

    bool pass = nm > 0 && homography_is_valid(H);
    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;

    DeleteSiftData(&sd1);
    DeleteSiftData(&sd2);
    return pass;
}

// -- Test: ExtractAndMatchAndFindHomographyAndWarp --------

static bool test_extract_match_homography_warp(const Image_t& im1, const Image_t& im2)
{
    std::cout << "[TEST] ExtractAndMatchAndFindHomographyAndWarp" << std::endl;

    SiftData sd1, sd2;
    ExtractSiftOptions_t eo = default_extract_options();
    FindHomographyOptions_t ho = default_homography_options();

    float H[9];
    int nm = 0;
    Image_t w1 = {}, w2 = {};

    ExtractAndMatchAndFindHomographyAndWarp(&im1, &im2, &sd1, &sd2, H, &nm, &eo, &ho, &w1, &w2);
    if (check_error("ExtractAndMatchAndFindHomographyAndWarp"))
    {
        DeleteSiftData(&sd1); DeleteSiftData(&sd2);
        FreeImage(&w1); FreeImage(&w2);
        return false;
    }

    std::cout << "  Inliers: " << nm << std::endl;
    std::cout << "  Warped size: " << w1.width_ << "x" << w1.height_ << std::endl;
    print_homography(H);

    bool pass = nm > 0 && homography_is_valid(H)
             && w1.host_img_ != nullptr && w2.host_img_ != nullptr
             && w1.width_ > 0 && w1.height_ > 0;
    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;

    FreeImage(&w1);
    FreeImage(&w2);
    DeleteSiftData(&sd1);
    DeleteSiftData(&sd2);
    return pass;
}

// -- Test: ExtractAndMatchAndFindHomographyAndWarp_GPU ----

static bool test_extract_match_homography_warp_gpu(const Image_t& im1, const Image_t& im2)
{
    std::cout << "[TEST] ExtractAndMatchAndFindHomographyAndWarp_GPU" << std::endl;

    SiftData sd1, sd2;
    ExtractSiftOptions_t eo = default_extract_options();
    FindHomographyOptions_t ho = default_homography_options();

    float H[9];
    int nm = 0;
    ImageStrided_t w1 = {}, w2 = {};

    ExtractAndMatchAndFindHomographyAndWarp_GPU(&im1, &im2, &sd1, &sd2, H, &nm, &eo, &ho, &w1, &w2);
    if (check_error("ExtractAndMatchAndFindHomographyAndWarp_GPU"))
    {
        DeleteSiftData(&sd1); DeleteSiftData(&sd2);
        FreeImage_GPU(&w1); FreeImage_GPU(&w2);
        return false;
    }

    std::cout << "  Inliers: " << nm << std::endl;
    std::cout << "  Warped size: " << w1.width_ << "x" << w1.height_ << ", stride=" << w1.stride_ << std::endl;
    print_homography(H);

    bool pass = nm > 0 && homography_is_valid(H)
             && w1.strided_img_ != nullptr && w2.strided_img_ != nullptr
             && w1.width_ > 0 && w1.height_ > 0 && w1.stride_ > 0;
    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;

    FreeImage_GPU(&w1);
    FreeImage_GPU(&w2);
    DeleteSiftData(&sd1);
    DeleteSiftData(&sd2);
    return pass;
}

// -- Helpers for model-type comparison --------------------

static const char* model_name(int type)
{
    return type == CUSIFT_MODEL_SIMILARITY ? "SIMILARITY" : "HOMOGRAPHY";
}

static float homography_frobenius_diff(const float* A, const float* B)
{
    float sum = 0.0f;
    for (int i = 0; i < 9; i++)
    {
        float d = A[i] - B[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

// -- Test: FindHomography with both model types ----------

static bool test_find_homography_both_models(const Image_t& im1, const Image_t& im2)
{
    std::cout << "[TEST] FindHomography -- compare model types" << std::endl;

    SiftData sd1, sd2;
    ExtractSiftOptions_t eo = default_extract_options();

    ExtractSiftFromImage(&im1, &sd1, &eo);
    ExtractSiftFromImage(&im2, &sd2, &eo);
    MatchSiftData(&sd1, &sd2);
    if (check_error("Extract+Match (setup)")) { DeleteSiftData(&sd1); DeleteSiftData(&sd2); return false; }

    float H_homo[9], H_sim[9];
    int nm_homo = 0, nm_sim = 0;
    bool pass = true;

    // Run both model types
    int models[] = { CUSIFT_MODEL_HOMOGRAPHY, CUSIFT_MODEL_SIMILARITY };
    float* results[] = { H_homo, H_sim };
    int* counts[] = { &nm_homo, &nm_sim };

    for (int i = 0; i < 2; i++)
    {
        FindHomographyOptions_t ho = default_homography_options();
        ho.model_type_ = models[i];

        FindHomography(&sd1, results[i], counts[i], &ho);
        if (check_error("FindHomography"))
        {
            std::cerr << "  [FAIL] " << model_name(models[i]) << " failed" << std::endl;
            pass = false;
            continue;
        }
        std::cout << "  " << model_name(models[i]) << " -- inliers: " << *counts[i] << std::endl;
        print_homography(results[i]);
        if (*counts[i] == 0 || !homography_is_valid(results[i]))
        {
            std::cerr << "  [FAIL] " << model_name(models[i]) << " produced invalid result" << std::endl;
            pass = false;
        }
    }

    // Compare
    if (homography_is_valid(H_homo) && homography_is_valid(H_sim))
    {
        std::cout << "  Element-wise difference:" << std::endl;
        for (int r = 0; r < 3; r++)
            std::cout << "    [" << (H_homo[r*3+0] - H_sim[r*3+0])
                      << "  "   << (H_homo[r*3+1] - H_sim[r*3+1])
                      << "  "   << (H_homo[r*3+2] - H_sim[r*3+2]) << "]" << std::endl;
        float frob = homography_frobenius_diff(H_homo, H_sim);
        std::cout << "  Frobenius norm of difference: " << frob << std::endl;
    }

    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;

    DeleteSiftData(&sd1);
    DeleteSiftData(&sd2);
    return pass;
}

// -- Test: ExtractAndMatchAndFindHomography -- both models -

static bool test_extract_match_homography_both_models(const Image_t& im1, const Image_t& im2)
{
    std::cout << "[TEST] ExtractAndMatchAndFindHomography -- compare model types" << std::endl;

    float H_homo[9], H_sim[9];
    int nm_homo = 0, nm_sim = 0;
    bool pass = true;

    int models[] = { CUSIFT_MODEL_HOMOGRAPHY, CUSIFT_MODEL_SIMILARITY };
    float* results[] = { H_homo, H_sim };
    int* counts[] = { &nm_homo, &nm_sim };

    for (int i = 0; i < 2; i++)
    {
        SiftData sd1, sd2;
        ExtractSiftOptions_t eo = default_extract_options();
        FindHomographyOptions_t ho = default_homography_options();
        ho.model_type_ = models[i];

        ExtractAndMatchAndFindHomography(&im1, &im2, &sd1, &sd2, results[i], counts[i], &eo, &ho);
        if (check_error("ExtractAndMatchAndFindHomography"))
        {
            std::cerr << "  [FAIL] " << model_name(models[i]) << " failed" << std::endl;
            pass = false;
        }
        else
        {
            std::cout << "  " << model_name(models[i]) << " -- inliers: " << *counts[i] << std::endl;
            print_homography(results[i]);
            if (*counts[i] == 0 || !homography_is_valid(results[i]))
            {
                std::cerr << "  [FAIL] " << model_name(models[i]) << " produced invalid result" << std::endl;
                pass = false;
            }
        }
        DeleteSiftData(&sd1);
        DeleteSiftData(&sd2);
    }

    if (homography_is_valid(H_homo) && homography_is_valid(H_sim))
    {
        float frob = homography_frobenius_diff(H_homo, H_sim);
        std::cout << "  Frobenius norm of difference: " << frob << std::endl;
    }

    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;
    return pass;
}

// -- Test: Full pipeline (warp) -- both models ------------

static bool test_extract_match_homography_warp_both_models(const Image_t& im1, const Image_t& im2)
{
    std::cout << "[TEST] ExtractAndMatchAndFindHomographyAndWarp -- compare model types" << std::endl;

    float H_homo[9], H_sim[9];
    int nm_homo = 0, nm_sim = 0;
    bool pass = true;

    int models[] = { CUSIFT_MODEL_HOMOGRAPHY, CUSIFT_MODEL_SIMILARITY };
    float* results[] = { H_homo, H_sim };
    int* counts[] = { &nm_homo, &nm_sim };

    for (int i = 0; i < 2; i++)
    {
        SiftData sd1, sd2;
        ExtractSiftOptions_t eo = default_extract_options();
        FindHomographyOptions_t ho = default_homography_options();
        ho.model_type_ = models[i];

        Image_t w1 = {}, w2 = {};
        ExtractAndMatchAndFindHomographyAndWarp(&im1, &im2, &sd1, &sd2, results[i], counts[i], &eo, &ho, &w1, &w2);
        if (check_error("ExtractAndMatchAndFindHomographyAndWarp"))
        {
            std::cerr << "  [FAIL] " << model_name(models[i]) << " failed" << std::endl;
            pass = false;
        }
        else
        {
            std::cout << "  " << model_name(models[i]) << " -- inliers: " << *counts[i]
                      << ", warped: " << w1.width_ << "x" << w1.height_ << std::endl;
            print_homography(results[i]);
            if (*counts[i] == 0 || !homography_is_valid(results[i])
                || w1.host_img_ == nullptr || w1.width_ == 0)
            {
                std::cerr << "  [FAIL] " << model_name(models[i]) << " produced invalid result" << std::endl;
                pass = false;
            }
        }
        FreeImage(&w1);
        FreeImage(&w2);
        DeleteSiftData(&sd1);
        DeleteSiftData(&sd2);
    }

    if (homography_is_valid(H_homo) && homography_is_valid(H_sim))
    {
        float frob = homography_frobenius_diff(H_homo, H_sim);
        std::cout << "  Frobenius norm of difference: " << frob << std::endl;
    }

    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;
    return pass;
}

// -- Test: VRAM estimation functions ---------------------

static bool test_estimate_vram(const Image_t& im1, const Image_t& im2)
{
    std::cout << "[TEST] VRAM Estimation Functions" << std::endl;

    ExtractSiftOptions_t eo = default_extract_options();
    FindHomographyOptions_t ho = default_homography_options();

    bool pass = true;

    // EstimateVramExtractSift
    size_t extract_vram = EstimateVramExtractSift(im1.width_, im1.height_, &eo);
    std::cout << "  EstimateVramExtractSift(" << im1.width_ << "x" << im1.height_ << "): "
              << extract_vram << " bytes (" << (extract_vram / (1024.0 * 1024.0)) << " MB)" << std::endl;
    if (extract_vram == 0) { std::cerr << "  [FAIL] extract estimate is 0" << std::endl; pass = false; }

    // EstimateVramMatchSift
    size_t match_vram = EstimateVramMatchSift(eo.max_keypoints_, eo.max_keypoints_);
    std::cout << "  EstimateVramMatchSift(" << eo.max_keypoints_ << ", " << eo.max_keypoints_ << "): "
              << match_vram << " bytes (" << (match_vram / (1024.0 * 1024.0)) << " MB)" << std::endl;
    if (match_vram == 0) { std::cerr << "  [FAIL] match estimate is 0" << std::endl; pass = false; }

    // EstimateVramFindHomography
    size_t homo_vram = EstimateVramFindHomography(eo.max_keypoints_, &ho);
    std::cout << "  EstimateVramFindHomography(" << eo.max_keypoints_ << "): "
              << homo_vram << " bytes (" << (homo_vram / (1024.0 * 1024.0)) << " MB)" << std::endl;
    if (homo_vram == 0) { std::cerr << "  [FAIL] homography estimate is 0" << std::endl; pass = false; }

    // EstimateVramWarpImages
    size_t warp_vram = EstimateVramWarpImages(im1.width_, im1.height_, im2.width_, im2.height_);
    std::cout << "  EstimateVramWarpImages(" << im1.width_ << "x" << im1.height_
              << ", " << im2.width_ << "x" << im2.height_ << "): "
              << warp_vram << " bytes (" << (warp_vram / (1024.0 * 1024.0)) << " MB)" << std::endl;
    if (warp_vram == 0) { std::cerr << "  [FAIL] warp estimate is 0" << std::endl; pass = false; }

    // EstimateVramFullPipeline
    size_t full_vram = EstimateVramFullPipeline(im1.width_, im1.height_, im2.width_, im2.height_, &eo, &ho);
    std::cout << "  EstimateVramFullPipeline: "
              << full_vram << " bytes (" << (full_vram / (1024.0 * 1024.0)) << " MB)" << std::endl;
    if (full_vram == 0) { std::cerr << "  [FAIL] full pipeline estimate is 0" << std::endl; pass = false; }

    // Sanity: full pipeline should be >= extraction (since extraction is typically the peak)
    if (full_vram < extract_vram)
    {
        std::cerr << "  [FAIL] full pipeline estimate (" << full_vram
                  << ") < extract estimate (" << extract_vram << ")" << std::endl;
        pass = false;
    }

    // Sanity: match estimate should equal 2 * sizeof(SiftPoint) * max_keypoints
    size_t expected_match = 2 * sizeof(SiftPoint) * eo.max_keypoints_;
    if (match_vram != expected_match)
    {
        std::cerr << "  [FAIL] match estimate (" << match_vram
                  << ") != expected (" << expected_match << ")" << std::endl;
        pass = false;
    }

    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;
    return pass;
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

    Image_t im1 = { image1.data(), width1, height1 };
    Image_t im2 = { image2.data(), width2, height2 };

    InitializeCudaSift();

    int passed = 0, failed = 0, total = 0;

    auto run = [&](bool result) { total++; result ? passed++ : failed++; std::cout << std::endl; };

    run(test_extract(im1, im2));
    run(test_match(im1, im2));
    run(test_find_homography(im1, im2));
    run(test_warp_images_cpu(im1, im2));
    run(test_warp_images_gpu(im1, im2));
    run(test_warp_images_gpu_strided(im1, im2));
    run(test_save_sift_data(im1));
    run(test_extract_and_match(im1, im2));
    run(test_extract_match_homography(im1, im2));
    run(test_extract_match_homography_warp(im1, im2));
    run(test_extract_match_homography_warp_gpu(im1, im2));
    run(test_find_homography_both_models(im1, im2));
    run(test_extract_match_homography_both_models(im1, im2));
    run(test_extract_match_homography_warp_both_models(im1, im2));
    run(test_estimate_vram(im1, im2));

    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed, " << total << " total" << std::endl;

    return failed > 0 ? 1 : 0;
}







