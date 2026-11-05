#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "cusift.h"
#include <iostream>
#include <fstream>
#include <string>
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
    cusift_error_record err;
    if (cusift_last_error(&err))
    {
        std::cerr << "  [FAIL] " << test_name << ": " << err.function << ": " << err.message << std::endl;
        return true;
    }
    return false;
}

// Message of the last error, or "" when the last call succeeded.
static std::string last_error_message()
{
    cusift_error_record err;
    return cusift_last_error(&err) ? std::string(err.message) : std::string();
}

static cusift_extract_options default_extract_options()
{
    cusift_extract_options opts;
    opts.contrast_threshold = 2.0f;
    opts.lowest_scale = 0.0f;
    opts.highest_scale = std::numeric_limits<float>::infinity();
    opts.edge_threshold = 10.0f;
    opts.initial_blur = 1.0f;
    opts.max_keypoints = 10000;
    opts.num_octaves = 5;
    opts.scale_suppression_radius = 0.0f;
    return opts;
}

static cusift_homography_options default_homography_options()
{
    cusift_homography_options opts;
    opts.ransac_iterations = 10000;
    opts.min_score = 0.0f;
    opts.max_ambiguity = 1.0f;
    opts.inlier_threshold = 3.0f;
    opts.refine_iterations = 5;
    opts.refine_min_score = 0.0f;
    opts.refine_max_ambiguity = 1.0f;
    opts.refine_inlier_threshold = 2.0f;
    opts.seed = 42;
    opts.model = CUSIFT_MODEL_HOMOGRAPHY;
    opts.attempts = 1;
    opts.goal = CUSIFT_GOAL_MAX_INLIERS;
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

// -- Test: cusift_extract --------------------------

static bool test_extract(const cusift_image& im1, const cusift_image& im2)
{
    std::cout << "[TEST] cusift_extract" << std::endl;

    cusift_keypoints sift_data1{}, sift_data2{};
    cusift_extract_options options = default_extract_options();

    cusift_extract(&im1, &sift_data1, &options);
    if (check_error("cusift_extract (image1)")) return false;

    cusift_extract(&im2, &sift_data2, &options);
    if (check_error("cusift_extract (image2)")) return false;

    std::cout << "  Image 1: " << sift_data1.count << " keypoints" << std::endl;
    std::cout << "  Image 2: " << sift_data2.count << " keypoints" << std::endl;

    bool pass = sift_data1.count > 0 && sift_data2.count > 0;
    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;

    cusift_keypoints_free(&sift_data1);
    cusift_keypoints_free(&sift_data2);
    return pass;
}

// -- Test: cusift_match ---------------------------------

static bool test_match(const cusift_image& im1, const cusift_image& im2)
{
    std::cout << "[TEST] cusift_match" << std::endl;

    cusift_keypoints sift_data1{}, sift_data2{};
    cusift_extract_options options = default_extract_options();

    cusift_extract(&im1, &sift_data1, &options);
    cusift_extract(&im2, &sift_data2, &options);
    if (check_error("Extract (setup)")) { cusift_keypoints_free(&sift_data1); cusift_keypoints_free(&sift_data2); return false; }

    cusift_match(&sift_data1, &sift_data2);
    if (check_error("cusift_match")) { cusift_keypoints_free(&sift_data1); cusift_keypoints_free(&sift_data2); return false; }

    int matched = 0;
    for (int i = 0; i < sift_data1.count; i++)
        if (sift_data1.host[i].match >= 0)
            matched++;

    std::cout << "  Matched " << matched << " / " << sift_data1.count << " keypoints" << std::endl;

    bool pass = matched > 0;
    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;

    cusift_keypoints_free(&sift_data1);
    cusift_keypoints_free(&sift_data2);
    return pass;
}

// -- Test: cusift_estimate_homography --------------------------------

static bool test_find_homography(const cusift_image& im1, const cusift_image& im2)
{
    std::cout << "[TEST] cusift_estimate_homography" << std::endl;

    cusift_keypoints sift_data1{}, sift_data2{};
    cusift_extract_options eo = default_extract_options();
    cusift_homography_options ho = default_homography_options();

    cusift_extract(&im1, &sift_data1, &eo);
    cusift_extract(&im2, &sift_data2, &eo);
    cusift_match(&sift_data1, &sift_data2);
    if (check_error("Extract+Match (setup)")) { cusift_keypoints_free(&sift_data1); cusift_keypoints_free(&sift_data2); return false; }

    float homography[9];
    int num_matches = 0;
    cusift_estimate_homography(&sift_data1, homography, &num_matches, &ho);
    if (check_error("cusift_estimate_homography")) { cusift_keypoints_free(&sift_data1); cusift_keypoints_free(&sift_data2); return false; }

    std::cout << "  Inliers: " << num_matches << std::endl;
    print_homography(homography);

    bool pass = num_matches > 0 && homography_is_valid(homography);
    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;

    cusift_keypoints_free(&sift_data1);
    cusift_keypoints_free(&sift_data2);
    return pass;
}

// -- Test: cusift_warp (CPU path) -------------------------

static bool test_warp_images_cpu(const cusift_image& im1, const cusift_image& im2)
{
    std::cout << "[TEST] cusift_warp (CPU)" << std::endl;

    cusift_keypoints sd1{}, sd2{};
    cusift_extract_options eo = default_extract_options();
    cusift_homography_options ho = default_homography_options();

    cusift_extract(&im1, &sd1, &eo);
    cusift_extract(&im2, &sd2, &eo);
    cusift_match(&sd1, &sd2);

    float H[9];
    int nm = 0;
    cusift_estimate_homography(&sd1, H, &nm, &ho);
    if (check_error("setup") || !homography_is_valid(H))
    {
        std::cout << "  [SKIP] no valid homography" << std::endl;
        cusift_keypoints_free(&sd1); cusift_keypoints_free(&sd2);
        return false;
    }

    cusift_image w1 = {}, w2 = {};
    cusift_warp(&im1, &im2, H, &w1, &w2, false);
    if (check_error("cusift_warp (CPU)")) { cusift_keypoints_free(&sd1); cusift_keypoints_free(&sd2); return false; }

    std::cout << "  Warped size: " << w1.width << "x" << w1.height << std::endl;

    bool pass = w1.pixels != nullptr && w2.pixels != nullptr && w1.width > 0 && w1.height > 0;
    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;

    cusift_image_free(&w1);
    cusift_image_free(&w2);
    cusift_keypoints_free(&sd1);
    cusift_keypoints_free(&sd2);
    return pass;
}

// -- Test: cusift_warp (GPU path) -------------------------

static bool test_warp_images_gpu(const cusift_image& im1, const cusift_image& im2)
{
    std::cout << "[TEST] cusift_warp (GPU)" << std::endl;

    cusift_keypoints sd1{}, sd2{};
    cusift_extract_options eo = default_extract_options();
    cusift_homography_options ho = default_homography_options();

    cusift_extract(&im1, &sd1, &eo);
    cusift_extract(&im2, &sd2, &eo);
    cusift_match(&sd1, &sd2);

    float H[9];
    int nm = 0;
    cusift_estimate_homography(&sd1, H, &nm, &ho);
    if (check_error("setup") || !homography_is_valid(H))
    {
        std::cout << "  [SKIP] no valid homography" << std::endl;
        cusift_keypoints_free(&sd1); cusift_keypoints_free(&sd2);
        return false;
    }

    cusift_image w1 = {}, w2 = {};
    cusift_warp(&im1, &im2, H, &w1, &w2, true);
    if (check_error("cusift_warp (GPU)")) { cusift_keypoints_free(&sd1); cusift_keypoints_free(&sd2); return false; }

    std::cout << "  Warped size: " << w1.width << "x" << w1.height << std::endl;

    bool pass = w1.pixels != nullptr && w2.pixels != nullptr && w1.width > 0 && w1.height > 0;
    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;

    cusift_image_free(&w1);
    cusift_image_free(&w2);
    cusift_keypoints_free(&sd1);
    cusift_keypoints_free(&sd2);
    return pass;
}

// -- Test: cusift_warp_to_device (returns device memory) --------

static bool test_warp_images_gpu_strided(const cusift_image& im1, const cusift_image& im2)
{
    std::cout << "[TEST] cusift_warp_to_device" << std::endl;

    cusift_keypoints sd1{}, sd2{};
    cusift_extract_options eo = default_extract_options();
    cusift_homography_options ho = default_homography_options();

    cusift_extract(&im1, &sd1, &eo);
    cusift_extract(&im2, &sd2, &eo);
    cusift_match(&sd1, &sd2);

    float H[9];
    int nm = 0;
    cusift_estimate_homography(&sd1, H, &nm, &ho);
    if (check_error("setup") || !homography_is_valid(H))
    {
        std::cout << "  [SKIP] no valid homography" << std::endl;
        cusift_keypoints_free(&sd1); cusift_keypoints_free(&sd2);
        return false;
    }

    cusift_device_image w1 = {}, w2 = {};
    cusift_warp_to_device(&im1, &im2, H, &w1, &w2);
    if (check_error("cusift_warp_to_device")) { cusift_keypoints_free(&sd1); cusift_keypoints_free(&sd2); return false; }

    std::cout << "  Warped size: " << w1.width << "x" << w1.height << ", stride=" << w1.pitch_bytes << std::endl;

    bool pass = w1.device_pixels != nullptr && w2.device_pixels != nullptr
             && w1.width > 0 && w1.height > 0 && w1.pitch_bytes > 0;
    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;

    cusift_device_image_free(&w1);
    cusift_device_image_free(&w2);
    cusift_keypoints_free(&sd1);
    cusift_keypoints_free(&sd2);
    return pass;
}

// -- Test: cusift_keypoints_save_json ----------------------------------

static bool test_save_sift_data(const cusift_image& im1)
{
    std::cout << "[TEST] cusift_keypoints_save_json" << std::endl;

    cusift_keypoints sd{};
    cusift_extract_options eo = default_extract_options();

    cusift_extract(&im1, &sd, &eo);
    if (check_error("Extract (setup)")) { cusift_keypoints_free(&sd); return false; }

    const char* tmp_file = "test_sift_output.json";
    cusift_keypoints_save_json(tmp_file, &sd);

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

    cusift_keypoints_free(&sd);
    return pass;
}

// -- Test: cusift_extract_and_match ---------------------------

static bool test_extract_and_match(const cusift_image& im1, const cusift_image& im2)
{
    std::cout << "[TEST] cusift_extract_and_match" << std::endl;

    cusift_keypoints sd1{}, sd2{};
    cusift_extract_options eo = default_extract_options();

    cusift_extract_and_match(&im1, &im2, &sd1, &sd2, &eo);
    if (check_error("cusift_extract_and_match")) { cusift_keypoints_free(&sd1); cusift_keypoints_free(&sd2); return false; }

    int matched = 0;
    for (int i = 0; i < sd1.count; i++)
        if (sd1.host[i].match >= 0)
            matched++;

    std::cout << "  Image 1: " << sd1.count << " keypoints, Image 2: " << sd2.count << " keypoints" << std::endl;
    std::cout << "  Matched: " << matched << std::endl;

    bool pass = sd1.count > 0 && sd2.count > 0 && matched > 0;
    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;

    cusift_keypoints_free(&sd1);
    cusift_keypoints_free(&sd2);
    return pass;
}

// -- Test: cusift_align ---------------

static bool test_extract_match_homography(const cusift_image& im1, const cusift_image& im2)
{
    std::cout << "[TEST] cusift_align" << std::endl;

    cusift_keypoints sd1{}, sd2{};
    cusift_extract_options eo = default_extract_options();
    cusift_homography_options ho = default_homography_options();

    float H[9];
    int nm = 0;

    cusift_align(&im1, &im2, &sd1, &sd2, &eo, &ho, H, &nm);
    if (check_error("cusift_align")) { cusift_keypoints_free(&sd1); cusift_keypoints_free(&sd2); return false; }

    std::cout << "  Inliers: " << nm << std::endl;
    print_homography(H);

    bool pass = nm > 0 && homography_is_valid(H);
    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;

    cusift_keypoints_free(&sd1);
    cusift_keypoints_free(&sd2);
    return pass;
}

// -- Test: cusift_align_and_warp --------

static bool test_extract_match_homography_warp(const cusift_image& im1, const cusift_image& im2)
{
    std::cout << "[TEST] cusift_align_and_warp" << std::endl;

    cusift_keypoints sd1{}, sd2{};
    cusift_extract_options eo = default_extract_options();
    cusift_homography_options ho = default_homography_options();

    float H[9];
    int nm = 0;
    cusift_image w1 = {}, w2 = {};

    cusift_align_and_warp(&im1, &im2, &sd1, &sd2, &eo, &ho, H, &nm, &w1, &w2);
    if (check_error("cusift_align_and_warp"))
    {
        cusift_keypoints_free(&sd1); cusift_keypoints_free(&sd2);
        cusift_image_free(&w1); cusift_image_free(&w2);
        return false;
    }

    std::cout << "  Inliers: " << nm << std::endl;
    std::cout << "  Warped size: " << w1.width << "x" << w1.height << std::endl;
    print_homography(H);

    bool pass = nm > 0 && homography_is_valid(H)
             && w1.pixels != nullptr && w2.pixels != nullptr
             && w1.width > 0 && w1.height > 0;
    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;

    cusift_image_free(&w1);
    cusift_image_free(&w2);
    cusift_keypoints_free(&sd1);
    cusift_keypoints_free(&sd2);
    return pass;
}

// -- Test: cusift_align_and_warp_to_device ----

static bool test_extract_match_homography_warp_gpu(const cusift_image& im1, const cusift_image& im2)
{
    std::cout << "[TEST] cusift_align_and_warp_to_device" << std::endl;

    cusift_keypoints sd1{}, sd2{};
    cusift_extract_options eo = default_extract_options();
    cusift_homography_options ho = default_homography_options();

    float H[9];
    int nm = 0;
    cusift_device_image w1 = {}, w2 = {};

    cusift_align_and_warp_to_device(&im1, &im2, &sd1, &sd2, &eo, &ho, H, &nm, &w1, &w2);
    if (check_error("cusift_align_and_warp_to_device"))
    {
        cusift_keypoints_free(&sd1); cusift_keypoints_free(&sd2);
        cusift_device_image_free(&w1); cusift_device_image_free(&w2);
        return false;
    }

    std::cout << "  Inliers: " << nm << std::endl;
    std::cout << "  Warped size: " << w1.width << "x" << w1.height << ", stride=" << w1.pitch_bytes << std::endl;
    print_homography(H);

    bool pass = nm > 0 && homography_is_valid(H)
             && w1.device_pixels != nullptr && w2.device_pixels != nullptr
             && w1.width > 0 && w1.height > 0 && w1.pitch_bytes > 0;
    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;

    cusift_device_image_free(&w1);
    cusift_device_image_free(&w2);
    cusift_keypoints_free(&sd1);
    cusift_keypoints_free(&sd2);
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

// -- Test: cusift_estimate_homography with both model types ----------

static bool test_find_homography_both_models(const cusift_image& im1, const cusift_image& im2)
{
    std::cout << "[TEST] cusift_estimate_homography -- compare model types" << std::endl;

    cusift_keypoints sd1{}, sd2{};
    cusift_extract_options eo = default_extract_options();

    cusift_extract(&im1, &sd1, &eo);
    cusift_extract(&im2, &sd2, &eo);
    cusift_match(&sd1, &sd2);
    if (check_error("Extract+Match (setup)")) { cusift_keypoints_free(&sd1); cusift_keypoints_free(&sd2); return false; }

    float H_homo[9], H_sim[9];
    int nm_homo = 0, nm_sim = 0;
    bool pass = true;

    // Run both model types
    int models[] = { CUSIFT_MODEL_HOMOGRAPHY, CUSIFT_MODEL_SIMILARITY };
    float* results[] = { H_homo, H_sim };
    int* counts[] = { &nm_homo, &nm_sim };

    for (int i = 0; i < 2; i++)
    {
        cusift_homography_options ho = default_homography_options();
        ho.model = models[i];

        cusift_estimate_homography(&sd1, results[i], counts[i], &ho);
        if (check_error("cusift_estimate_homography"))
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

    cusift_keypoints_free(&sd1);
    cusift_keypoints_free(&sd2);
    return pass;
}

// -- Test: cusift_align -- both models -

static bool test_extract_match_homography_both_models(const cusift_image& im1, const cusift_image& im2)
{
    std::cout << "[TEST] cusift_align -- compare model types" << std::endl;

    float H_homo[9], H_sim[9];
    int nm_homo = 0, nm_sim = 0;
    bool pass = true;

    int models[] = { CUSIFT_MODEL_HOMOGRAPHY, CUSIFT_MODEL_SIMILARITY };
    float* results[] = { H_homo, H_sim };
    int* counts[] = { &nm_homo, &nm_sim };

    for (int i = 0; i < 2; i++)
    {
        cusift_keypoints sd1{}, sd2{};
        cusift_extract_options eo = default_extract_options();
        cusift_homography_options ho = default_homography_options();
        ho.model = models[i];

        cusift_align(&im1, &im2, &sd1, &sd2, &eo, &ho, results[i], counts[i]);
        if (check_error("cusift_align"))
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
        cusift_keypoints_free(&sd1);
        cusift_keypoints_free(&sd2);
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

static bool test_extract_match_homography_warp_both_models(const cusift_image& im1, const cusift_image& im2)
{
    std::cout << "[TEST] cusift_align_and_warp -- compare model types" << std::endl;

    float H_homo[9], H_sim[9];
    int nm_homo = 0, nm_sim = 0;
    bool pass = true;

    int models[] = { CUSIFT_MODEL_HOMOGRAPHY, CUSIFT_MODEL_SIMILARITY };
    float* results[] = { H_homo, H_sim };
    int* counts[] = { &nm_homo, &nm_sim };

    for (int i = 0; i < 2; i++)
    {
        cusift_keypoints sd1{}, sd2{};
        cusift_extract_options eo = default_extract_options();
        cusift_homography_options ho = default_homography_options();
        ho.model = models[i];

        cusift_image w1 = {}, w2 = {};
        cusift_align_and_warp(&im1, &im2, &sd1, &sd2, &eo, &ho, results[i], counts[i], &w1, &w2);
        if (check_error("cusift_align_and_warp"))
        {
            std::cerr << "  [FAIL] " << model_name(models[i]) << " failed" << std::endl;
            pass = false;
        }
        else
        {
            std::cout << "  " << model_name(models[i]) << " -- inliers: " << *counts[i]
                      << ", warped: " << w1.width << "x" << w1.height << std::endl;
            print_homography(results[i]);
            if (*counts[i] == 0 || !homography_is_valid(results[i])
                || w1.pixels == nullptr || w1.width == 0)
            {
                std::cerr << "  [FAIL] " << model_name(models[i]) << " produced invalid result" << std::endl;
                pass = false;
            }
        }
        cusift_image_free(&w1);
        cusift_image_free(&w2);
        cusift_keypoints_free(&sd1);
        cusift_keypoints_free(&sd2);
    }

    if (homography_is_valid(H_homo) && homography_is_valid(H_sim))
    {
        float frob = homography_frobenius_diff(H_homo, H_sim);
        std::cout << "  Frobenius norm of difference: " << frob << std::endl;
    }

    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;
    return pass;
}

// -- Test: cusift_align --------

static bool test_extract_match_homography_multi(const cusift_image& im1, const cusift_image& im2)
{
    std::cout << "[TEST] cusift_align" << std::endl;

    bool pass = true;

    // Sub-test 1: MAX_INLIERS goal with multiple attempts
    {
        constexpr int num_attempts = 20;
        std::cout << "  Goal: MAX_INLIERS, " << num_attempts << " attempts" << std::endl;
        cusift_keypoints sd1{}, sd2{};
        cusift_extract_options eo = default_extract_options();
        cusift_homography_options ho = default_homography_options();
        ho.seed = 0; // non-deterministic seeds across attempts

        float H[9];
        int nm = 0;

        cusift_homography_options ho_k = ho;
        ho_k.attempts = num_attempts;
        ho_k.goal = CUSIFT_GOAL_MAX_INLIERS;
        cusift_align(&im1, &im2, &sd1, &sd2, &eo, &ho_k, H, &nm);
        if (check_error("Multi (MAX_INLIERS)"))
        {
            pass = false;
        }
        else
        {
            std::cout << "    Inliers: " << nm << std::endl;
            print_homography(H);
            if (nm <= 0 || !homography_is_valid(H))
            {
                std::cerr << "    [FAIL] invalid result" << std::endl;
                pass = false;
            }
        }
        cusift_keypoints_free(&sd1);
        cusift_keypoints_free(&sd2);
    }

    // Sub-test 2: MIN_EYE_DIFF goal with multiple attempts
    {
        constexpr int num_attempts = 20;
        std::cout << "  Goal: MIN_EYE_DIFF, " << num_attempts << " attempts" << std::endl;
        cusift_keypoints sd1{}, sd2{};
        cusift_extract_options eo = default_extract_options();
        cusift_homography_options ho = default_homography_options();
        ho.seed = 0;

        float H[9];
        int nm = 0;

        cusift_homography_options ho_k = ho;
        ho_k.attempts = num_attempts;
        ho_k.goal = CUSIFT_GOAL_MIN_IDENTITY_DISTANCE;
        cusift_align(&im1, &im2, &sd1, &sd2, &eo, &ho_k, H, &nm);
        if (check_error("Multi (MIN_EYE_DIFF)"))
        {
            pass = false;
        }
        else
        {
            std::cout << "    Inliers: " << nm << std::endl;
            print_homography(H);
            if (nm <= 0 || !homography_is_valid(H))
            {
                std::cerr << "    [FAIL] invalid result" << std::endl;
                pass = false;
            }
        }
        cusift_keypoints_free(&sd1);
        cusift_keypoints_free(&sd2);
    }

    // Sub-test 3: single attempt should behave like the non-Multi version
    {
        std::cout << "  Single attempt (attempts=1)" << std::endl;
        cusift_keypoints sd1{}, sd2{};
        cusift_extract_options eo = default_extract_options();
        cusift_homography_options ho = default_homography_options();

        float H[9];
        int nm = 0;

        cusift_homography_options ho_k = ho;
        ho_k.attempts = 1;
        ho_k.goal = CUSIFT_GOAL_MAX_INLIERS;
        cusift_align(&im1, &im2, &sd1, &sd2, &eo, &ho_k, H, &nm);
        if (check_error("Multi (single attempt)"))
        {
            pass = false;
        }
        else
        {
            std::cout << "    Inliers: " << nm << std::endl;
            print_homography(H);
            if (nm <= 0 || !homography_is_valid(H))
            {
                std::cerr << "    [FAIL] invalid result" << std::endl;
                pass = false;
            }
        }
        cusift_keypoints_free(&sd1);
        cusift_keypoints_free(&sd2);
    }

    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;
    return pass;
}

// -- Test: cusift_align_and_warp -

static bool test_extract_match_homography_multi_warp(const cusift_image& im1, const cusift_image& im2)
{
    std::cout << "[TEST] cusift_align_and_warp" << std::endl;

    bool pass = true;

    {
        constexpr int num_attempts = 20;
        std::cout << "  Goal: MAX_INLIERS, " << num_attempts << " attempts" << std::endl;
        cusift_keypoints sd1{}, sd2{};
        cusift_extract_options eo = default_extract_options();
        cusift_homography_options ho = default_homography_options();
        ho.seed = 0;

        float H[9];
        int nm = 0;
        cusift_image w1 = {}, w2 = {};

        cusift_homography_options ho_k = ho;
        ho_k.attempts = 5;
        ho_k.goal = CUSIFT_GOAL_MAX_INLIERS;
        cusift_align_and_warp(&im1, &im2, &sd1, &sd2, &eo, &ho_k, H, &nm, &w1, &w2);
        if (check_error("Multi_AndWarp (MAX_INLIERS)"))
        {
            pass = false;
        }
        else
        {
            std::cout << "    Inliers: " << nm << std::endl;
            std::cout << "    Warped size: " << w1.width << "x" << w1.height << std::endl;
            print_homography(H);
            if (nm <= 0 || !homography_is_valid(H)
                || w1.pixels == nullptr || w2.pixels == nullptr
                || w1.width <= 0 || w1.height <= 0)
            {
                std::cerr << "    [FAIL] invalid result" << std::endl;
                pass = false;
            }
        }
        cusift_image_free(&w1);
        cusift_image_free(&w2);
        cusift_keypoints_free(&sd1);
        cusift_keypoints_free(&sd2);
    }

    {
        constexpr int num_attempts = 20;
        std::cout << "  Goal: MIN_EYE_DIFF, " << num_attempts << " attempts" << std::endl;
        cusift_keypoints sd1{}, sd2{};
        cusift_extract_options eo = default_extract_options();
        cusift_homography_options ho = default_homography_options();
        ho.seed = 0;

        float H[9];
        int nm = 0;
        cusift_image w1 = {}, w2 = {};

        cusift_homography_options ho_k = ho;
        ho_k.attempts = num_attempts;
        ho_k.goal = CUSIFT_GOAL_MIN_IDENTITY_DISTANCE;
        cusift_align_and_warp(&im1, &im2, &sd1, &sd2, &eo, &ho_k, H, &nm, &w1, &w2);
        if (check_error("Multi_AndWarp (MIN_EYE_DIFF)"))
        {
            pass = false;
        }
        else
        {
            std::cout << "    Inliers: " << nm << std::endl;
            std::cout << "    Warped size: " << w1.width << "x" << w1.height << std::endl;
            print_homography(H);
            if (nm <= 0 || !homography_is_valid(H)
                || w1.pixels == nullptr || w2.pixels == nullptr
                || w1.width <= 0 || w1.height <= 0)
            {
                std::cerr << "    [FAIL] invalid result" << std::endl;
                pass = false;
            }
        }
        cusift_image_free(&w1);
        cusift_image_free(&w2);
        cusift_keypoints_free(&sd1);
        cusift_keypoints_free(&sd2);
    }

    {
        std::cout << "  Single attempt (attempts=1)" << std::endl;
        cusift_keypoints sd1{}, sd2{};
        cusift_extract_options eo = default_extract_options();
        cusift_homography_options ho = default_homography_options();

        float H[9];
        int nm = 0;
        cusift_image w1 = {}, w2 = {};

        cusift_homography_options ho_k = ho;
        ho_k.attempts = 1;
        ho_k.goal = CUSIFT_GOAL_MAX_INLIERS;
        cusift_align_and_warp(&im1, &im2, &sd1, &sd2, &eo, &ho_k, H, &nm, &w1, &w2);
        if (check_error("Multi_AndWarp (single attempt)"))
        {
            pass = false;
        }
        else
        {
            std::cout << "    Inliers: " << nm << std::endl;
            std::cout << "    Warped size: " << w1.width << "x" << w1.height << std::endl;
            print_homography(H);
            if (nm <= 0 || !homography_is_valid(H)
                || w1.pixels == nullptr || w2.pixels == nullptr
                || w1.width <= 0 || w1.height <= 0)
            {
                std::cerr << "    [FAIL] invalid result" << std::endl;
                pass = false;
            }
        }
        cusift_image_free(&w1);
        cusift_image_free(&w2);
        cusift_keypoints_free(&sd1);
        cusift_keypoints_free(&sd2);
    }

    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;
    return pass;
}

// -- Test: VRAM estimation functions ---------------------

static bool test_estimate_vram(const cusift_image& im1, const cusift_image& im2)
{
    std::cout << "[TEST] VRAM Estimation Functions" << std::endl;

    cusift_extract_options eo = default_extract_options();
    cusift_homography_options ho = default_homography_options();

    bool pass = true;

    // cusift_memory_extract
    size_t extract_vram = cusift_memory_extract(im1.width, im1.height, &eo);
    std::cout << "  cusift_memory_extract(" << im1.width << "x" << im1.height << "): "
              << extract_vram << " bytes (" << (extract_vram / (1024.0 * 1024.0)) << " MB)" << std::endl;
    if (extract_vram == 0) { std::cerr << "  [FAIL] extract estimate is 0" << std::endl; pass = false; }

    // cusift_memory_match
    size_t match_vram = cusift_memory_match(eo.max_keypoints, eo.max_keypoints);
    std::cout << "  cusift_memory_match(" << eo.max_keypoints << ", " << eo.max_keypoints << "): "
              << match_vram << " bytes (" << (match_vram / (1024.0 * 1024.0)) << " MB)" << std::endl;
    if (match_vram == 0) { std::cerr << "  [FAIL] match estimate is 0" << std::endl; pass = false; }

    // cusift_memory_homography
    size_t homo_vram = cusift_memory_homography(eo.max_keypoints, &ho);
    std::cout << "  cusift_memory_homography(" << eo.max_keypoints << "): "
              << homo_vram << " bytes (" << (homo_vram / (1024.0 * 1024.0)) << " MB)" << std::endl;
    if (homo_vram == 0) { std::cerr << "  [FAIL] homography estimate is 0" << std::endl; pass = false; }

    // cusift_memory_warp
    size_t warp_vram = cusift_memory_warp(im1.width, im1.height, im2.width, im2.height);
    std::cout << "  cusift_memory_warp(" << im1.width << "x" << im1.height
              << ", " << im2.width << "x" << im2.height << "): "
              << warp_vram << " bytes (" << (warp_vram / (1024.0 * 1024.0)) << " MB)" << std::endl;
    if (warp_vram == 0) { std::cerr << "  [FAIL] warp estimate is 0" << std::endl; pass = false; }

    // cusift_memory_pipeline
    size_t full_vram = cusift_memory_pipeline(im1.width, im1.height, im2.width, im2.height, &eo, &ho);
    std::cout << "  cusift_memory_pipeline: "
              << full_vram << " bytes (" << (full_vram / (1024.0 * 1024.0)) << " MB)" << std::endl;
    if (full_vram == 0) { std::cerr << "  [FAIL] full pipeline estimate is 0" << std::endl; pass = false; }

    // Sanity: full pipeline should be >= extraction (since extraction is typically the peak)
    if (full_vram < extract_vram)
    {
        std::cerr << "  [FAIL] full pipeline estimate (" << full_vram
                  << ") < extract estimate (" << extract_vram << ")" << std::endl;
        pass = false;
    }

    // Sanity: match estimate should equal 2 * sizeof(cusift_keypoint) * max_keypoints
    size_t expected_match = 2 * sizeof(cusift_keypoint) * eo.max_keypoints;
    if (match_vram != expected_match)
    {
        std::cerr << "  [FAIL] match estimate (" << match_vram
                  << ") != expected (" << expected_match << ")" << std::endl;
        pass = false;
    }

    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;
    return pass;
}

// -- Test: cusift_compute_canvas ---------------------------

static bool test_warp_geometry(const cusift_image& im1, const cusift_image& im2)
{
    std::cout << "[TEST] cusift_compute_canvas" << std::endl;

    cusift_keypoints sd1{}, sd2{};
    cusift_extract_options eo = default_extract_options();
    cusift_homography_options ho = default_homography_options();

    cusift_extract(&im1, &sd1, &eo);
    cusift_extract(&im2, &sd2, &eo);
    cusift_match(&sd1, &sd2);

    float H[9];
    int nm = 0;
    cusift_estimate_homography(&sd1, H, &nm, &ho);
    if (check_error("setup") || !homography_is_valid(H))
    {
        std::cout << "  [SKIP] no valid homography" << std::endl;
        cusift_keypoints_free(&sd1); cusift_keypoints_free(&sd2);
        return false;
    }

    const int w1 = im1.width, h1 = im1.height, w2 = im2.width, h2 = im2.height;
    bool pass = true;

    cusift_canvas g{};
    cusift_compute_canvas(H, w1, h1, w2, h2, &g);
    pass = pass && !check_error("cusift_compute_canvas");
    std::cout << "  Canvas: " << g.canvas_width << "x" << g.canvas_height
              << ", origin (" << g.origin_u << ", " << g.origin_v << ")" << std::endl;
    for (int k = 0; k < 4; k++)
        std::cout << "  image-2 corner " << k << " -> (" << g.corners2[k][0] << ", " << g.corners2[k][1] << ")" << std::endl;

    // 1. The warp allocates exactly the canvas the geometry predicts.
    cusift_image wa = {}, wb = {};
    cusift_warp(&im1, &im2, H, &wa, &wb, false);
    pass = pass && !check_error("cusift_warp");
    if (wa.width != g.canvas_width || wa.height != g.canvas_height)
    {
        std::cout << "  [FAIL] warped size " << wa.width << "x" << wa.height << " != geometry" << std::endl;
        pass = false;
    }

    // 2. Image 1 sits at the origin shift and its corners are integers.
    if (g.origin_u > 0 || g.origin_v > 0
        || g.corners1[0][0] != (float)(-g.origin_u) || g.corners1[0][1] != (float)(-g.origin_v)
        || g.corners1[2][0] != (float)(w1 - 1 - g.origin_u) || g.corners1[2][1] != (float)(h1 - 1 - g.origin_v))
    {
        std::cout << "  [FAIL] image-1 corners do not match the origin shift" << std::endl;
        pass = false;
    }

    // 3. The first valid column/row of the warped image 1 is the origin shift,
    //    which is how callers had to recover the origin before this API existed.
    if (wa.pixels)
    {
        int firstCol = -1, firstRow = -1;
        const int rowOfImage1 = -g.origin_v;   // a canvas row that image 1 covers
        const int colOfImage1 = -g.origin_u;   // a canvas column that image 1 covers
        for (int x = 0; x < wa.width && firstCol < 0; x++)
            if (!std::isnan(wa.pixels[(size_t)rowOfImage1 * wa.width + x])) firstCol = x;
        for (int y = 0; y < wa.height && firstRow < 0; y++)
            if (!std::isnan(wa.pixels[(size_t)y * wa.width + colOfImage1])) firstRow = y;
        if (firstCol != -g.origin_u || firstRow != -g.origin_v)
        {
            std::cout << "  [FAIL] first valid pixel of warped image 1 at (" << firstCol << ", " << firstRow
                      << "), expected (" << -g.origin_u << ", " << -g.origin_v << ")" << std::endl;
            pass = false;
        }
    }

    // 4. Round trip: each image-2 corner on the canvas, moved back to image-1
    //    coordinates and pushed through H, must land on the matching corner of image 2.
    const float c2[4][2] = {{0.0f, 0.0f}, {(float)(w2 - 1), 0.0f}, {(float)(w2 - 1), (float)(h2 - 1)}, {0.0f, (float)(h2 - 1)}};
    float maxErr = 0.0f;
    for (int k = 0; k < 4; k++)
    {
        const float u = g.corners2[k][0] + (float)g.origin_u;
        const float v = g.corners2[k][1] + (float)g.origin_v;
        const float z = H[6] * u + H[7] * v + H[8];
        const float x = (H[0] * u + H[1] * v + H[2]) / z;
        const float y = (H[3] * u + H[4] * v + H[5]) / z;
        maxErr = std::max(maxErr, std::hypot(x - c2[k][0], y - c2[k][1]));
    }
    std::cout << "  Corner round-trip error through H: " << maxErr << " px" << std::endl;
    if (!(maxErr < 1e-2f))
    {
        std::cout << "  [FAIL] round-trip error too large" << std::endl;
        pass = false;
    }

    // 5. Degenerate inputs are reported through the error state.
    float Hzero[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    cusift_canvas gz{};
    cusift_compute_canvas(Hzero, w1, h1, w2, h2, &gz);
    if (!cusift_last_error(nullptr))
    {
        std::cout << "  [FAIL] singular homography was not rejected" << std::endl;
        pass = false;
    }

    // 6. A canvas beyond the 2x guard is rejected like the warp rejects it,
    //    but the geometry is still filled so the caller can see the size.
    float Hfar[9] = {1.0f, 0.0f, -(float)(3 * w1), 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};  // image 2 sits 3 widths to the right
    cusift_canvas gf{};
    cusift_compute_canvas(Hfar, w1, h1, w2, h2, &gf);
    if (!cusift_last_error(nullptr) || gf.canvas_width <= 2 * std::max(w1, w2))
    {
        std::cout << "  [FAIL] oversized canvas not reported (had_error=" << cusift_last_error(nullptr)
                  << ", width=" << gf.canvas_width << ")" << std::endl;
        pass = false;
    }

    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;

    cusift_image_free(&wa);
    cusift_image_free(&wb);
    cusift_keypoints_free(&sd1);
    cusift_keypoints_free(&sd2);
    return pass;
}

// -- Test: error log file and error callback ---------------

struct ErrorTally
{
    int errors = 0;
    int warnings = 0;
    std::string last_function;
    std::string last_message;
};

static void tally_callback(const cusift_error_record* r, void* user)
{
    ErrorTally* t = static_cast<ErrorTally*>(user);
    if (r->severity == 0) t->errors++; else t->warnings++;
    t->last_function = r->function;
    t->last_message = r->message;
}

static std::string read_file(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static bool test_error_log(const cusift_image& im1)
{
    std::cout << "[TEST] Error log file and callback" << std::endl;
    bool pass = true;
    const std::string path = "cusift-test-errors.log";
    const std::string rotated = path + ".1";
    std::remove(path.c_str());
    std::remove(rotated.c_str());

    ErrorTally tally;
    cusift_set_error_callback(tally_callback, &tally);
    cusift_set_error_log(path.c_str(), 1500); // small cap so the loop below rotates the file

    cusift_extract_options eo = default_extract_options();
    float dummy[64] = {0.0f};
    cusift_image bad = { dummy, 4, 4 };
    const int N = 25;
    for (int i = 0; i < N; i++)
    {
        cusift_keypoints sd{};
        cusift_extract(&bad, &sd, &eo);
        cusift_keypoints_free(&sd);
    }

    // 1. Every error reached the callback, with the public function and message.
    if (tally.errors != N || tally.warnings != 0
        || tally.last_function != "cusift_extract"
        || tally.last_message.find("at least 8 pixels") == std::string::npos)
    {
        std::cout << "  [FAIL] callback saw " << tally.errors << " errors / " << tally.warnings << " warnings, last: "
                  << tally.last_function << ": " << tally.last_message << std::endl;
        pass = false;
    }

    // 2. The file holds complete lines and was rotated at the cap.
    const std::string content = read_file(path);
    const bool has_line = content.find("cusift_extract") != std::string::npos
                       && content.find("at least 8 pixels") != std::string::npos
                       && content.find(" error ") != std::string::npos
                       && content.find("Z ") != std::string::npos;
    const bool was_rotated = !read_file(rotated).empty();
    std::cout << "  log: " << content.size() << " bytes, rotated file " << (was_rotated ? "present" : "absent") << std::endl;
    std::cout << "  last line: " << content.substr(content.rfind('\n', content.size() - 2) + 1);
    if (!has_line || !was_rotated || content.size() > 1500)
    {
        std::cout << "  [FAIL] log content or rotation wrong" << std::endl;
        pass = false;
    }

    // 3. Nothing was dropped.
    if (cusift_error_log_dropped() != 0)
    {
        std::cout << "  [FAIL] dropped " << cusift_error_log_dropped() << " records" << std::endl;
        pass = false;
    }

    // 4. A successful call reports nothing.
    {
        cusift_keypoints sd{};
        cusift_extract(&im1, &sd, &eo);
        cusift_keypoints_free(&sd);
        if (tally.errors != N)
        {
            std::cout << "  [FAIL] a successful call produced a report" << std::endl;
            pass = false;
        }
    }

    // 5. Turning the file off keeps the callback; turning the callback off silences it.
    cusift_set_error_log(NULL, 0);
    {
        cusift_keypoints sd{};
        cusift_extract(&bad, &sd, &eo);
        cusift_keypoints_free(&sd);
    }
    const size_t size_after_off = read_file(path).size();
    cusift_set_error_callback(NULL, NULL);
    {
        cusift_keypoints sd{};
        cusift_extract(&bad, &sd, &eo);
        cusift_keypoints_free(&sd);
    }
    if (tally.errors != N + 1 || size_after_off != content.size())
    {
        std::cout << "  [FAIL] disable semantics wrong (errors " << tally.errors << ", size " << size_after_off << ")" << std::endl;
        pass = false;
    }

    std::remove(path.c_str());
    std::remove(rotated.c_str());
    std::cout << "  " << (pass ? "[PASS]" : "[FAIL]") << std::endl;
    return pass;
}

// -- Test: input validation and device reset --------------

static bool test_input_validation(const cusift_image& im1, const cusift_image& im2)
{
    std::cout << "[TEST] Input validation and cusift_device_reset" << std::endl;
    bool pass = true;
    cusift_extract_options eo = default_extract_options();
    std::string msg;

    // 1. Dimensions outside [8, 65535] must be rejected before any GPU work or
    //    any read of the pixel buffer (the buffer here is far too small).
    float dummy[64] = {0.0f};
    struct Case { const char* name; int w, h; };
    const Case cases[] = {{"4x4", 4, 4}, {"4000x1", 4000, 1}, {"70000x8", 70000, 8}, {"8x70000", 8, 70000}};
    for (const Case& c : cases)
    {
        cusift_keypoints sd{};
        cusift_image bad = { dummy, c.w, c.h };
        cusift_extract(&bad, &sd, &eo);
        msg = last_error_message();
        bool rejected = !msg.empty();
        std::cout << "  " << c.name << ": " << (rejected ? msg.c_str() : "ACCEPTED (wrong)") << std::endl;
        if (!rejected) pass = false;
        cusift_keypoints_free(&sd);
    }

    // 2. A device pointer passed as pixels must be classified and rejected
    //    on every path, including the CPU warp that used to dereference it.
    {
        cusift_keypoints sd1{}, sd2{};
        cusift_homography_options ho = default_homography_options();
        float H[9]; int nm = 0;
        cusift_extract(&im1, &sd1, &eo);
        cusift_extract(&im2, &sd2, &eo);
        cusift_match(&sd1, &sd2);
        cusift_estimate_homography(&sd1, H, &nm, &ho);
        cusift_device_image g1 = {}, g2 = {};
        if (!check_error("setup") && homography_is_valid(H))
        {
            cusift_warp_to_device(&im1, &im2, H, &g1, &g2);
            if (!check_error("cusift_warp_to_device") && g1.device_pixels)
            {
                cusift_image dev = { g1.device_pixels, g1.width, g1.height };   // device memory
                cusift_keypoints sd3{};
                cusift_extract(&dev, &sd3, &eo);
                msg = last_error_message();
                bool rejected = !msg.empty();
                std::cout << "  device pointer -> cusift_extract: " << (rejected ? msg.c_str() : "ACCEPTED (wrong)") << std::endl;
                if (!rejected) pass = false;
                cusift_keypoints_free(&sd3);

                cusift_image o1 = {}, o2 = {};
                cusift_warp(&dev, &im2, H, &o1, &o2, false);
                msg = last_error_message();
                rejected = !msg.empty();
                std::cout << "  device pointer -> cusift_warp (CPU): " << (rejected ? msg.c_str() : "ACCEPTED (wrong)") << std::endl;
                if (!rejected) pass = false;
                cusift_image_free(&o1); cusift_image_free(&o2);
            }
        }
        cusift_device_image_free(&g1); cusift_device_image_free(&g2);
        cusift_keypoints_free(&sd1); cusift_keypoints_free(&sd2);
    }

    // 3. The reset must leave the device usable.  Every library object above
    //    has been freed, as the contract requires.
    cusift_device_reset();
    if (check_error("cusift_device_reset")) pass = false;
    {
        cusift_keypoints sd{};
        cusift_extract(&im1, &sd, &eo);
        if (check_error("cusift_extract after reset") || sd.count <= 0) pass = false;
        std::cout << "  after reset: " << sd.count << " keypoints" << std::endl;
        cusift_keypoints_free(&sd);
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

    cusift_image im1 = { image1.data(), width1, height1 };
    cusift_image im2 = { image2.data(), width2, height2 };

    cusift_init();

    int passed = 0, failed = 0, total = 0;

    auto run = [&](bool result) { total++; result ? passed++ : failed++; std::cout << std::endl; };

    run(test_extract(im1, im2));
    run(test_match(im1, im2));
    run(test_find_homography(im1, im2));
    run(test_warp_images_cpu(im1, im2));
    run(test_warp_images_gpu(im1, im2));
    run(test_warp_images_gpu_strided(im1, im2));
    run(test_warp_geometry(im1, im2));
    run(test_save_sift_data(im1));
    run(test_extract_and_match(im1, im2));
    run(test_extract_match_homography(im1, im2));
    run(test_extract_match_homography_warp(im1, im2));
    run(test_extract_match_homography_warp_gpu(im1, im2));
    run(test_find_homography_both_models(im1, im2));
    run(test_extract_match_homography_both_models(im1, im2));
    run(test_extract_match_homography_warp_both_models(im1, im2));
    run(test_extract_match_homography_multi(im1, im2));
    run(test_extract_match_homography_multi_warp(im1, im2));
    run(test_estimate_vram(im1, im2));
    run(test_error_log(im1));
    run(test_input_validation(im1, im2)); // last: it resets the device

    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed, " << total << " total" << std::endl;

    return failed > 0 ? 1 : 0;
}







