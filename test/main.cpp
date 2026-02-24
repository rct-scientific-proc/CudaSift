#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "cusift.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// Load an image as a row-major float array (grayscale).
// Returns the pixel data via *out_pixels; caller must free() it.
static bool LoadImageAsFloat(const char* path, float** out_pixels, int* out_w, int* out_h)
{
    int w, h, channels;
    unsigned char* raw = stbi_load(path, &w, &h, &channels, 1); // force 1 channel
    if (!raw)
    {
        fprintf(stderr, "Failed to load image: %s\n", path);
        return false;
    }

    float* pixels = (float*)malloc(sizeof(float) * w * h);
    if (!pixels)
    {
        stbi_image_free(raw);
        return false;
    }

    for (int i = 0; i < w * h; i++)
        pixels[i] = (float)raw[i];

    stbi_image_free(raw);
    *out_pixels = pixels;
    *out_w = w;
    *out_h = h;
    return true;
}

static void PrintHomography(const float* H)
{
    printf("Homography:\n");
    for (int r = 0; r < 3; r++)
        printf("  [%12.6f %12.6f %12.6f]\n", H[r * 3 + 0], H[r * 3 + 1], H[r * 3 + 2]);
}

// ── Simple config file parser ───────────────────────────
// Format: one "key: value" per line. Lines starting with '#' are comments.
// Supported keys (for ExtractSiftOptions_t):
//   thresh, lowest_scale, edge_thresh, init_blur, max_keypoints, num_octaves
// Supported keys (for FindHomographyOptions_t):
//   num_loops, min_score, max_ambiguity, homo_thresh,
//   improve_num_loops, improve_min_score, improve_max_ambiguity, improve_thresh, seed
static bool LoadConfig(const char* path, ExtractSiftOptions_t* eo, FindHomographyOptions_t* ho)
{
    FILE* f = fopen(path, "r");
    if (!f)
    {
        fprintf(stderr, "Failed to open config file: %s\n", path);
        return false;
    }
    char line[256];
    while (fgets(line, sizeof(line), f))
    {
        // Skip comments and blank lines
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r')
            continue;
        char key[128];
        char valStr[128];
        if (sscanf(line, "%127[^:]: %127s", key, valStr) != 2)
            continue;

        // Extract options
        if      (strcmp(key, "thresh") == 0)         eo->thresh_        = (float)atof(valStr);
        else if (strcmp(key, "lowest_scale") == 0)    eo->lowest_scale_  = (float)atof(valStr);
        else if (strcmp(key, "edge_thresh") == 0)     eo->edge_thresh_   = (float)atof(valStr);
        else if (strcmp(key, "init_blur") == 0)       eo->init_blur_     = (float)atof(valStr);
        else if (strcmp(key, "max_keypoints") == 0)   eo->max_keypoints_ = atoi(valStr);
        else if (strcmp(key, "num_octaves") == 0)     eo->num_octaves_   = atoi(valStr);
        // Homography options
        else if (strcmp(key, "num_loops") == 0)              ho->num_loops_              = atoi(valStr);
        else if (strcmp(key, "min_score") == 0)              ho->min_score_              = (float)atof(valStr);
        else if (strcmp(key, "max_ambiguity") == 0)          ho->max_ambiguity_          = (float)atof(valStr);
        else if (strcmp(key, "homo_thresh") == 0)            ho->thresh_                 = (float)atof(valStr);
        else if (strcmp(key, "improve_num_loops") == 0)      ho->improve_num_loops_      = atoi(valStr);
        else if (strcmp(key, "improve_min_score") == 0)      ho->improve_min_score_      = (float)atof(valStr);
        else if (strcmp(key, "improve_max_ambiguity") == 0)  ho->improve_max_ambiguity_  = (float)atof(valStr);
        else if (strcmp(key, "improve_thresh") == 0)         ho->improve_thresh_         = (float)atof(valStr);
        else if (strcmp(key, "seed") == 0)                   ho->seed_                   = (unsigned int)strtoul(valStr, nullptr, 10);
        else
            fprintf(stderr, "Config: unknown key '%s'\n", key);
    }
    fclose(f);
    return true;
}

int main(int argc, char** argv)
{
    const char* img1_path = "data/img1.png";
    const char* img2_path = "data/img2.png";
    const char* out1_path = "sift1.json";
    const char* out2_path = "sift2.json";

    if (argc >= 3)
    {
        img1_path = argv[1];
        img2_path = argv[2];
    }
    if (argc >= 5)
    {
        out1_path = argv[3];
        out2_path = argv[4];
    }

    // ── Extract SIFT features ───────────────────────────
    ExtractSiftOptions_t extract_opts = {};
    extract_opts.thresh_        = 0.09f;
    extract_opts.lowest_scale_  = 0.0f;
    extract_opts.edge_thresh_   = 10.0f;
    extract_opts.init_blur_     = 1.0f;
    extract_opts.max_keypoints_ = 32768;
    extract_opts.num_octaves_   = 5;

    FindHomographyOptions_t homo_opts = {};
    homo_opts.num_loops_              = 10000;
    homo_opts.min_score_              = 0.0f;
    homo_opts.max_ambiguity_          = 0.80f;
    homo_opts.thresh_                 = 0.09f;
    homo_opts.improve_num_loops_      = 5;
    homo_opts.improve_min_score_      = 0.0f;
    homo_opts.improve_max_ambiguity_  = 0.80f;
    homo_opts.improve_thresh_         = 0.07f;
    homo_opts.seed_                   = 42;

    if (argc >= 6)
    {
        // Load config file — values override the defaults above
        if (!LoadConfig(argv[5], &extract_opts, &homo_opts))
            fprintf(stderr, "Warning: could not load config '%s', using defaults\n", argv[5]);
        else
            printf("Loaded config from %s\n", argv[5]);
    }

    // ── Initialize CUDA ─────────────────────────────────
    InitializeCudaSift();

    // ── Load images ─────────────────────────────────────
    float* pixels1 = nullptr;
    float* pixels2 = nullptr;
    int w1, h1, w2, h2;

    if (!LoadImageAsFloat(img1_path, &pixels1, &w1, &h1))
        return 1;
    if (!LoadImageAsFloat(img2_path, &pixels2, &w2, &h2))
    {
        free(pixels1);
        return 1;
    }

    printf("Image 1: %s (%d x %d)\n", img1_path, w1, h1);
    printf("Image 2: %s (%d x %d)\n", img2_path, w2, h2);

    Image_t image1 = { pixels1, w1, h1 };
    Image_t image2 = { pixels2, w2, h2 };

    // ── Extract SIFT features ───────────────────────────
    SiftData sift1, sift2;
    memset(&sift1, 0, sizeof(sift1));
    memset(&sift2, 0, sizeof(sift2));

    printf("Extracting SIFT features from image 1...\n");
    ExtractSiftFromImage(&image1, &sift1, &extract_opts);
    printf("  Found %d keypoints\n", sift1.numPts);

    printf("Extracting SIFT features from image 2...\n");
    ExtractSiftFromImage(&image2, &sift2, &extract_opts);
    printf("  Found %d keypoints\n", sift2.numPts);

    // ── Match SIFT features ─────────────────────────────
    printf("Matching SIFT features...\n");
    MatchSiftData(&sift1, &sift2);

    // ── Find homography ─────────────────────────────────
    float homography[9];
    int num_matches = 0;

    printf("Finding homography...\n");
    FindHomography(&sift1, homography, &num_matches, &homo_opts);
    printf("  Matches (inliers): %d\n", num_matches);
    PrintHomography(homography);

    // ── Save SIFT data to JSON ──────────────────────────
    printf("Saving SIFT data to %s and %s...\n", out1_path, out2_path);
    SaveSiftData(out1_path, &sift1);
    SaveSiftData(out2_path, &sift2);

    // ── Cleanup ─────────────────────────────────────────
    DeleteSiftData(&sift1);
    DeleteSiftData(&sift2);
    free(pixels1);
    free(pixels2);

    printf("Done.\n");
    return 0;
}
