# Migrating to cusift 3.0

3.0 renames the whole public surface to one convention: every exported symbol
is prefixed `cusift_` and written in snake_case, types carry no suffix, and
struct fields are plain snake_case. Two structural changes ride along: the
six pipeline functions collapse into three (plus a device-output variant),
and the three error getters become one. Nothing about detection, matching or
warping changed; a program that compiled against 2.5 produces identical
numbers once renamed.

The Python package follows the same names. Its `find_homography`,
`warp_images` and the four `extract_and_match_and_find_homography...` methods
are renamed; option dataclass fields are renamed to match the C structs.

## Functions

| 2.x | 3.0 |
|---|---|
| `InitializeCudaSift` | `cusift_init` |
| `CusiftResetDevice` | `cusift_device_reset` |
| `CusiftHadError`, `CusiftGetLastErrorString`, `CusiftGetLastCudaError` | `cusift_last_error(record *)`, returns 1 when the last call failed and fills a `cusift_error_record` |
| `CusiftSetErrorLog` | `cusift_set_error_log` |
| `CusiftSetErrorCallback` | `cusift_set_error_callback` |
| `CusiftErrorLogDroppedCount` | `cusift_error_log_dropped` |
| `ExtractSiftFromImage` | `cusift_extract` |
| `MatchSiftData` | `cusift_match` (second argument is now `const`) |
| `FindHomography` | `cusift_estimate_homography` |
| `ComputeWarpGeometry` | `cusift_compute_canvas` |
| `WarpImages` | `cusift_warp` |
| `WarpImages_GPU` | `cusift_warp_to_device` |
| `ExtractAndMatchSift` | `cusift_extract_and_match` |
| `ExtractAndMatchAndFindHomography` | `cusift_align` |
| `ExtractAndMatchAndFindHomography_Multi` | `cusift_align` with `attempts` and `goal` set in the options |
| `ExtractAndMatchAndFindHomographyAndWarp` | `cusift_align_and_warp` |
| `ExtractAndMatchAndFindHomography_Multi_AndWarp` | `cusift_align_and_warp` with `attempts` and `goal` set in the options |
| `ExtractAndMatchAndFindHomographyAndWarp_GPU` | `cusift_align_and_warp_to_device` |
| `DeleteSiftData` | `cusift_keypoints_free` |
| `FreeImage` | `cusift_image_free` |
| `FreeImage_GPU` | `cusift_device_image_free` |
| `SaveSiftData` | `cusift_keypoints_save_json` |
| `EstimateVramExtractSift` | `cusift_memory_extract` |
| `EstimateVramMatchSift` | `cusift_memory_match` |
| `EstimateVramFindHomography` | `cusift_memory_homography` |
| `EstimateVramWarpImages` | `cusift_memory_warp` |
| `EstimateVramFullPipeline` | `cusift_memory_pipeline` |
| (new) | `cusift_version`, `cusift_default_extract_options`, `cusift_default_homography_options` |

`attempts = 1` reproduces the former single-estimate functions exactly
(attempt 0 uses the caller's seed), so folding the `_Multi` variants into the
options changes no result.

## Types

| 2.x | 3.0 |
|---|---|
| `SiftPoint` | `cusift_keypoint` |
| `SiftData` | `cusift_keypoints` |
| `Image_t` | `cusift_image` |
| `ImageStrided_t` | `cusift_device_image` |
| `WarpGeometry_t` | `cusift_canvas` |
| `ExtractSiftOptions_t` | `cusift_extract_options` |
| `FindHomographyOptions_t` | `cusift_homography_options` |
| `CusiftErrorRecord_t` | `cusift_error_record` |
| `CusiftErrorCallback_t` | `cusift_error_callback` |
| `CUSIFT_MODEL_HOMOGRAPHY`, `CUSIFT_MODEL_SIMILARITY` (macros) | same names, now the `cusift_model` enum |
| `CUSIFT_HOMOGRAPHY_GOAL_MAX_INLIERS` | `CUSIFT_GOAL_MAX_INLIERS` (`cusift_goal` enum) |
| `CUSIFT_HOMOGRAPHY_GOAL_MIN_EYE_DIFF` | `CUSIFT_GOAL_MIN_IDENTITY_DISTANCE` |

Struct layouts are unchanged except `cusift_homography_options`, which gains
`attempts` and `goal` at the end.

## Fields

| Struct | 2.x | 3.0 |
|---|---|---|
| keypoint | `xpos`, `ypos` | `x`, `y` |
| keypoint | `match_xpos`, `match_ypos` | `match_x`, `match_y` |
| keypoint | `data` | `descriptor` |
| keypoint | `empty` | `reserved` |
| keypoint set | `numPts`, `maxPts` | `count`, `capacity` |
| keypoint set | `h_data`, `d_data` | `host`, `device` |
| image | `host_img_`, `width_`, `height_` | `pixels`, `width`, `height` |
| device image | `strided_img_`, `width_`, `height_`, `stride_` | `device_pixels`, `width`, `height`, `pitch_bytes` |
| extract options | `thresh_` | `contrast_threshold` |
| extract options | `lowest_scale_`, `highest_scale_` | `lowest_scale`, `highest_scale` |
| extract options | `edge_thresh_` | `edge_threshold` |
| extract options | `init_blur_` | `initial_blur` |
| extract options | `max_keypoints_`, `num_octaves_` | `max_keypoints`, `num_octaves` |
| extract options | `scale_suppression_radius_` | `scale_suppression_radius` |
| homography options | `num_loops_` | `ransac_iterations` |
| homography options | `min_score_`, `max_ambiguity_` | `min_score`, `max_ambiguity` |
| homography options | `thresh_` | `inlier_threshold` |
| homography options | `improve_num_loops_` | `refine_iterations` |
| homography options | `improve_min_score_`, `improve_max_ambiguity_` | `refine_min_score`, `refine_max_ambiguity` |
| homography options | `improve_thresh_` | `refine_inlier_threshold` |
| homography options | `seed_`, `model_type_` | `seed`, `model` |
| homography options | (new) | `attempts`, `goal` |

## Errors

```c
/* 2.x */
if (CusiftHadError()) {
    int line; char file[256], msg[256];
    CusiftGetLastErrorString(&line, file, sizeof(file), msg, sizeof(msg));
    int cuda = CusiftGetLastCudaError();
}

/* 3.0 */
cusift_error_record err;
if (cusift_last_error(&err)) {
    /* err.function, err.file, err.line, err.cuda_error, err.cuda_error_name, err.message;
       the strings stay valid until the next library call on this thread */
}
```

## Python

| 2.x | 3.0 |
|---|---|
| `KeypointList` | `Keypoints` |
| `find_homography` | `estimate_homography` |
| `warp_images` | `warp` |
| `compute_warp_geometry` | `compute_canvas` (returns a `Canvas`, formerly `WarpGeometry`) |
| `extract_and_match_and_find_homography`, `..._multi` | `align(..., options=HomographyOptions(attempts=..., goal=...))` |
| `extract_and_match_and_find_homography_and_warp`, `..._multi_and_warp` | `align_and_warp(...)` |
| `error_log_dropped_count` | `error_log_dropped` |
| `ExtractOptions.thresh`, `.edge_thresh`, `.init_blur` | `.contrast_threshold`, `.edge_threshold`, `.initial_blur` |
| `HomographyOptions.num_loops`, `.thresh`, `.improve_*`, `.model_type` | `.ransac_iterations`, `.inlier_threshold`, `.refine_*`, `.model` |
| `HOMOGRAPHY_GOAL_MAX_INLIERS`, `HOMOGRAPHY_GOAL_MIN_EYE_DIFF` | `GOAL_MAX_INLIERS`, `GOAL_MIN_IDENTITY_DISTANCE` |

`Keypoint.x`, `.y`, `.match_x`, `.match_y`, `.descriptor`, `MatchResult`,
`ErrorRecord`, `extract`, `match`, `extract_and_match`,
`reset_device`, `set_error_log`, `set_error_callback` and the drawing helpers
are unchanged.
