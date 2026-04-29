# CudaSift

GPU-accelerated SIFT feature extraction, descriptor matching, RANSAC homography
estimation, and image warping &mdash; exposed as a small, stable C API
(`cusift.h`) usable from C, C++, and Python.

This is a maintained fork of Mårten Björkman's
[Celebrandil/CudaSift](https://github.com/Celebrandil/CudaSift), restructured
into a proper library with a clean public interface, end-to-end convenience
functions, deterministic RANSAC, error reporting, VRAM estimation, and Python
bindings.

---

## Features

- **SIFT detection and 128-d descriptors** on the GPU using the original
  CudaSift Difference-of-Gaussians pyramid and Lowe's descriptor.
- **Brute-force descriptor matching** with Lowe's ratio test, fully on the GPU.
- **RANSAC** estimators for both
  - 8-DOF projective **homography** (4-point DLT), and
  - 4-DOF **similarity** transform (rotation + uniform scale + translation,
    2-point closed-form).
- **Iterative refinement** of the estimated homography (IRLS with binary Huber
  weighting, AVX2/FMA on the host).
- **Image warping** to a common canvas (CPU or GPU paths).
- **Convenience pipelines** that fuse extract &rarr; match &rarr; homography
  &rarr; warp into a single call.
- **VRAM estimation** for every stage so callers can pre-flight their memory
  budget before allocating.
- **Out-of-band error reporting**: every API call sets a thread-local
  `(file, line, message)` triple instead of `throw`/`assert`, making it easy
  to use from C and other languages.
- **Python bindings** via `ctypes`, packaged under `cusift_py/`.
- **Concurrency-safe**: per-call device contexts replace the original
  `__constant__` / `__device__` globals, so multiple host threads or processes
  can share a single GPU.

## Project layout

```
include/
    cusift.h                 Public C API (the only header callers need)
src/
    cusift.cu                Public API implementation
    cudaSiftH.{cu,h}         Host-side SIFT pipeline
    cudaSiftD.{cu,h}         Device kernels (ScaleDown, Laplace, FindPoints, ...)
    cudaImage.{cu,h}         Pitched device image wrapper
    matching.cu              Descriptor matching + RANSAC kernels
    geomFuncs.{c,h}          AVX2/FMA host-side IRLS homography refinement
    cudautils.h              Error handling, warp-shuffle helpers
    RAII_Gaurds.hpp          Device/host pointer guards
    cudaSift.h               Internal aggregate header
test/
    main.cpp                 17-test correctness/regression suite
    speed.cpp                Multi-resolution benchmarking harness
    img1.png, img2.png       Sample image pair
cusift_py/
    cusift/                  Python package (ctypes bindings)
CMakeLists.txt
```

## Requirements

- CUDA Toolkit 11.0 or newer (tested against 13.x).
- A C++17 compiler (MSVC 2019+, GCC 9+, Clang 10+).
- AVX2/FMA-capable CPU (Haswell, Zen, or newer) for the host-side
  `ImproveHomography` path.
- CMake 3.23 or newer.
- (Optional) Python 3.8+ for `cusift_py`.

Supported GPU architectures default to **75, 80, 86, 89** (Turing through Ada
Lovelace). Override with `-DCMAKE_CUDA_ARCHITECTURES="..."` or pass
`-DCUSIFT_ALL_CUDA_ARCHS=ON` to compile for every architecture supported by
your toolkit.

## Building

```bash
git clone https://github.com/rct-scientific-proc/CudaSift.git
cd CudaSift
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

CMake options:

| Option                  | Default | Description                                     |
| ----------------------- | ------- | ----------------------------------------------- |
| `CUSIFT_BUILD_SHARED`   | `OFF`   | Build `cusift` as a shared library (DLL/.so).   |
| `CUSIFT_USE_FAST_MATH`  | `OFF`   | Compile CUDA with `--use_fast_math`.            |
| `CUSIFT_ALL_CUDA_ARCHS` | `OFF`   | Build for every architecture supported by NVCC. |

To install:

```bash
cmake --install build --prefix install
```

This installs the `cusift` library, the `cusift_demo` and `cusift_speed`
executables, and the public header `include/cusift.h`.

## Running the tests

After building, run the full regression suite against the bundled sample
images:

```bash
./build/Release/cusift_demo test/img1.png test/img2.png
```

Expected output ends with:

```
========================================
Results: 17 passed, 0 failed, 17 total
```

For benchmarking across multiple input resolutions:

```bash
./build/Release/cusift_speed test/img1.png test/img2.png
```

## Quick start (C/C++)

```cpp
#include "cusift.h"
#include <limits>
#include <vector>
#include <cstdio>

int main()
{
    InitializeCudaSift();

    // Load two grayscale float32 images (row-major, [0, 255]).
    std::vector<float> img1, img2;
    int w1, h1, w2, h2;
    /* ... fill in img1, img2, w1/h1, w2/h2 ... */

    Image_t in1{img1.data(), w1, h1};
    Image_t in2{img2.data(), w2, h2};

    ExtractSiftOptions_t extract_opts{};
    extract_opts.thresh_                    = 2.0f;
    extract_opts.lowest_scale_              = 0.0f;
    extract_opts.highest_scale_             = std::numeric_limits<float>::infinity();
    extract_opts.edge_thresh_               = 10.0f;
    extract_opts.init_blur_                 = 1.0f;
    extract_opts.max_keypoints_             = 10000;
    extract_opts.num_octaves_               = 5;
    extract_opts.scale_suppression_radius_  = 0.0f;

    FindHomographyOptions_t homog_opts{};
    homog_opts.num_loops_         = 10000;
    homog_opts.thresh_            = 3.0f;
    homog_opts.improve_num_loops_ = 5;
    homog_opts.improve_thresh_    = 2.0f;
    homog_opts.seed_              = 42;          // deterministic; 0 = random_device
    homog_opts.model_type_        = CUSIFT_MODEL_HOMOGRAPHY;

    SiftData s1{}, s2{};
    float    H[9];
    int      num_matches = 0;

    ExtractAndMatchAndFindHomography(&in1, &in2, &s1, &s2, H, &num_matches,
                                     &extract_opts, &homog_opts);

    if (CusiftHadError()) {
        int line; char file[256], msg[256];
        CusiftGetLastErrorString(&line, file, msg);
        std::fprintf(stderr, "%s:%d  %s\n", file, line, msg);
        return 1;
    }

    /* H is a row-major 3x3, num_matches is the inlier count. */

    DeleteSiftData(&s1);
    DeleteSiftData(&s2);
}
```

The build exports a `cusift` CMake target you can consume from a downstream
project after `find_package(cusift)` or `add_subdirectory()`:

```cmake
target_link_libraries(my_app PRIVATE cusift)
```

## Quick start (Python)

```python
import numpy as np
from PIL import Image
from cusift import (
    InitializeCudaSift,
    ExtractAndMatchAndFindHomography,
    ExtractSiftOptions, FindHomographyOptions,
)

InitializeCudaSift()

img1 = np.asarray(Image.open("test/img1.png").convert("L"), dtype=np.float32)
img2 = np.asarray(Image.open("test/img2.png").convert("L"), dtype=np.float32)

extract_opts    = ExtractSiftOptions()
homography_opts = FindHomographyOptions()

H, num_matches, sift1, sift2 = ExtractAndMatchAndFindHomography(
    img1, img2, extract_opts, homography_opts,
)
print(f"Found {num_matches} inliers, H =\n{H}")
```

See `cusift_py/cusift/cusift.py` for the full Python surface and
`cusift_py/test.py` for a runnable example.

## API surface (overview)

All public functions live in [`include/cusift.h`](include/cusift.h) and are
documented inline. The big picture:

| Function                                         | Purpose                                            |
| ------------------------------------------------ | -------------------------------------------------- |
| `InitializeCudaSift`                             | Pick a CUDA device. Call once.                     |
| `ExtractSiftFromImage`                           | DoG + descriptor extraction.                       |
| `MatchSiftData`                                  | Brute-force descriptor matching with ratio test.   |
| `FindHomography`                                 | RANSAC homography or similarity, with refinement.  |
| `WarpImages` / `WarpImages_GPU`                  | Warp both images to a common canvas.               |
| `ExtractAndMatchSift`                            | Convenience: extract both + match.                 |
| `ExtractAndMatchAndFindHomography`               | Convenience: extract + match + RANSAC.             |
| `ExtractAndMatchAndFindHomography_Multi`         | Multi-attempt RANSAC with goal-based selection.    |
| `ExtractAndMatchAndFindHomographyAndWarp[_GPU]`  | Full pipeline through to warped output.            |
| `ExtractAndMatchAndFindHomography_Multi_AndWarp` | Multi-attempt full pipeline.                       |
| `SaveSiftData`                                   | Persist a `SiftData` to JSON.                      |
| `DeleteSiftData`, `FreeImage`, `FreeImage_GPU`   | Resource cleanup.                                  |
| `CusiftHadError`, `CusiftGetLastErrorString`     | Error reporting.                                   |
| `EstimateVram*`                                  | Pre-flight VRAM accounting for every stage.        |

## Error handling

The library never throws across the C boundary. Every public function clears
the thread-local error flag on entry; check it after the call:

```c
if (CusiftHadError()) {
    int line; char file[256], msg[256];
    CusiftGetLastErrorString(&line, file, msg);
    /* report and recover */
}
```

## VRAM estimation

The `EstimateVram*` family lets you size your GPU budget without actually
allocating anything:

```c
size_t bytes = EstimateVramFullPipeline(w1, h1, w2, h2,
                                        &extract_opts,
                                        &homography_opts);
```

The peak typically occurs during SIFT extraction (the scale-space pyramid).
Returned values include all temporary buffers as well as the caller-visible
`SiftData` and warped-image allocations.

## License

MIT. Original copyright &copy; 2017 Mårten Björkman; ongoing modifications
&copy; rct-scientific-proc. See [LICENSE](LICENSE) for the full text.

## References

- Lowe, D. "Distinctive Image Features from Scale-Invariant Keypoints."
  *International Journal of Computer Vision*, 60(2), 91&ndash;110, 2004.
- Björkman, M., Bergström, N., Kragic, D. "Detecting, segmenting and
  tracking unknown objects using multi-label MRF inference." *CVIU*, 118,
  111&ndash;127, January 2014.
- Upstream project: <https://github.com/Celebrandil/CudaSift>
