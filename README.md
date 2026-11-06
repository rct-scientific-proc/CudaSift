# CudaSift

GPU-accelerated SIFT feature extraction, descriptor matching, RANSAC homography
estimation, and image warping onto a common canvas &mdash; exposed as a small,
stable C API (`cusift.h`, every symbol prefixed `cusift_`) usable from C, C++,
and Python.

This is a maintained fork of Mårten Björkman's
[Celebrandil/CudaSift](https://github.com/Celebrandil/CudaSift), restructured
into a proper library with a clean public interface, fused pipelines,
deterministic RANSAC, error reporting, device-memory estimation, and Python
bindings. Version 3.0 renamed the whole public surface; see
[docs/migration-3.0.md](docs/migration-3.0.md) for the old-to-new tables.

---

## Features

- **SIFT detection and 128-d descriptors** on the GPU using the original
  CudaSift Difference-of-Gaussians pyramid and Lowe's descriptor.
- **Brute-force descriptor matching** on the GPU, recording the best and
  second-best similarity per keypoint for the ratio test.
- **RANSAC** estimators for both an 8-DOF projective **homography** (4-point
  DLT) and a 4-DOF **similarity** transform (2-point closed form), followed by
  **least-squares refinement** over the inliers (AVX2/FMA on x86 hosts).
- **Image warping** to a common canvas (CPU or GPU), plus the canvas geometry
  on its own: size, origin and where each image's corners land.
- **Pipelines** that fuse extract, match, estimate and warp into one call,
  reusing the device-resident images across stages.
- **Error reporting without exceptions or return codes**: a thread-local error
  record after every call, an optional size-capped log file, and an optional
  callback, so the library is usable from C and from other languages.
- **Device-memory estimation** for every stage, so callers can pre-flight
  their budget before allocating.
- **Embedded-ready**: validated inputs, no silent clamps, every allocation
  checked, a process-local device reset for recovery from sticky CUDA errors,
  and a build default that covers every GPU the toolkit supports.
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
    RAII_Guards.hpp          Device/host pointer guards
    cudaSift.h               Internal aggregate header
test/
    main.cpp                 20-test correctness/regression suite
    speed.cpp                Multi-resolution benchmarking harness
    img1.png, img2.png       Sample image pair
docs/
    migration-3.0.md         2.x -> 3.0 rename tables
    image-stitching-with-homography.pdf
                             How the library goes from matches to aligned canvases
    common-canvas-mathematics.pdf
                             The projective geometry of the canvas and the warp
    kernel-correctness-fixes-v2.3.0.md
                             v2.3.0 kernel fixes, and what initial_blur means
cusift_py/
    cusift/                  Python package (ctypes bindings)
examples/
    cuda_optimize.py         Optuna-based parameter tuning example
CMakeLists.txt
cmake/
    cusiftConfig.cmake.in    Template consumed by find_package(cusift)
```

## Requirements

- CUDA Toolkit 11.0 or newer (tested against 13.x).
- A C++17 compiler (MSVC 2019+, GCC 9+, Clang 10+).
- CMake 3.23 or newer.
- (Optional) An AVX2/FMA-capable x86 CPU for the vectorised refinement path;
  other hosts use the scalar fallback.
- (Optional) Python 3.8+ for `cusift_py`.

By default the build compiles one cubin per **major** architecture your
toolkit supports plus PTX for the newest (`CUSIFT_CUDA_ARCHITECTURES=all-major`).
A cubin for X.0 runs on every X.y, so this covers every GPU the toolkit can
drive, including Jetson Orin (8.7). For a build tuned to one board pass
`-DCUSIFT_CUDA_ARCHITECTURES=87` (Orin) or `72` (Xavier, CUDA 11/12 toolkits;
CUDA 13 no longer supports Volta-class parts), and pass
`-DCUSIFT_ALL_CUDA_ARCHS=ON` for a cubin per minor architecture. The project
writes `CMAKE_CUDA_ARCHITECTURES` from this setting on every configure, so a
value cached by an earlier configure (or passed directly) is replaced, with a
status message saying so.

## Building

```bash
git clone https://github.com/rct-scientific-proc/CudaSift.git
cd CudaSift
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

CMake options:

| Option                      | Default     | Description                                     |
| --------------------------- | ----------- | ----------------------------------------------- |
| `CUSIFT_BUILD_SHARED`       | `OFF`       | Build `cusift` as a shared library (DLL/.so).   |
| `CUSIFT_USE_FAST_MATH`      | `OFF`       | Compile CUDA with `--use_fast_math`.            |
| `CUSIFT_CUDA_ARCHITECTURES` | `all-major` | Architectures to build for (`CMAKE_CUDA_ARCHITECTURES` syntax). |
| `CUSIFT_ALL_CUDA_ARCHS`     | `OFF`       | Build for every architecture supported by NVCC. |
| `CUSIFT_ENABLE_AVX2`        | `ON`        | Vectorise the refinement on x86 hosts.          |

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
Results: 20 passed, 0 failed, 20 total
```

For benchmarking across multiple input resolutions:

```bash
./build/Release/cusift_speed test/img1.png test/img2.png
```

## Quick start (C/C++)

```cpp
#include "cusift.h"
#include <vector>
#include <cstdio>

int main()
{
    cusift_init();

    // Two grayscale float32 images, row-major, values in [0, 255], in host memory.
    std::vector<float> img1, img2;
    int w1, h1, w2, h2;
    /* ... fill in img1, img2, w1/h1, w2/h2 ... */

    cusift_image in1{img1.data(), w1, h1};
    cusift_image in2{img2.data(), w2, h2};

    cusift_extract_options extract_opts;
    cusift_default_extract_options(&extract_opts);
    extract_opts.contrast_threshold = 2.0f;

    cusift_homography_options homography_opts;
    cusift_default_homography_options(&homography_opts);
    homography_opts.seed = 42;        // repeatable; 0 draws a random seed

    cusift_keypoints kp1{}, kp2{};    // zero-initialise once; reusable across calls
    float H[9];
    int inliers = 0;

    cusift_align(&in1, &in2, &kp1, &kp2, &extract_opts, &homography_opts, H, &inliers);

    cusift_error_record err;
    if (cusift_last_error(&err)) {
        std::fprintf(stderr, "%s: %s (%s:%d)\n", err.function, err.message, err.file, err.line);
        return 1;
    }

    /* H is a row-major 3x3 mapping image-1 to image-2 coordinates;
       inliers is the RANSAC inlier count. */

    cusift_keypoints_free(&kp1);
    cusift_keypoints_free(&kp2);
}
```

To get the two images on a common canvas as well, call `cusift_align_and_warp`
with two extra `cusift_image` outputs (release them with `cusift_image_free`),
or `cusift_align_and_warp_to_device` to keep them on the GPU.

The build exports a `cusift::cusift` CMake target you can consume from a
downstream project after `find_package(cusift CONFIG REQUIRED)` (the install
step writes `cusiftConfig.cmake` and a versioned `cusiftConfigVersion.cmake`)
or via `add_subdirectory()`:

```cmake
find_package(cusift CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE cusift::cusift)
```

## Quick start (Python)

```python
import numpy as np
from PIL import Image
from cusift import CuSift, HomographyOptions

sift = CuSift()                      # finds build/Release/cusift.dll or libcusift.so

img1 = np.asarray(Image.open("test/img1.png").convert("L"), dtype=np.float32)
img2 = np.asarray(Image.open("test/img2.png").convert("L"), dtype=np.float32)

kp1, kp2, matches, H, inliers = sift.align(
    img1, img2, homography_options=HomographyOptions(seed=42))
print(f"{len(kp1)} and {len(kp2)} keypoints, {inliers} inliers, H =\n{H}")
kp1.free(); kp2.free()

# or stage by stage
kp1, kp2 = sift.extract(img1), sift.extract(img2)
matches = sift.match(kp1, kp2)
H, inliers = sift.estimate_homography(kp1)
warped1, warped2 = sift.warp(img1, img2, H)       # NaN where no source covers a pixel
canvas = sift.compute_canvas(H, *img1.shape[::-1], *img2.shape[::-1])

# pre-flight device-memory estimate in bytes (also memory_extract/_match/_homography/_warp)
need = sift.memory_pipeline(*img1.shape[::-1], *img2.shape[::-1])
```

`ExtractOptions` and `HomographyOptions` carry the same field names and
defaults as the C structs. See `cusift_py/cusift/cusift.py` for the full
surface and `cusift_py/test.py` for a runnable example.

## API surface

All public functions live in [`include/cusift.h`](include/cusift.h) and are
documented inline.

| Function | Purpose |
| --- | --- |
| `cusift_version`, `cusift_init`, `cusift_device_reset` | Library version; device selection; process-local recovery from a sticky CUDA error. |
| `cusift_last_error`, `cusift_set_error_log`, `cusift_set_error_callback`, `cusift_error_log_dropped` | Error reporting: thread-local record, log file, callback. |
| `cusift_default_extract_options`, `cusift_default_homography_options` | Fill an options struct with the documented defaults. |
| `cusift_extract` | DoG detection + descriptors. |
| `cusift_match` | Brute-force descriptor matching of a query set against a train set. |
| `cusift_estimate_homography` | RANSAC homography or similarity with refinement; several attempts via the options. |
| `cusift_compute_canvas` | Canvas size, origin and corner positions for a homography, without warping. |
| `cusift_warp`, `cusift_warp_to_device` | Warp both images to the common canvas, output on the host or the device. |
| `cusift_extract_and_match`, `cusift_align`, `cusift_align_and_warp`, `cusift_align_and_warp_to_device` | Fused pipelines. |
| `cusift_keypoints_free`, `cusift_image_free`, `cusift_device_image_free` | Resource release. |
| `cusift_keypoints_save_json` | Persist a keypoint set, descriptors included. |
| `cusift_memory_extract`, `_match`, `_homography`, `_warp`, `_pipeline` | Pre-flight device-memory accounting per stage. |

## Error handling

The library never throws across the C boundary and never returns error codes.
Every public function (except the resource-release and error-sink setters)
clears the thread-local error state on entry; ask for it after the call:

```c
cusift_error_record err;
if (cusift_last_error(&err)) {
    /* err.function, err.file, err.line, err.cuda_error, err.cuda_error_name,
       err.message; the strings stay valid until the next library call on
       this thread */
}
```

Argument problems (NULL pointers, out-of-range options, images outside
8 to 65535 pixels per side, a cudaMalloc pointer passed as a host image) are
reported the same way rather than by crashing.

### Error log and callback

Every error, and the few warnings the library can emit (the AVX2 refinement
path refusing to run on a CPU without AVX2, a device free failing during
cleanup), can also be appended to a file and delivered to a callback,
independent of the thread-local state:

```c
cusift_set_error_log("", 0);                        /* "" = <temp dir>/cusift-errors.log, 0 = 1 MiB cap */
cusift_set_error_log("/var/log/cusift.log", 4u << 20);
cusift_set_error_log(NULL, 0);                      /* off (the default) */
cusift_set_error_callback(my_handler, user_data);   /* NULL disables */
```

Each line carries a UTC timestamp, severity, thread id, the public function,
source location, CUDA error code and name, and the message. When a line would
push the file past the cap, the file is renamed to `<path>.1` and restarted, so
disk use is bounded at twice the cap. Logging never fails the call being
logged; records that could not be written are counted by
`cusift_error_log_dropped()`. The file can also be enabled without a code
change through `CUSIFT_ERROR_LOG=<path>` (or `default`) and
`CUSIFT_ERROR_LOG_MAX_BYTES`, read on the first error if `cusift_set_error_log`
was never called.

## Device memory estimation

The `cusift_memory_*` family sizes the GPU budget without allocating anything:

```c
size_t bytes = cusift_memory_pipeline(w1, h1, w2, h2, &extract_opts, &homography_opts);
```

The peak typically occurs during SIFT extraction (the scale-space pyramid).
Returned values include all temporary buffers as well as the caller-visible
keypoint and warped-image allocations. Once a homography is known,
`cusift_compute_canvas` gives the exact warp output size instead of the 2x
bound the estimate assumes.

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
