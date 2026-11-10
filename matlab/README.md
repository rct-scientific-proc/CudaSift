# cusift from MATLAB

`cusift_matlab` is a second shared library built from the same sources as
`cusift`, with a flat C surface ([include/cusift_matlab.h](../include/cusift_matlab.h))
that MATLAB's `loadlibrary` can parse. There is no MEX file and nothing is
compiled with MATLAB's toolchain. The `CuSift` class in this directory wraps
the `calllib` calls.

## Why a second library

`loadlibrary` cannot represent what `cusift.h` uses: a struct that points to
another struct (`cusift_keypoints`), fixed-size array fields
(`descriptor[128]`, `corners[4][2]`), a callback function pointer and `bool`.
The façade keeps keypoint sets and warped images inside the library behind
`uint64` handles, passes arrays as plain `float *` / `int *`, and reports
every failure through an `int` status plus `cusiftml_last_error()`.

The library is also built to be self-contained: `CUSIFT_STATIC_RUNTIME=ON`
links the C and C++ runtimes into it (`/MT` on MSVC, `-static-libstdc++` on
GCC), so it loads regardless of the `msvcp140.dll` or `libstdc++.so.6` MATLAB
brought along. That is safe because the API boundary is C, no C++ object or
exception crosses it, and the library frees everything it allocates.

## Build

```bash
cmake -S . -B build -DCUSIFT_BUILD_MATLAB=ON -DCUSIFT_STATIC_RUNTIME=ON
cmake --build build --config Release
```

This produces `build/Release/cusift_matlab.dll` (Windows) or
`build/libcusift_matlab.so` (Linux) next to the usual outputs. On Windows the
only non-system import left is `vcomp140.dll` (OpenMP), which the VC++
redistributable that MATLAB installs provides; add
`-DCUSIFT_ENABLE_OPENMP=OFF` to drop it at the cost of a single-threaded
host warp. On Linux, check the result with

```bash
readelf -d build/libcusift_matlab.so | grep NEEDED      # no libstdc++.so.6
objdump -T build/libcusift_matlab.so | grep -c GLIBCXX   # 0
```

`cmake --install build --prefix <prefix>` puts the library in `<prefix>/bin`
(Windows) or `<prefix>/lib`, the header in `<prefix>/include`, and these
MATLAB files in `<prefix>/share/cusift/matlab`.

## Load

Add this directory to the MATLAB path. `CuSift()` loads the library through
`cusift_load`, which looks next to these files and then in `../build/Release`,
`../build` and the install layout; pass a directory to look elsewhere.

`loadlibrary` parses the header with the C compiler selected by
`mex -setup C`. The free MinGW-w64 support package is enough. To use the
library on machines without a compiler, generate a prototype file once from a
checkout:

```matlab
cusift_load('', true)     % writes matlab/cusift_matlab_proto.m
```

and ship it with the `.m` files; `cusift_load` prefers it when present.

## Use

```matlab
sift = CuSift();
img1 = imread('test/img1.png');
img2 = imread('test/img2.png');

[kp1, kp2, H, inliers] = sift.align(img1, img2, [], sift.homographyOptions('seed', 42));
[w1, w2] = sift.warp(img1, img2, H);              % NaN where no source covers a pixel

% or stage by stage
kp1 = sift.extract(img1);
kp2 = sift.extract(img2);
sift.match(kp1, kp2);
[H, inliers] = sift.estimateHomography(kp1);
canvas = sift.computeCanvas(H, size(img1), size(img2));

xy = kp1.xy();                 % N x 2, 0-based pixel coordinates
m  = kp1.match();              % N x 1, 1-based index into kp2, 0 = unmatched
d  = kp1.descriptors();        % 128 x N single
```

Run `cusift_demo` for the full sequence on the bundled images.

Images are passed as they come from `imread` (any numeric class, grayscale,
values 0–255); the class converts to `single` and handles the row-major
transpose. Homographies are 3×3 doubles mapping image-1 coordinates to
image 2, with `H(3,3) = 1`.

Options are plain structs: `sift.extractOptions('max_keypoints', 8192)` and
`sift.homographyOptions('model', 1, 'attempts', 3, 'goal', 0)`. Field names
and defaults are those of `cusift.h`.

## Lifetime

A `CuSiftKeypoints` object releases its set when it is deleted or `free`d.
Warped images are copied into MATLAB arrays and released immediately. Before
`unloadlibrary`, call `CuSift.unload()` (or `CuSift.freeAll()`): unloading
alone does not free GPU memory. `resetDevice()` frees every handle first,
because the CUDA reset destroys their device memory.

## Limits

- No error callback: `loadlibrary` cannot pass a MATLAB function into C.
  Use `setErrorLog` for a file, or `CuSift.lastError()` after a failure.
- The device-pointer functions of `cusift.h` (`*_to_device`) are not exposed;
  MATLAB has no use for a raw CUDA pointer.
- The MATLAB class was written without a MATLAB installation at hand; the
  library underneath it is tested from Python through the same C surface.
  If a `calllib` call errors on argument types, `libfunctionsview('cusift_matlab')`
  shows the signatures MATLAB derived from the header.
