# Kernel correctness fixes (v2.3.0)

> Written against the 2.x API. Option names below are the 2.x ones
> (`init_blur_`, `thresh_`, ...); see docs/migration-3.0.md for their 3.0
> equivalents (`initial_blur`, `contrast_threshold`, ...). The kernel
> analysis is unaffected.

This release fixes seven correctness defects in the CUDA kernels. Two of them
affected every extraction run at the library's own documented default settings,
so **detection results change in this release**. If you have tuned thresholds
against v2.2.0 or earlier, re-check them; see [Upgrade notes](#upgrade-notes).

Nothing in the public API changed. `cusift.h` has the same functions and structs,
and the only header edit is documentation for `init_blur_`.

Everything below was measured on an RTX 4070 (sm_89), CUDA 13.4, with
`CUSIFT_USE_FAST_MATH=ON`, using the `test/img1.png` / `test/img2.png` pair
(1920x1080) unless stated otherwise.

## Summary

| ID | Commit | Severity | Area | Fires at default settings? |
|----|--------|----------|------|----------------------------|
| K1 | `c08e058` | Critical | Scale-space construction (`cudaSiftH.cu`, `cusift.h`) | Yes |
| K2 | `658eaaa` | High | Descriptors (`cudaSiftD.cu`) | Only on exactly-flat image content |
| K3 | `61dc181` | Low | Sub-pixel refinement (`cudaSiftD.cu`) | No (guard) |
| K4 | `900f7ab` | Low | Orientation assignment (`cudaSiftD.cu`) | No (guard) |
| K5 | `4294875` | Medium | `MinMaxKernel` (`cudaImage.cu`) | No (unreachable code) |
| K6 | `131f3da` | Low | `WarpDualKernel` (`cusift.cu`) | No (latent) |
| K7 | `27cea46` | Cleanup | `cudaSiftD.cu`, `matching.cu` | n/a |

Net effect on the test pair, at the default `init_blur_ = 1.0`:

| | v2.2.0 | v2.3.0 |
|---|---|---|
| Keypoints, image 1 | 4541 | 3986 |
| Homography inliers | 1279 | 1493 |
| Inlier rate | 28.2% | 37.5% |

Fewer keypoints, more of them geometrically consistent. The drop in raw count is
expected and explained under K1.

## Upgrade notes

1. **Detection results change.** K1 repairs the two finest scale levels of the
   finest octave, which previously produced nothing, and simultaneously makes
   every pyramid level correctly blurred (previously each level had
   `init_blur^2` wrongly subtracted out of its Gaussian variance). On detailed
   photographs the net is fewer but better keypoints; on noise-dominated or
   synthetic content the count rises.

2. **`init_blur_` now behaves as a smooth quality/quantity dial** and is
   meaningful across its whole range. Before this release it was effectively
   unusable above about 1.0. See
   [init_blur_: semantics and usable values](#init_blur_-semantics-and-usable-values).

3. **Rebuild, and check which library you load.** `cusift_py` searches
   `build/Release/cusift.dll` first, then `build/Debug/`, then `build/`. A stale
   artifact in an earlier directory silently wins. If you use the Python
   bindings, confirm the DLL you are loading is from this release.

4. **Descriptors from flat regions changed meaning.** They used to be a constant
   vector that matched everything (K2). They are now either real descriptors or,
   for a window with no gradient at all, the zero vector, which scores 0 against
   everything. If you built any downstream filtering around the old behaviour,
   that filtering is no longer needed.

## init_blur_: semantics and usable values

### What it means

`init_blur_` is the sigma of a Gaussian pre-blur applied to the input image
before the scale-space pyramid is built. The pyramid's own scales are measured
*relative* to that pre-blurred base, so in the finest octave a keypoint reported
at scale `s` sits at a total image blur of roughly `sqrt(init_blur_^2 + s^2)`.

This is upstream CudaSift's convention and is what the code now implements.
`ExtractSift()` passes `0.0f` to `PrepareLaplaceKernels()`; the pre-blur is
applied once, by the `LowPass()` call, and is not subtracted again from each
level's variance.

The previous header text ("the library applies a low-pass filter so that the
effective blur of the base image matches the first scale of the Gaussian
pyramid") described a different, absolute-scale convention that the code did not
correctly implement, and which is impossible for `init_blur_ >= 0.87` because the
finest octave's first level sits at `2^(-1/5) = 0.8706`. That text has been
replaced.

### Measured behaviour

Same image pair, same options, only `init_blur_` varying
(`max_keypoints_ = 32768`, `num_octaves_ = 5`, `thresh_ = 2.0`,
`edge_thresh_ = 10.0`, RANSAC `thresh_ = 3.0`, `seed_ = 42`):

| `init_blur_` | kp img1 (v2.2.0) | inlier % (v2.2.0) | kp img1 (v2.3.0) | inlier % (v2.3.0) |
|---|---|---|---|---|
| 0.00 | 9096 | 20.5% | 9096 | 21.0% |
| 0.25 | 9496 | 19.8% | 9085 | 18.7% |
| 0.50 | 9152 | 20.1% | 7567 | 24.6% |
| 0.75 | 7622 | 21.4% | 5789 | 28.7% |
| **1.00** | **4541** | **28.2%** | **3986** | **37.5%** |
| 1.25 | 21679 | 18.2% | 3069 | 42.0% |
| 1.50 | 16046 | 22.7% | 2593 | 42.2% |
| 2.00 | 1692 | 41.3% | 1972 | 45.6% |
| 3.00 | 437 | 43.2% | 1524 | 49.9% |

The v2.2.0 column is not a smooth curve. At 1.25 and 1.50 it produced 21679 and
16046 keypoints, roughly five times the count at 1.00, at a *lower* inlier rate
than any other setting. Those were spurious extrema manufactured by degenerate
Gaussian kernels, not features.

In v2.3.0 the keypoint count falls monotonically across the whole range and the
inlier rate rises steadily from 0.25 upward, so the parameter trades keypoint
count for geometric consistency, which is what it is supposed to do. The one
wrinkle is at the very bottom: 0.00 scores slightly better than 0.25 (21.0% vs
18.7%). With no pre-blur the detector is responding largely to pixel noise, and
which of those marginal detections survive RANSAC is not stable, so treat
differences of a couple of points down there as noise rather than signal.

As a sanity check on the K1 fix, note that both columns report exactly 9096
keypoints at `init_blur_ = 0.0`. That is expected: at zero the old
`var = scale_i^2 - initBlur^2` and the new `var = scale_i^2` are the same
expression, so the fix is a no-op there by construction and only the descriptor
and pi changes (K2, K7) move the inlier count.

### Recommended values

| Range | Use when |
|---|---|
| `0.0` – `0.25` | Clean or synthetic input, or you want maximum keypoints and will filter downstream. Expect the lowest inlier rate. |
| `0.5` – `0.75` | Mild sensor noise; a middle ground. |
| **`1.0` (default)** | **General-purpose starting point.** Matches upstream CudaSift's default and gives a good count/quality balance. |
| `1.25` – `2.0` | Noisy input, or when you want fewer, more reliable keypoints and faster matching (matching cost scales with keypoint count). |
| `3.0` and above | Very noisy input, or coarse alignment only. Keypoint count falls off substantially. |

### Valid domain and edge cases

Any finite value `>= 0` is valid and numerically safe. `PrepareLaplaceKernels()`
now clamps each level's Gaussian variance to a small positive value, so no
setting can produce an inf or NaN filter tap, and no setting produces a NaN
keypoint position, scale or descriptor.

Two rough edges are worth knowing rather than relying on:

- **Negative values and NaN are silently coerced, not rejected.** `ExtractSift()`
  applies `max(init_blur_, 0.001f)`, which maps a negative value to 0.001, and
  because that is `fmaxf` semantics it maps NaN to 0.001 as well. Both therefore
  behave like "no pre-blur". Pass a sensible non-negative value; do not depend on
  this coercion.
- **`+inf` degenerates to a 9-tap box blur.** The low-pass variance becomes
  infinite, every tap evaluates to 1, and normalisation yields a uniform 1/9
  filter. The result is finite but meaningless.

Argument validation covers `max_keypoints_` and `num_loops_` but not
`init_blur_`. Adding a finite/non-negative check would be a reasonable
follow-up.

## The fixes

### K1 — `init_blur_` was subtracted twice, producing NaN Gaussian levels (`c08e058`)

`ExtractSift()` low-passed the input by `init_blur_` and then passed the same
value into `PrepareLaplaceKernels()`, whose per-level variance is
`var = scale_i^2 - initBlur^2`. The blur was accounted for twice.

The finest octave's scale ladder starts at `scale_0 = 2^(-1/5) = 0.8706`, so for
any `init_blur_ >= 0.8706` the variance went non-positive and
`expf(-j*j/2/var)` produced inf or NaN taps. Replicating the kernel table in
float32 for the finest octave:

| `init_blur_` | level 0 taps `[j=0..4]` | level 1 taps |
|---|---|---|
| 0.0 | `0.458 0.237 0.033 0.001 0.000` | healthy |
| 0.9 | `0 0 0 0 nan` | healthy |
| **1.0 (default)** | `0 0 0 0 0.5` | `nan x5` |
| 1.5 | all weight on the outermost tap | likewise |

Consequences at the default setting: DoG planes 0 and 1 of the finest octave
were entirely NaN, so scale indices 0 and 1 never yielded a keypoint. In a band
around 1.25 to 1.50 the degenerate filters (all weight at `j = 4`, i.e.
averaging pixels eight apart) manufactured the keypoint explosion visible in the
sweep table; which levels degenerate, and how, depends on `init_blur_`, so the
failure is not a simple threshold.

The fix passes `0.0f`, matching upstream, and clamps `var` to a small positive
value so no caller can poison a DoG plane again. `init_blur_` is re-documented in
`cusift.h`.

On the 400-blob synthetic probe, finest-octave keypoints per in-octave scale
index:

| `init_blur_` | s0, s1 before | s0, s1 after |
|---|---|---|
| 0.88 | 0, 0 | 2, 2 |
| 1.00 | 0, 0 | 0, 2 |
| 1.50 | 0, 0 (plus 461 spurious s2, 856 total) | 0, 0 (370 total) |

Note also that `ExtractSiftLoop()`'s `initBlur` parameter only feeds its own
recursion; `ExtractSiftOctave()` does not take it, so it has no effect on output.
It is commented rather than removed, to keep the change minimal.

### K2 — flat regions produced a uniform descriptor that matched everything (`658eaaa`)

`FastAtan2` computes `__fdiv_rn(min(|x|,|y|), max(|x|,|y|))`, so an exactly-zero
gradient evaluated `0/0 = NaN`. Any sample whose two bilinear texture fetches are
bit-identical, which means any exactly-flat or saturated region and any window
clamped outside the image, fed that NaN into the descriptor's angular weights and
through `atomicAdd` into the orientation histogram.

The NaN never surfaced as a NaN. The peak-suppression step is
`min(buffer[idx] * rsqrtf(sum), 0.2f)`, and `min` here is `fminf`, which discards
a NaN operand and returns the other. Every one of the 128 bins was therefore
clipped to exactly 0.2 and normalised to the constant vector
`1/sqrt(128) = 0.08839`.

Critically, **one** flat sample anywhere in the 16x16 window was enough to
destroy the whole descriptor, because the NaN entered the sum of squares used for
normalisation. This was not limited to keypoints sitting in flat areas.

Measured on a 512x512 image containing four uniformly-bright flat disks on a flat
background:

| | v2.2.0 | v2.3.0 |
|---|---|---|
| Keypoints | 138 | 126 |
| Keypoints within 1.5 px of a disk edge | 92 | 92 |
| Uniform descriptors | 138 | 0 |
| Distinct descriptor vectors | **1** | **84** |
| Mean pairwise cosine similarity | 1.0000 | 0.6993 |
| Descriptor pairs above 0.99 similarity | 9453 of 9453 (100%) | 231 of 7875 (2.9%) |

Before the fix there was literally one unique descriptor on the whole image,
including the 92 keypoints on a strong disk edge, and all 9453 pairs were
perfectly similar. Those descriptors went into RANSAC as ideal correspondences.
Note that the L2 norm was 1.0 in both cases: 128 copies of `1/sqrt(128)`
normalise to 1, so norm checks cannot detect this.

Three guards were added: `FastAtan2` returns 0 for a zero gradient;
zero-gradient samples skip their eight shared-memory atomics, which is also
faster on flat regions; and both normalisation steps check for a zero sum and
emit a zero descriptor instead of dividing.

The keypoint count difference (138 vs 126) is K1, not K2. Edge detections are
identical at 92 in both builds; the 12 losses are interior detections that the
correctly-blurred pyramid no longer flags.

### K3 — the sub-pixel refinement fallback was unbounded (`61dc181`)

When the full 3x3 Newton step leaves the +/-0.5 box, `FindPointsMultiNew` falls
back to three independent 1-D steps `dx/dxx`, `dy/dyy`, `ds/dss`, and stored them
with no bound. Those denominators are second derivatives that can be arbitrarily
small, and `__fdividef(x, 0)` is +/-inf, so a keypoint could in principle be
reported at a position or scale outside the sample it was detected in.

Each fallback step is now clamped to +/-0.5, with a NaN mapping to 0.

This is a guard, not a fix for observed damage. Across about 5,500 keypoints on
four synthetic images no keypoint's stored in-octave scale fell outside the valid
band, and results are bit-identical after the change, which also confirms the
clamp does not disturb well-behaved refinements.

### K4 — the orientation peak fit could produce NaN (`900f7ab`)

`ComputeOrientationsCONST` non-maximum-suppresses the smoothed 32-bin histogram,
so a window with no strict local maximum (perfectly flat, or perfectly
symmetric) leaves every bin at zero, `i1` at -1 and `maxval1` at 0. The parabolic
peak fit `i1 + 0.5*(val1-val2)/(2*maxval1-val1-val2)` is then `0/0`, and a NaN
orientation makes the descriptor kernel sample the texture at NaN coordinates.
The denominator can also vanish with a positive maximum.

Both cases are guarded: a non-positive maximum yields orientation 0 and skips the
secondary-orientation branch, and a zero denominator falls back to the bin
centre. The probes never produced a NaN orientation in practice, so no measured
behaviour changes.

### K5 — `MinMaxKernel` could deadlock and mis-ordered negative values (`4294875`)

Three independent defects in the min/max reduction behind
`CudaImage_Normalize()`:

1. `__any_sync(0xffffffff, ...)` was called from inside `if (laneInWarp == 0)`. A
   full mask requires all 32 lanes to arrive and the other 31 never do. This is
   undefined behaviour, and in practice it hangs: a harness exercising
   `CudaImage_Normalize` against the old kernel produced no output and had to be
   killed after 600 s, while the same harness against the rewrite finishes in
   under a second.
2. The cross-warp fold read `sMin[lane + offset]` but never wrote partial results
   back to shared memory, so with 8 warps lane 0 only ever saw warps 0, 1, 2 and
   4. The other four were silently dropped and the block's min/max could be
   wrong.
3. `atomicMin/atomicMax((int *)d_min, __float_as_int(v))` only orders
   non-negative floats, and the result is read back on the host as a plain float.

The kernel now carries identity elements through a branch-free warp reduction,
folds the per-warp results serially in one thread, and uses CAS-based float
atomics correct over the whole float range.

This code is still unreachable: every `CudaImage_Normalize()` call site remains
commented out, so no shipping behaviour changes. It is now correct if anyone
enables it. Verified with a standalone harness covering single- and multi-block
images, all-negative and zero-spanning ranges, a ragged width that leaves lanes
out of range, a 33x4097 image and a constant image.

### K6 — `WarpDualKernel` wrote its second output with the first one's stride (`131f3da`)

`d_out1` and `d_out2` come from two independent `cudaMallocPitch` calls, but the
kernel took a single `dstPitch` and used it for both, while the host read
`d_out2` back using `out2Pitch`. The two pitches are equal on every current
driver, so this never misbehaved, but nothing guaranteed it and a mismatch would
have been an out-of-bounds device write. The kernel now takes `dst1Pitch` and
`dst2Pitch` and all five launch sites pass both.

### K7 — one definition of pi, two no-op barriers, one redundant kernel (`27cea46`)

- `cudaSiftD.cu` carried three values of pi: `3.14159274f` in `FastAtan2`,
  `3.1415f` in the descriptor kernel and `3.1416f` in the orientation kernel, so
  the angle a sample was assigned and the angle it was binned at disagreed by
  about 0.003%. All five sites now use `CUSIFT_PI_F` / `CUSIFT_HALFPI_F`.
  Bin-edge behaviour is unchanged, because `atan2f` saturates at exactly
  `CUSIFT_PI_F`.
- `ComputeHomographies` called `__syncthreads()` twice around a loop touching
  only per-thread locals. The barriers only serialised the block.
- `MatchSiftData_private` launched `CleanMatches` to zero every score before
  `FindMaxCorr10`, but `FindMaxCorr10` writes `score`, `match`, `match_xpos`,
  `match_ypos` and `ambiguity` for every `p1 < numPts1`. Both the launch and the
  kernel are removed.

## Verification

Against the v2.3.0 build:

- `cusift_demo` passes 17 of 17 tests.
- `compute-sanitizer --tool memcheck --leak-check full` over the full demo and
  over extraction on 4000x34, 1920x1080 and 640x481 inputs: 0 errors, 0 bytes
  leaked. This confirms the v2.2.0 memory-safety work still holds after the
  kernel changes here.
- The public-API argument guards added in v2.2.0 still pass, including
  `max_keypoints_` overflow rejection, NULL arguments, oversized images and
  `SiftData` reuse.
- No descriptor, orientation, position or scale is non-finite for any
  `init_blur_` in `{-5, -0.5, 0, 0.25, 0.5, 1, 2, 4, 8, 16, NaN, +inf}`.

## Not addressed

The performance findings from the same review are untouched. In rough order of
expected benefit:

- `LaplaceMultiMem` processes one image row per block, so every source row is
  fetched from global memory nine times across neighbouring blocks. `LAPLACE_H`
  is defined but unused. This is the heaviest pyramid kernel.
- `ExtractSift` performs a `cudaMalloc`/`cudaMemset`/`cudaFree` of its context
  block, a texture create/destroy per octave, a synchronous 20-byte copy of the
  constant `ScaleDown` taps per octave, and a `thrust::sort` temporary allocation
  on every call. An extractor handle that caches these across frames would remove
  all of it; that changes the public API.
- `thrust::sort` moves whole 576-byte `SiftPoint` records rather than sorting an
  index.
- `FindMaxCorr10` leaves half its threads idle in the dot-product phase; for
  large keypoint counts the similarity matrix is a GEMM.
- `FindPointsMultiNew` runs 32-thread blocks, capping occupancy, and reads the
  centre DoG plane twice.
- No kernel declares `__launch_bounds__`.

Two design characteristics inherited from upstream also remain: orientation uses
a fixed 11x11 window regardless of scale, and both orientation and descriptor
gradients are sampled from the octave's base image rather than the Gaussian level
matching the keypoint's scale.
