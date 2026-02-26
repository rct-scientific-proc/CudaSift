"""
test.py - Exercise the cusift Python API: extract, match, homography, warp.

Usage:
    cd cusift_py
    python test.py
"""

import sys
from pathlib import Path

import numpy as np

import PIL.Image

# Ensure the cusift package is importable from this directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cusift import CuSift, ExtractOptions, HomographyOptions

# -- Paths --------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "local" / "data"
IMG1 = DATA_DIR / "img1.png"
IMG2 = DATA_DIR / "img2.png"


def main() -> None:
    print("=" * 60)
    print("CuSIFT Python API Test")
    print("=" * 60)

    # -- Initialise -------------------------------------------------------
    sift = CuSift()
    print("[OK] CuSift initialised\n")

    # -- Extract ----------------------------------------------------------
    extract_opts = ExtractOptions(
        thresh=3.0,
        lowest_scale=0.0,
        edge_thresh=10.0,
        init_blur=1.0,
        max_keypoints=32768,
        num_octaves=5,
    )

    print(f"Extracting SIFT features from {IMG1.name} ...")
    kp1 = sift.extract(str(IMG1), options=extract_opts)
    print(f"  → {len(kp1)} keypoints")

    print(f"Extracting SIFT features from {IMG2.name} ...")
    kp2 = sift.extract(str(IMG2), options=extract_opts)
    print(f"  → {len(kp2)} keypoints")

    if len(kp1) == 0 or len(kp2) == 0:
        print("ERROR: No keypoints extracted - cannot continue.")
        sys.exit(1)

    # Print a sample keypoint
    sample = kp1[0]
    print(f"\n  Sample keypoint: x={sample.x:.1f}  y={sample.y:.1f}  "
          f"scale={sample.scale:.3f}  orientation={sample.orientation:.3f}")
    print(f"  Descriptor (first 8): {sample.descriptor[:8]}")

    # -- Match ------------------------------------------------------------
    print("\nMatching features ...")
    matches = sift.match(kp1, kp2)
    print(f"  → {len(matches)} correspondences")

    if len(matches) == 0:
        print("ERROR: No matches found - cannot continue.")
        sys.exit(1)

    # Print a few matches
    for m in matches[:5]:
        print(f"    kp1[{m.query_index}] ({m.x1:.1f}, {m.y1:.1f}) "
              f"↔ kp2[{m.match_index}] ({m.x2:.1f}, {m.y2:.1f})  "
              f"err={m.error:.4f}")

    # -- Find Homography --------------------------------------------------
    print("\nFinding homography ...")
    homo_opts = HomographyOptions(
        num_loops=10000,
        min_score=0.0,
        max_ambiguity=0.80,
        thresh=5.0,
        improve_num_loops=5,
        improve_min_score=0.0,
        improve_max_ambiguity=0.80,
        improve_thresh=3.0,
        seed=42,
    )
    H, n_inliers = sift.find_homography(kp1, options=homo_opts)
    print(f"  → {n_inliers} inliers")
    print("  Homography:")
    for r in range(3):
        vals = " ".join(f"{H[r, c]:12.6f}" for c in range(3))
        print(f"    [{vals}]")

    # -- Warp Images ------------------------------------------------------
    print("\nWarping images ...")
    warped1, warped2 = sift.warp_images(str(IMG1), str(IMG2), H, use_gpu=True)
    print(f"  warped1 shape: {warped1.shape}  dtype: {warped1.dtype}")
    print(f"  warped2 shape: {warped2.shape}  dtype: {warped2.dtype}")

    # Optionally save warped images as PNGs
    try:
        from PIL import Image

        out1 = DATA_DIR / "img1_warped_py.png"
        out2 = DATA_DIR / "img2_warped_py.png"
        # NaN pixels (out-of-bounds padding from the warp kernel) → 0 (black)
        # Force all non-NaN values to be clipped to 1-255, and then set NaN values to 0
        warped1 = np.nan_to_num(np.clip(warped1, 1, 255), nan=0.0)
        warped2 = np.nan_to_num(np.clip(warped2, 1, 255), nan=0.0)
        Image.fromarray(warped1.astype(np.uint8), mode="L").save(out1)
        Image.fromarray(warped2.astype(np.uint8), mode="L").save(out2)
        print(f"\n  Saved warped images to:\n    {out1}\n    {out2}")
    except ImportError:
        print("\n  (Pillow not installed - skipping warped image save)")
    except Exception as e:
        print(f"\n  Error saving warped images: {e}")

    # -- Cleanup ----------------------------------------------------------
    kp1.free()
    kp2.free()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
