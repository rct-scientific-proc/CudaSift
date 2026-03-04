from cusift import CuSift, ExtractOptions, HomographyOptions

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the CLI argument parser and return parsed arguments."""
    p = argparse.ArgumentParser(
        prog="cuda_coreg",
        description=(
            "SIFT-based image coregistration using the CuSift GPU-accelerated library.\n"
            "Extracts SIFT features, matches them, estimates a homography via RANSAC,\n"
            "and warps the two images into alignment."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -- Positional ------------------------------------------------------
    p.add_argument("image1", type=str, help="Path to the first (reference) image.")
    p.add_argument("image2", type=str, help="Path to the second (target) image.")
    p.add_argument("output_dir", type=str, help="Directory where all outputs are saved.")

    # -- Library path ----------------------------------------------------
    p.add_argument(
        "--lib", type=str, default=None, metavar="PATH",
        help="Explicit path to the cusift shared library (cusift.dll / libcusift.so).",
    )

    # -- Upscaling option --------------------------------------------------
    p.add_argument(
        "--scale-factor", type=float, default=1.0, metavar="FACTOR",
        help="Upscale input images by the given factor (default: 1)."
    )

    # -- Extraction options ----------------------------------------------
    ext = p.add_argument_group("SIFT extraction parameters")
    ext.add_argument("--thresh", type=float, default=3.0,
                     help="DoG contrast threshold (default: 3.0).")
    ext.add_argument("--lowest-scale", type=float, default=0.0,
                     help="Minimum feature scale in pixels (default: 0.0).")
    ext.add_argument("--edge-thresh", type=float, default=10.0,
                     help="Edge rejection threshold (default: 10.0).")
    ext.add_argument("--init-blur", type=float, default=1.0,
                     help="Assumed input-image blur sigma (default: 1.0).")
    ext.add_argument("--max-keypoints", type=int, default=32768,
                     help="Maximum keypoints per image (default: 32768).")
    ext.add_argument("--num-octaves", type=int, default=5,
                     help="Number of scale-space octaves (default: 5).")
    ext.add_argument("--scale-suppression-radius", type=float, default=0.0,
                     help="Scale-NMS radius multiplier (0 = disabled). "
                          "Removes smaller-scale keypoints near larger ones. "
                          "6.0 is a good starting value (default: 0.0).")

    # -- Homography / RANSAC options -------------------------------------
    hom = p.add_argument_group("Homography / RANSAC parameters")
    hom.add_argument("--num-loops", type=int, default=10000,
                     help="RANSAC iterations (default: 10000).")
    hom.add_argument("--min-score", type=float, default=0.0,
                     help="Minimum match score (default: 0.0).")
    hom.add_argument("--max-ambiguity", type=float, default=0.90,
                     help="Maximum match ambiguity (default: 0.90).")
    hom.add_argument("--ransac-thresh", type=float, default=5.0,
                     help="RANSAC inlier distance threshold (default: 5.0).")
    hom.add_argument("--improve-num-loops", type=int, default=5,
                     help="Refinement iterations (default: 5).")
    hom.add_argument("--improve-min-score", type=float, default=0.0,
                     help="Refinement minimum match score (default: 0.0).")
    hom.add_argument("--improve-max-ambiguity", type=float, default=0.80,
                     help="Refinement maximum ambiguity (default: 0.80).")
    hom.add_argument("--improve-thresh", type=float, default=3.0,
                     help="Refinement inlier distance threshold (default: 3.0).")
    hom.add_argument("--seed", type=int, default=42,
                     help="RANSAC random seed; 0 = non-deterministic (default: 42).")

    # -- Visualisation options -------------------------------------------
    vis = p.add_argument_group("Visualisation parameters")
    vis.add_argument("--min-subsampling", type=float, default=0.0,
                     help="Only draw descriptors for keypoints whose subsampling >= this value (default: 0.0).")

    return p.parse_args(argv)


def _save_warped_image(pixels: np.ndarray, path: Path) -> None:
    """Normalise a float32 warped image and save as an 8-bit PNG."""
    from PIL import Image

    arr = np.nan_to_num(np.clip(pixels, 1, 255), nan=0.0)
    Image.fromarray(arr.astype(np.uint8), mode="L").save(str(path))


def _make_imfuse(warped1: np.ndarray, warped2: np.ndarray) -> np.ndarray:
    """Create a MATLAB-style imfuse false-color composite from two warped images.

    Both inputs are float32 arrays of the same shape.  NaN or 0 pixels
    are treated as background (no data).  The overlap region is used to
    linearly equalise the intensities of the two images to a common
    target (mean=128) before compositing.

    The output is an ``(H, W, 3)`` uint8 RGB array where image1 is
    mapped to magenta (R+B) and image2 to green (G).  Aligned areas
    appear grey/white; differences show as colour.

    Returns
    -------
    numpy.ndarray
        ``(H, W, 3)`` uint8 RGB composite.
    """
    # -- Build valid-pixel masks (non-NaN and non-zero) -------------------
    valid1 = np.isfinite(warped1) & (warped1 != 0)
    valid2 = np.isfinite(warped2) & (warped2 != 0)
    overlap = valid1 & valid2

    # -- Clean copies with background set to 0 ---------------------------
    img1 = np.where(valid1, warped1, 0.0).astype(np.float64)
    img2 = np.where(valid2, warped2, 0.0).astype(np.float64)

    # -- Intensity equalisation over the overlap region -------------------
    target_mean = 128.0
    if overlap.any():
        for img, valid in [(img1, valid1), (img2, valid2)]:
            region = img[overlap]
            mu = region.mean()
            sigma = region.std()
            if sigma > 1e-6:
                img[valid] = (img[valid] - mu) / sigma
                # Rescale so the overlap mean lands on target_mean with
                # a reasonable spread (target_std = 40 keeps most values
                # within 0-255 while preserving contrast).
                img[valid] = img[valid] * 40.0 + target_mean
            else:
                img[valid] = target_mean

    # -- Clip to [0, 255] and zero out background -------------------------
    img1 = np.clip(img1, 0, 255) * valid1
    img2 = np.clip(img2, 0, 255) * valid2

    u1 = img1.astype(np.uint8)
    u2 = img2.astype(np.uint8)

    # -- Compose: image1 → magenta (R+B), image2 → green (G) -------------
    rgb = np.zeros((*warped1.shape, 3), dtype=np.uint8)
    rgb[..., 0] = u1        # R ← image1
    rgb[..., 1] = u2        # G ← image2
    rgb[..., 2] = u1        # B ← image1

    return rgb


def _filter_inliers(
    matches: list,
    H: np.ndarray,
    thresh: float,
) -> list:
    """Return only matches consistent with the homography *H*.

    A match is an inlier when the Euclidean distance between the
    projected point ``H @ [x1, y1, 1]`` and ``[x2, y2]`` is at most
    *thresh* pixels — the same geometric test RANSAC uses.
    """
    if len(matches) == 0:
        return []

    pts1 = np.array([[m.x1, m.y1] for m in matches], dtype=np.float64)
    ones = np.ones((len(matches), 1), dtype=np.float64)
    pts1_h = np.hstack([pts1, ones])                    # (N, 3)
    proj = (H @ pts1_h.T).T                             # (N, 3)
    proj_xy = proj[:, :2] / proj[:, 2:3]                # perspective divide

    pts2 = np.array([[m.x2, m.y2] for m in matches], dtype=np.float64)
    errors = np.linalg.norm(proj_xy - pts2, axis=1)

    return [m for m, e in zip(matches, errors) if e <= thresh]

from scipy.interpolate import RectBivariateSpline
def upscale_image(img: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """Upscale *img* by *factor* using bicubic spline interpolation.

    Returns the original array unchanged when *factor* is 1.0.
    """
    if factor == 1.0:
        return img
    h, w = img.shape
    x = np.arange(w)
    y = np.arange(h)
    f = RectBivariateSpline(y, x, img, kx=3, ky=3)
    x_new = np.linspace(0, w - 1, int(w * factor))
    y_new = np.linspace(0, h - 1, int(h * factor))
    return f(y_new, x_new).astype(np.float32)
    

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # -- Validate inputs -------------------------------------------------
    img1_path = Path(args.image1)
    img2_path = Path(args.image2)
    out_dir = Path(args.output_dir)

    if not img1_path.is_file():
        print(f"Error: image1 not found: {img1_path}", file=sys.stderr)
        sys.exit(1)
    if not img2_path.is_file():
        print(f"Error: image2 not found: {img2_path}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # -- Build option structs --------------------------------------------
    extract_opts = ExtractOptions(
        thresh=args.thresh,
        lowest_scale=args.lowest_scale,
        edge_thresh=args.edge_thresh,
        init_blur=args.init_blur,
        max_keypoints=args.max_keypoints,
        num_octaves=args.num_octaves,
        scale_suppression_radius=args.scale_suppression_radius,
    )

    homography_opts = HomographyOptions(
        num_loops=args.num_loops,
        min_score=args.min_score,
        max_ambiguity=args.max_ambiguity,
        thresh=args.ransac_thresh,
        improve_num_loops=args.improve_num_loops,
        improve_min_score=args.improve_min_score,
        improve_max_ambiguity=args.improve_max_ambiguity,
        improve_thresh=args.improve_thresh,
        seed=args.seed,
    )

    # -- Initialise CuSift -----------------------------------------------
    print("Initializing CuSift ...")
    sift = CuSift(dll_path=args.lib)
    print("[OK] CuSift initialized.\n")

    # -- Run the full pipeline -------------------------------------------
    print(f"Image 1: {img1_path}")
    print(f"Image 2: {img2_path}")
    print("Running full pipeline (extract → match → homography → warp) ...")

    # Load images to numpy arrays first
    from PIL import Image
    img1_raw = np.array(Image.open(str(img1_path)).convert("L"), dtype=np.float32)
    img2_raw = np.array(Image.open(str(img2_path)).convert("L"), dtype=np.float32)

    # Upscale if requested (scale_factor=1.0 is a no-op)
    scale_factor = args.scale_factor
    img1 = upscale_image(img1_raw, scale_factor)
    img2 = upscale_image(img2_raw, scale_factor)

    t_start = time.perf_counter()
    kp1, kp2, matches, H, n_inliers, warped1, warped2 = (
        sift.extract_and_match_and_find_homography_and_warp(
            img1,
            img2,
            extract_options=extract_opts,
            homography_options=homography_opts,
        )
    )
    t_elapsed = time.perf_counter() - t_start

    print(f"  Pipeline completed in {t_elapsed:.4f} s")
    print(f"  Keypoints image1: {len(kp1)}")
    print(f"  Keypoints image2: {len(kp2)}")
    print(f"  Matches:          {len(matches)}")
    print(f"  RANSAC inliers:   {n_inliers}")
    print(f"  Warped1 shape:    {warped1.shape}")
    print(f"  Warped2 shape:    {warped2.shape}")
    print("  Homography:")
    for r in range(3):
        vals = " ".join(f"{H[r, c]:12.6f}" for c in range(3))
        print(f"    [{vals}]")

    # -- Save warped images ----------------------------------------------
    warped1_path = out_dir / "warped_image1.png"
    warped2_path = out_dir / "warped_image2.png"
    _save_warped_image(warped1, warped1_path)
    _save_warped_image(warped2, warped2_path)
    print(f"\n  Saved warped images:")
    print(f"    {warped1_path}")
    print(f"    {warped2_path}")

    # -- Save false-color composite (imfuse-style) ------------------------
    from PIL import Image as PILImage
    composite = _make_imfuse(warped1, warped2)
    composite_path = out_dir / "composite_imfuse.png"
    PILImage.fromarray(composite, mode="RGB").save(str(composite_path))
    print(f"  Saved imfuse composite: {composite_path}")

    # -- Save homography matrix (plain text) ------------------------------
    homography_path = out_dir / "homography.txt"
    np.savetxt(str(homography_path), H, fmt="%.10f")
    print(f"  Saved homography matrix: {homography_path}")

    # -- Save keypoints as JSON (without descriptors) ----------------------
    def _keypoint_to_dict(kp):
        return {
            "x": float(kp.x),
            "y": float(kp.y),
            "scale": float(kp.scale),
            "sharpness": float(kp.sharpness),
            "edgeness": float(kp.edgeness),
            "orientation": float(kp.orientation),
            "score": float(kp.score),
            "ambiguity": float(kp.ambiguity),
            "subsampling": float(kp.subsampling),
        }

    kp1_data = {
        "num_keypoints": len(kp1),
        "keypoints": [_keypoint_to_dict(kp) for kp in kp1],
    }
    kp2_data = {
        "num_keypoints": len(kp2),
        "keypoints": [_keypoint_to_dict(kp) for kp in kp2],
    }

    kp1_path = out_dir / "keypoints_image1.json"
    kp2_path = out_dir / "keypoints_image2.json"
    kp1_path.write_text(json.dumps(kp1_data, indent=2))
    kp2_path.write_text(json.dumps(kp2_data, indent=2))
    print(f"  Saved keypoints: {kp1_path}, {kp2_path}")

    # -- Save matches as JSON ---------------------------------------------
    def _match_to_dict(m):
        return {
            "query_index": m.query_index,
            "match_index": m.match_index,
            "x1": float(m.x1),
            "y1": float(m.y1),
            "x2": float(m.x2),
            "y2": float(m.y2),
            "error": float(m.error),
            "score": float(m.score),
            "ambiguity": float(m.ambiguity),
        }

    matches_data = {
        "num_matches": len(matches),
        "matches": [_match_to_dict(m) for m in matches],
    }
    matches_path = out_dir / "matches.json"
    matches_path.write_text(json.dumps(matches_data, indent=2))
    print(f"  Saved matches: {matches_path}")

    # -- Save descriptors as binary (Mx128 float32) -----------------------
    # Each file is a contiguous block of M*128 IEEE 754 float32 values
    # (little-endian on x86).  To reload: np.fromfile(path, dtype=np.float32).reshape(M, 128)
    desc1 = np.array([kp.descriptor for kp in kp1], dtype=np.float32)
    desc2 = np.array([kp.descriptor for kp in kp2], dtype=np.float32)

    desc1_path = out_dir / "descriptors_image1.bin"
    desc2_path = out_dir / "descriptors_image2.bin"
    desc1.tofile(str(desc1_path))
    desc2.tofile(str(desc2_path))
    print(f"  Saved descriptors (float32 binary): {desc1_path} ({desc1.shape}), {desc2_path} ({desc2.shape})")

    # -- Write JSON metadata ---------------------------------------------
    metadata = {
        "image1": str(img1_path.resolve()),
        "image2": str(img2_path.resolve()),
        "pipeline_time_seconds": round(t_elapsed, 6),
        "num_keypoints_image1": len(kp1),
        "num_keypoints_image2": len(kp2),
        "num_matches": len(matches),
        "num_inliers": n_inliers,
        "homography": H.tolist(),
        "warped1_shape": list(warped1.shape),
        "warped2_shape": list(warped2.shape),
        "extract_options": {
            "thresh": extract_opts.thresh,
            "lowest_scale": extract_opts.lowest_scale,
            "edge_thresh": extract_opts.edge_thresh,
            "init_blur": extract_opts.init_blur,
            "max_keypoints": extract_opts.max_keypoints,
            "num_octaves": extract_opts.num_octaves,
        },
        "homography_options": {
            "num_loops": homography_opts.num_loops,
            "min_score": homography_opts.min_score,
            "max_ambiguity": homography_opts.max_ambiguity,
            "thresh": homography_opts.thresh,
            "improve_num_loops": homography_opts.improve_num_loops,
            "improve_min_score": homography_opts.improve_min_score,
            "improve_max_ambiguity": homography_opts.improve_max_ambiguity,
            "improve_thresh": homography_opts.improve_thresh,
            "seed": homography_opts.seed,
        },
        "outputs": {
            "warped_image1": str(warped1_path),
            "warped_image2": str(warped2_path),
            "homography_txt": str(homography_path),
            "keypoints_image1": str(kp1_path),
            "keypoints_image2": str(kp2_path),
            "matches": str(matches_path),
            "descriptors_image1": str(desc1_path),
            "descriptors_image2": str(desc2_path),
            "descriptor_format": "contiguous float32 little-endian, shape (M, 128)",
            "composite_imfuse": str(composite_path),
        },
    }
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"  Saved metadata: {meta_path}")

    # -- Visualizations --------------------------------------------------
    print("\nGenerating visualizations ...")

    # Use the (potentially upscaled) images for all visualizations so that
    # keypoint coordinates match the pixel grid.
    img1_h, img1_w = img1.shape
    img2_h, img2_w = img2.shape

    # Save upscaled greyscale PNGs for draw_matches (which requires file paths)
    upscaled_img1_path = vis_dir / "_upscaled_image1.png"
    upscaled_img2_path = vis_dir / "_upscaled_image2.png"
    Image.fromarray(np.clip(img1, 0, 255).astype(np.uint8), mode="L").save(str(upscaled_img1_path))
    Image.fromarray(np.clip(img2, 0, 255).astype(np.uint8), mode="L").save(str(upscaled_img2_path))

    # Keypoints on each image
    kp1_vis = vis_dir / "keypoints_image1.png"
    kp2_vis = vis_dir / "keypoints_image2.png"
    sift.draw_keypoints(img1, kp1, str(kp1_vis), width=img1_w, height=img1_h)
    sift.draw_keypoints(img2, kp2, str(kp2_vis), width=img2_w, height=img2_h)
    print(f"  Saved keypoint visualizations: {kp1_vis}, {kp2_vis}")

    # Match visualization (side-by-side)
    if matches:
        matches_vis = vis_dir / "matches.png"
        sift.draw_matches(str(upscaled_img1_path), str(upscaled_img2_path), matches, str(matches_vis))
        print(f"  Saved match visualization: {matches_vis}")

        # Inlier-only matches (reprojection error <= RANSAC threshold)
        inlier_matches = _filter_inliers(matches, H, args.ransac_thresh)
        if inlier_matches:
            inliers_vis = vis_dir / "matches_inliers.png"
            sift.draw_matches(
                str(upscaled_img1_path), str(upscaled_img2_path), inlier_matches, str(inliers_vis),
            )
            print(f"  Saved inlier match visualization ({len(inlier_matches)} inliers): {inliers_vis}")

    # Descriptor visualizations (filtered by --min-subsampling)
    if args.min_subsampling > 0:
        desc1_vis = vis_dir / "descriptors" / "image1"
        desc2_vis = vis_dir / "descriptors" / "image2"
        sift.draw_descriptors(
            img1, kp1, args.min_subsampling, str(desc1_vis),
            width=img1_w, height=img1_h,
        )
        sift.draw_descriptors(
            img2, kp2, args.min_subsampling, str(desc2_vis),
            width=img2_w, height=img2_h,
        )
        print(f"  Saved descriptor visualizations to: {desc1_vis}, {desc2_vis}")

    # -- Cleanup ---------------------------------------------------------
    kp1.free()
    kp2.free()

    print(f"\nDone. All results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()







