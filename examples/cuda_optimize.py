"""
cuda_optimize.py - Use Optuna to find CuSift parameters that minimise the
mean absolute difference (MAD) between overlapping pixels of two warped images.

Usage:
    python cuda_optimize.py <image1> <image2> <output_dir> [options]

The objective function:
  1. Runs the full CuSift pipeline (extract → match → homography → warp)
     with the trial's suggested parameters.
  2. Identifies the overlap region (both images have valid, non-zero pixels).
  3. Normalises both images to zero-mean / unit-std in the overlap.
  4. Returns the mean absolute difference (MAD) of the normalised overlap.
     Trials that produce no overlap or no inliers are pruned.

After optimisation the best parameters, warped images, and a composite are
saved to *output_dir*.
"""

from cusift import CuSift, CuSiftError, ExtractOptions, HomographyOptions

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import optuna
except ImportError:
    print(
        "Error: optuna is required.  Install it with:  pip install optuna",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _overlap_mad(warped1: np.ndarray, warped2: np.ndarray) -> float:
    """Mean absolute difference of normalised overlap pixels.

    Returns ``float('inf')`` when there is no overlap.
    """
    valid1 = np.isfinite(warped1) & (warped1 != 0)
    valid2 = np.isfinite(warped2) & (warped2 != 0)
    overlap = valid1 & valid2

    n_overlap = int(overlap.sum())
    if n_overlap == 0:
        return float("inf")

    r1 = warped1[overlap].astype(np.float64)
    r2 = warped2[overlap].astype(np.float64)

    # Normalise each to zero-mean / unit-std in the overlap
    mu1, s1 = r1.mean(), r1.std()
    mu2, s2 = r2.mean(), r2.std()
    if s1 > 1e-6:
        r1 = (r1 - mu1) / s1
    else:
        r1 = r1 - mu1
    if s2 > 1e-6:
        r2 = (r2 - mu2) / s2
    else:
        r2 = r2 - mu2

    return float(np.mean(np.abs(r1 - r2)))


def _make_imfuse(warped1: np.ndarray, warped2: np.ndarray) -> np.ndarray:
    """Magenta / green false-colour composite (same as cuda_coreg)."""
    valid1 = np.isfinite(warped1) & (warped1 != 0)
    valid2 = np.isfinite(warped2) & (warped2 != 0)
    overlap = valid1 & valid2

    img1 = np.where(valid1, warped1, 0.0).astype(np.float64)
    img2 = np.where(valid2, warped2, 0.0).astype(np.float64)

    target_mean = 128.0
    if overlap.any():
        for img, valid in [(img1, valid1), (img2, valid2)]:
            region = img[overlap]
            mu = region.mean()
            sigma = region.std()
            if sigma > 1e-6:
                img[valid] = (img[valid] - mu) / sigma * 40.0 + target_mean
            else:
                img[valid] = target_mean

    img1 = np.clip(img1, 0, 255) * valid1
    img2 = np.clip(img2, 0, 255) * valid2

    rgb = np.zeros((*warped1.shape, 3), dtype=np.uint8)
    rgb[..., 0] = img1.astype(np.uint8)
    rgb[..., 1] = img2.astype(np.uint8)
    rgb[..., 2] = img1.astype(np.uint8)
    return rgb


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="cuda_optimize",
        description=(
            "Use Optuna to optimise CuSift extraction and homography\n"
            "parameters, minimising the overlap pixel difference between\n"
            "the two warped images."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("image1", type=str, help="Path to the first (reference) image.")
    p.add_argument("image2", type=str, help="Path to the second (target) image.")
    p.add_argument("output_dir", type=str, help="Directory for results.")

    p.add_argument(
        "--lib", type=str, default=None, metavar="PATH",
        help="Explicit path to the cusift shared library.",
    )
    p.add_argument(
        "--n-trials", type=int, default=100,
        help="Number of Optuna trials (default: 100).",
    )
    p.add_argument(
        "--timeout", type=float, default=None,
        help="Maximum optimisation time in seconds (default: unlimited).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Sampler seed for reproducibility (default: 42).",
    )

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

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

    # -- Initialise CuSift once (reused across all trials) ----------------
    print("Initializing CuSift ...")
    sift = CuSift(dll_path=args.lib)
    print("[OK] CuSift initialized.\n")

    img1_str = str(img1_path)
    img2_str = str(img2_path)

    # -- Objective --------------------------------------------------------
    def objective(trial: optuna.Trial) -> float:
        # --- Extraction parameters ---
        thresh = trial.suggest_float("thresh", 2.0, 10.0)
        lowest_scale = trial.suggest_float("lowest_scale", 0.0, 8.0)
        edge_thresh = trial.suggest_float("edge_thresh", 5.0, 20.0)
        init_blur = trial.suggest_float("init_blur", 0.3, 2.0)
        num_octaves = trial.suggest_int("num_octaves", 3, 7)

        # --- Homography / RANSAC parameters ---
        num_loops = trial.suggest_int("num_loops", 1000, 50000, step=1000)
        max_ambiguity = trial.suggest_float("max_ambiguity", 0.5, 1.0)
        ransac_thresh = trial.suggest_float("ransac_thresh", 1.0, 15.0)
        improve_max_ambiguity = trial.suggest_float(
            "improve_max_ambiguity", 0.5, 1.0,
        )
        improve_thresh = trial.suggest_float("improve_thresh", 0.5, 10.0)
        improve_num_loops = trial.suggest_int("improve_num_loops", 20, 300, step=20)

        extract_opts = ExtractOptions(
            thresh=thresh,
            lowest_scale=lowest_scale,
            edge_thresh=edge_thresh,
            init_blur=init_blur,
            max_keypoints=65536,
            num_octaves=num_octaves,
        )
        homography_opts = HomographyOptions(
            num_loops=num_loops,
            min_score=0.0,
            max_ambiguity=max_ambiguity,
            thresh=ransac_thresh,
            improve_min_score=0.0,
            improve_max_ambiguity=improve_max_ambiguity,
            improve_thresh=improve_thresh,
            seed=args.seed,  # non-deterministic inside RANSAC for diversity
            improve_num_loops=improve_num_loops,
        )

        try:
            kp1, kp2, matches, H, n_inliers, w1, w2 = (
                sift.extract_and_match_and_find_homography_and_warp(
                    img1_str,
                    img2_str,
                    extract_options=extract_opts,
                    homography_options=homography_opts,
                )
            )
        except CuSiftError:
            # Library-level error → prune this trial
            raise optuna.TrialPruned("CuSift error during pipeline")

        if n_inliers < 4:
            kp1.free()
            kp2.free()
            raise optuna.TrialPruned(f"Too few inliers ({n_inliers})")

        mad = _overlap_mad(w1, w2)

        kp1.free()
        kp2.free()

        if not np.isfinite(mad):
            raise optuna.TrialPruned("No overlap between warped images")

        return mad

    # -- Run optimisation -------------------------------------------------
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name="cusift_coreg_optimize",
    )

    print(f"Starting Optuna optimisation ({args.n_trials} trials) ...\n")
    t0 = time.perf_counter()
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )
    t_opt = time.perf_counter() - t0

    # -- Report -----------------------------------------------------------
    best = study.best_trial
    print("\n" + "=" * 60)
    print(f"Optimisation complete  ({len(study.trials)} trials in {t_opt:.1f} s)")
    print(f"Best MAD:   {best.value:.6f}")
    print(f"Best trial: #{best.number}")
    print("Best parameters:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # -- Re-run with best params and save outputs -------------------------
    bp = best.params
    extract_opts = ExtractOptions(
        thresh=bp["thresh"],
        lowest_scale=bp["lowest_scale"],
        edge_thresh=bp["edge_thresh"],
        init_blur=bp["init_blur"],
        max_keypoints=32768,
        num_octaves=bp["num_octaves"],
    )
    homography_opts = HomographyOptions(
        num_loops=bp["num_loops"],
        min_score=0.0,
        max_ambiguity=bp["max_ambiguity"],
        thresh=bp["ransac_thresh"],
        improve_num_loops=5,
        improve_min_score=0.0,
        improve_max_ambiguity=bp["improve_max_ambiguity"],
        improve_thresh=bp["improve_thresh"],
        seed=0,
    )

    print("\nRe-running best configuration ...")
    kp1, kp2, matches, H, n_inliers, warped1, warped2 = (
        sift.extract_and_match_and_find_homography_and_warp(
            img1_str,
            img2_str,
            extract_options=extract_opts,
            homography_options=homography_opts,
        )
    )

    mad = _overlap_mad(warped1, warped2)
    print(f"  Keypoints: {len(kp1)} / {len(kp2)}")
    print(f"  Matches:   {len(matches)}")
    print(f"  Inliers:   {n_inliers}")
    print(f"  MAD:       {mad:.6f}")
    print("  Homography:")
    for r in range(3):
        vals = " ".join(f"{H[r, c]:12.6f}" for c in range(3))
        print(f"    [{vals}]")

    # Save warped images
    from PIL import Image as PILImage

    def _save_warped(pixels, path):
        arr = np.nan_to_num(np.clip(pixels, 1, 255), nan=0.0)
        PILImage.fromarray(arr.astype(np.uint8), mode="L").save(str(path))

    _save_warped(warped1, out_dir / "warped_image1.png")
    _save_warped(warped2, out_dir / "warped_image2.png")

    # Save composite
    composite = _make_imfuse(warped1, warped2)
    composite_path = out_dir / "composite_imfuse.png"
    PILImage.fromarray(composite, mode="RGB").save(str(composite_path))

    # Save homography
    np.savetxt(str(out_dir / "homography.txt"), H, fmt="%.10f")

    # Save study results
    results = {
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "optimisation_time_seconds": round(t_opt, 3),
        "best_trial": best.number,
        "best_mad": best.value,
        "best_params": best.params,
        "best_homography": H.tolist(),
        "best_num_keypoints_image1": len(kp1),
        "best_num_keypoints_image2": len(kp2),
        "best_num_matches": len(matches),
        "best_num_inliers": n_inliers,
        "image1": str(img1_path.resolve()),
        "image2": str(img2_path.resolve()),
        "all_trials": [
            {
                "number": t.number,
                "state": t.state.name,
                "value": t.value,
                "params": t.params,
            }
            for t in study.trials
        ],
    }
    results_path = out_dir / "optimization_results.json"
    results_path.write_text(json.dumps(results, indent=2))

    # Save match visualization
    if matches:
        sift.draw_matches(
            img1_str, img2_str, matches,
            str(out_dir / "matches_best.png"),
        )

    kp1.free()
    kp2.free()

    print(f"\nAll outputs saved to: {out_dir.resolve()}")
    print(
        f"\nTo run cuda_coreg.py with these parameters:\n"
        f"  python cuda_coreg.py {args.image1} {args.image2} <output_dir>"
        f" --thresh {bp['thresh']:.4f}"
        f" --lowest-scale {bp['lowest_scale']:.4f}"
        f" --edge-thresh {bp['edge_thresh']:.4f}"
        f" --init-blur {bp['init_blur']:.4f}"
        f" --num-octaves {bp['num_octaves']}"
        f" --num-loops {bp['num_loops']}"
        f" --max-ambiguity {bp['max_ambiguity']:.4f}"
        f" --ransac-thresh {bp['ransac_thresh']:.4f}"
        f" --improve-max-ambiguity {bp['improve_max_ambiguity']:.4f}"
        f" --improve-thresh {bp['improve_thresh']:.4f}"
    )


if __name__ == "__main__":
    main()
