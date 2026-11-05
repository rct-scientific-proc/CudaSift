from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from cusift import (
    CuSift,
    CuSiftError,
    GOAL_MAX_INLIERS,
    GOAL_MIN_IDENTITY_DISTANCE,
    HomographyOptions,
)


def _load_grayscale_float32(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    return np.ascontiguousarray(arr)


def _save_image(path: Path, arr: np.ndarray) -> None:
    clean = np.nan_to_num(arr, nan=0.0)
    clean = np.clip(clean, 0, 255).astype(np.uint8)
    Image.fromarray(clean, mode="L").save(path)


def _serialize_keypoint(kp: object) -> dict:
    return {
        "x": float(kp.x),
        "y": float(kp.y),
        "scale": float(kp.scale),
        "sharpness": float(kp.sharpness),
        "edgeness": float(kp.edgeness),
        "orientation": float(kp.orientation),
        "score": float(kp.score),
        "ambiguity": float(kp.ambiguity),
        "match": int(kp.match),
        "match_x": float(kp.match_x),
        "match_y": float(kp.match_y),
        "match_error": float(kp.match_error),
        "subsampling": float(kp.subsampling),
    }


def _save_keypoints_and_descriptors(outdir: Path, keypoints: list, suffix: str) -> None:
    keypoints_path = outdir / f"keypoints_{suffix}.json"
    descriptors_path = outdir / f"descriptors_{suffix}.npy"

    payload = [_serialize_keypoint(kp) for kp in keypoints]
    with keypoints_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if keypoints:
        descriptors = np.stack([np.asarray(kp.descriptor, dtype=np.float32) for kp in keypoints], axis=0)
    else:
        descriptors = np.empty((0, 128), dtype=np.float32)
    np.save(descriptors_path, descriptors)


def _save_matches(outdir: Path, matches: list) -> None:
    matches_path = outdir / "matches.json"
    payload = [
        {
            "query_index": int(m.query_index),
            "match_index": int(m.match_index),
            "x1": float(m.x1),
            "y1": float(m.y1),
            "x2": float(m.x2),
            "y2": float(m.y2),
            "error": float(m.error),
            "score": float(m.score),
            "ambiguity": float(m.ambiguity),
        }
        for m in matches
    ]
    with matches_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _save_homography(outdir: Path, homography: np.ndarray) -> None:
    np.save(outdir / "homography.npy", np.asarray(homography, dtype=np.float32))


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _goal_from_cli(goal: str) -> int:
    if goal == "max_inliers":
        return GOAL_MAX_INLIERS
    if goal == "min_eye_diff":
        return GOAL_MIN_IDENTITY_DISTANCE
    raise ValueError(f"Unknown goal: {goal}")


def _print_stage(stage: str) -> None:
    print(f"\n=== {stage} ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CuSift Python API smoke/integration tests and save outputs."
    )
    parser.add_argument("i1", type=Path, help="Path to the first image")
    parser.add_argument("i2", type=Path, help="Path to the second image")
    parser.add_argument("outdir", type=Path, help="Output directory")
    parser.add_argument(
        "--dll",
        type=Path,
        default=None,
        help="Optional path to cusift shared library (cusift.dll / libcusift.so)",
    )
    parser.add_argument(
        "--num-homography-attempts",
        type=int,
        default=5,
        help="Number of attempts for multi-homography APIs",
    )
    parser.add_argument(
        "--homography-goal",
        choices=["max_inliers", "min_eye_diff"],
        default="max_inliers",
        help="Selection criterion for multi-homography APIs",
    )
    parser.add_argument(
        "--cpu-warp",
        action="store_true",
        help="Use CPU implementation for warp instead of GPU",
    )
    args = parser.parse_args()

    if not args.i1.exists() or not args.i2.exists():
        raise FileNotFoundError("Input image path does not exist")

    out_root = _ensure_dir(args.outdir)
    goal = _goal_from_cli(args.homography_goal)

    print("Loading input images as grayscale float32 arrays...")
    img1 = _load_grayscale_float32(args.i1)
    img2 = _load_grayscale_float32(args.i2)
    print(f"i1 shape: {img1.shape}, dtype: {img1.dtype}")
    print(f"i2 shape: {img2.shape}, dtype: {img2.dtype}")

    sift = CuSift(dll_path=args.dll)

    try:
        _print_stage("extract + match + estimate_homography + warp")
        op1 = _ensure_dir(out_root / "extract_match_find_homography_warp")
        kp1 = sift.extract(img1)
        kp2 = sift.extract(img2)
        try:
            _save_keypoints_and_descriptors(op1, kp1, "i1")
            _save_keypoints_and_descriptors(op1, kp2, "i2")

            matches = sift.match(kp1, kp2)
            _save_matches(op1, matches)

            H, n_inliers = sift.estimate_homography(kp1)
            _save_homography(op1, H)
            print(f"Homography:\n{H}\nInliers: {n_inliers}")

            warped1, warped2 = sift.warp(img1, img2, H, use_gpu=not args.cpu_warp)
            _save_image(op1 / "warped_i1.png", warped1)
            _save_image(op1 / "warped_i2.png", warped2)

            print(f"kp1={len(kp1)}, kp2={len(kp2)}, matches={len(matches)}, inliers={n_inliers}")
        finally:
            kp1.free()
            kp2.free()

        _print_stage("extract_and_match")
        op2 = _ensure_dir(out_root / "extract_and_match")
        kp1, kp2, matches = sift.extract_and_match(img1, img2)
        try:
            _save_keypoints_and_descriptors(op2, kp1, "i1")
            _save_keypoints_and_descriptors(op2, kp2, "i2")
            _save_matches(op2, matches)
            print(f"kp1={len(kp1)}, kp2={len(kp2)}, matches={len(matches)}")
        finally:
            kp1.free()
            kp2.free()

        _print_stage("align")
        op3 = _ensure_dir(out_root / "align")
        kp1, kp2, matches, H, n_inliers = sift.align(img1, img2)
        try:
            _save_keypoints_and_descriptors(op3, kp1, "i1")
            _save_keypoints_and_descriptors(op3, kp2, "i2")
            _save_matches(op3, matches)
            _save_homography(op3, H)
            print(f"kp1={len(kp1)}, kp2={len(kp2)}, matches={len(matches)}, inliers={n_inliers}")
        finally:
            kp1.free()
            kp2.free()

        _print_stage("align (several attempts)")
        op4 = _ensure_dir(out_root / "align_attempts")
        kp1, kp2, matches, H, n_inliers = sift.align(
            img1,
            img2,
            homography_options=HomographyOptions(attempts=args.num_homography_attempts, goal=goal),
        )
        print(f"Homography:\n{H}\nInliers: {n_inliers}")
        try:
            _save_keypoints_and_descriptors(op4, kp1, "i1")
            _save_keypoints_and_descriptors(op4, kp2, "i2")
            _save_matches(op4, matches)
            _save_homography(op4, H)
            print(f"kp1={len(kp1)}, kp2={len(kp2)}, matches={len(matches)}, inliers={n_inliers}")
        finally:
            kp1.free()
            kp2.free()

        _print_stage("align_and_warp")
        op5 = _ensure_dir(out_root / "align_and_warp")
        kp1, kp2, matches, H, n_inliers, warped1, warped2 = (
            sift.align_and_warp(img1, img2)
        )
        print(f"Homography:\n{H}\nInliers: {n_inliers}")
        try:
            _save_keypoints_and_descriptors(op5, kp1, "i1")
            _save_keypoints_and_descriptors(op5, kp2, "i2")
            _save_matches(op5, matches)
            _save_homography(op5, H)
            _save_image(op5 / "warped_i1.png", warped1)
            _save_image(op5 / "warped_i2.png", warped2)
            print(f"kp1={len(kp1)}, kp2={len(kp2)}, matches={len(matches)}, inliers={n_inliers}")
        finally:
            kp1.free()
            kp2.free()

        _print_stage("align_and_warp (several attempts)")
        op6 = _ensure_dir(out_root / "align_and_warp_attempts")
        kp1, kp2, matches, H, n_inliers, warped1, warped2 = (
            sift.align_and_warp(
                img1,
                img2,
                homography_options=HomographyOptions(attempts=args.num_homography_attempts, goal=goal),
            )
        )
        print(f"Homography:\n{H}\nInliers: {n_inliers}")
        try:
            _save_keypoints_and_descriptors(op6, kp1, "i1")
            _save_keypoints_and_descriptors(op6, kp2, "i2")
            _save_matches(op6, matches)
            _save_homography(op6, H)
            _save_image(op6 / "warped_i1.png", warped1)
            _save_image(op6 / "warped_i2.png", warped2)
            print(f"kp1={len(kp1)}, kp2={len(kp2)}, matches={len(matches)}, inliers={n_inliers}")
        finally:
            kp1.free()
            kp2.free()

    except CuSiftError as exc:
        location = f" ({exc.filename}:{exc.line})" if exc.filename else ""
        raise RuntimeError(f"CuSiftError{location}: {exc}") from exc

    print(f"\nDone. Results saved under: {out_root}")


if __name__ == "__main__":
    main()


