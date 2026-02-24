"""
cusift_ctypes.py – Load the cusift shared library via ctypes and run the
full SIFT extraction / matching / homography pipeline on two images.

Usage:
    python cusift_ctypes.py <img1> <img2> [--dll <path/to/cusift.dll>]

Requires: Pillow (pip install Pillow) for image loading.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_char_p,
    c_float,
    c_int,
    c_void_p,
    c_uint,
)
from pathlib import Path

import numpy as np

# ── ctypes struct mirrors of cusift.h ────────────────────


class SiftPoint(Structure):
    _fields_ = [
        ("xpos", c_float),
        ("ypos", c_float),
        ("scale", c_float),
        ("sharpness", c_float),
        ("edgeness", c_float),
        ("orientation", c_float),
        ("score", c_float),
        ("ambiguity", c_float),
        ("match", c_int),
        ("match_xpos", c_float),
        ("match_ypos", c_float),
        ("match_error", c_float),
        ("subsampling", c_float),
        ("empty", c_float * 3),
        ("data", c_float * 128),
    ]


class SiftData(Structure):
    _fields_ = [
        ("numPts", c_int),
        ("maxPts", c_int),
        ("h_data", POINTER(SiftPoint)),
        ("d_data", POINTER(SiftPoint)),
    ]


class Image_t(Structure):
    _fields_ = [
        ("host_img_", POINTER(c_float)),
        ("width_", c_int),
        ("height_", c_int),
    ]


class ExtractSiftOptions_t(Structure):
    _fields_ = [
        ("thresh_", c_float),
        ("lowest_scale_", c_float),
        ("edge_thresh_", c_float),
        ("init_blur_", c_float),
        ("max_keypoints_", c_int),
        ("num_octaves_", c_int),
    ]


class FindHomographyOptions_t(Structure):
    _fields_ = [
        ("num_loops_", c_int),
        ("min_score_", c_float),
        ("max_ambiguity_", c_float),
        ("thresh_", c_float),
        ("improve_num_loops_", c_int),
        ("improve_min_score_", c_float),
        ("improve_max_ambiguity_", c_float),
        ("improve_thresh_", c_float),
        ("seed_", c_uint),
    ]


# ── Helper: load & bind the DLL ─────────────────────────


def load_cusift(dll_path: str | Path) -> ctypes.CDLL:
    """Load cusift.dll and declare function signatures."""
    lib = ctypes.CDLL(str(dll_path))

    lib.InitializeCudaSift.restype = None
    lib.InitializeCudaSift.argtypes = []

    lib.ExtractSiftFromImage.restype = None
    lib.ExtractSiftFromImage.argtypes = [
        POINTER(Image_t),
        POINTER(SiftData),
        POINTER(ExtractSiftOptions_t),
    ]

    lib.MatchSiftData.restype = None
    lib.MatchSiftData.argtypes = [POINTER(SiftData), POINTER(SiftData)]

    lib.FindHomography.restype = None
    lib.FindHomography.argtypes = [
        POINTER(SiftData),
        POINTER(c_float),
        POINTER(c_int),
        POINTER(FindHomographyOptions_t),
    ]

    lib.DeleteSiftData.restype = None
    lib.DeleteSiftData.argtypes = [POINTER(SiftData)]

    lib.SaveSiftData.restype = None
    lib.SaveSiftData.argtypes = [c_char_p, POINTER(SiftData)]

    return lib


# ── Helper: load an image as grayscale float32 array ─────


def load_image_grayscale(path: str | Path, scale: int = 1) -> tuple[np.ndarray, int, int]:
    """Return (float32 row-major array, width, height). scale>1 repeats pixels."""
    from PIL import Image

    img = Image.open(path).convert("L")
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    w, h = img.size
    arr = np.asarray(img, dtype=np.float32)  # H×W, row-major, [0..255]
    return np.ascontiguousarray(arr), w, h


def make_image_t(pixels: np.ndarray, width: int, height: int) -> Image_t:
    """Wrap a numpy float32 array in an Image_t struct."""
    img = Image_t()
    img.host_img_ = pixels.ctypes.data_as(POINTER(c_float))
    img.width_ = width
    img.height_ = height
    return img


# ── Default DLL search ───────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_DLL_CANDIDATES = [
    _SCRIPT_DIR.parent / "build" / "Release" / "cusift.dll",
    _SCRIPT_DIR.parent / "build" / "Debug" / "cusift.dll",
    _SCRIPT_DIR.parent / "build" / "cusift.dll",
    # Linux / macOS
    _SCRIPT_DIR.parent / "build" / "libcusift.so",
    _SCRIPT_DIR.parent / "build" / "Release" / "libcusift.so",
]


def find_default_dll() -> Path:
    for p in _DEFAULT_DLL_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find cusift shared library. "
        "Build with -DCUSIFT_BUILD_SHARED=ON or pass --dll <path>."
    )


# ── Main ─────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="CuSIFT ctypes demo")
    parser.add_argument("img1", help="Path to first image")
    parser.add_argument("img2", help="Path to second image")
    parser.add_argument("--dll", default=None, help="Path to cusift shared library")
    parser.add_argument("--save1", default=None, help="JSON output for image 1 keypoints")
    parser.add_argument("--save2", default=None, help="JSON output for image 2 keypoints")
    parser.add_argument("--scale", type=int, default=1, help="Upscale images by this factor (default: 1)")
    args = parser.parse_args()

    dll_path = Path(args.dll).resolve() if args.dll else find_default_dll()
    print(f"Loading library: {dll_path}")

    # Add DLL directory to search path (Windows needs CUDA runtime DLLs)
    if sys.platform == "win32":
        os.add_dll_directory(str(dll_path.parent))

    lib = load_cusift(dll_path)

    # ── Initialize ───────────────────────────────────────
    lib.InitializeCudaSift()

    # ── Load images ──────────────────────────────────────
    pixels1, w1, h1 = load_image_grayscale(args.img1, scale=args.scale)
    pixels2, w2, h2 = load_image_grayscale(args.img2, scale=args.scale)
    print(f"Image 1: {args.img1} ({w1} x {h1})")
    print(f"Image 2: {args.img2} ({w2} x {h2})")

    img1 = make_image_t(pixels1, w1, h1)
    img2 = make_image_t(pixels2, w2, h2)

    # ── Extract SIFT ─────────────────────────────────────
    opts = ExtractSiftOptions_t(
        thresh_=3.0,
        lowest_scale_=0.0,
        edge_thresh_=10.0,
        init_blur_=1.0,
        max_keypoints_=32768,
        num_octaves_=5,
    )

    sift1 = SiftData()
    sift2 = SiftData()

    print("Extracting SIFT features from image 1...")
    lib.ExtractSiftFromImage(byref(img1), byref(sift1), byref(opts))
    print(f"  Found {sift1.numPts} keypoints")

    print("Extracting SIFT features from image 2...")
    lib.ExtractSiftFromImage(byref(img2), byref(sift2), byref(opts))
    print(f"  Found {sift2.numPts} keypoints")

    # ── Match ────────────────────────────────────────────
    print("Matching SIFT features...")
    lib.MatchSiftData(byref(sift1), byref(sift2))

    # ── Homography ───────────────────────────────────────
    homo_opts = FindHomographyOptions_t(
        num_loops_=10000,
        min_score_=0.0,
        max_ambiguity_=0.80,
        thresh_=5.0,
        improve_num_loops_=5,
        improve_min_score_=0.0,
        improve_max_ambiguity_=0.80,
        improve_thresh_=3.0,
        seed_=42,
    )

    homography = (c_float * 9)()
    num_matches = c_int(0)

    print("Finding homography...")
    lib.FindHomography(byref(sift1), homography, byref(num_matches), byref(homo_opts))
    print(f"  Matches (inliers): {num_matches.value}")

    print("Homography:")
    for r in range(3):
        vals = [f"{homography[r * 3 + c]:12.6f}" for c in range(3)]
        print(f"  [{' '.join(vals)}]")

    # ── Save (optional) ─────────────────────────────────
    if args.save1:
        lib.SaveSiftData(args.save1.encode(), byref(sift1))
        print(f"Saved image 1 keypoints to {args.save1}")
    if args.save2:
        lib.SaveSiftData(args.save2.encode(), byref(sift2))
        print(f"Saved image 2 keypoints to {args.save2}")

    # ── Cleanup ──────────────────────────────────────────
    lib.DeleteSiftData(byref(sift1))
    lib.DeleteSiftData(byref(sift2))
    print("Done.")


if __name__ == "__main__":
    main()
