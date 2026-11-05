"""
Low-level ctypes bindings for the cusift C shared library.

This module mirrors all structs and function signatures from ``cusift.h``.
Users should prefer the high-level API in :mod:`cusift.cusift`.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import (CFUNCTYPE, POINTER, Structure, c_bool, c_char, c_char_p, c_float, c_int,
                    c_longlong, c_size_t, c_uint, c_ulonglong, c_void_p)
from pathlib import Path

# -- ctypes struct mirrors of cusift.h ----------------------------------------


class cusift_keypoint(Structure):
    """Mirror of the C ``cusift_keypoint`` struct."""

    _fields_ = [
        ("x", c_float),
        ("y", c_float),
        ("scale", c_float),
        ("sharpness", c_float),
        ("edgeness", c_float),
        ("orientation", c_float),
        ("score", c_float),
        ("ambiguity", c_float),
        ("match", c_int),
        ("match_x", c_float),
        ("match_y", c_float),
        ("match_error", c_float),
        ("subsampling", c_float),
        ("reserved", c_float * 3),
        ("descriptor", c_float * 128),
    ]


class cusift_keypoints(Structure):
    """Mirror of the C ``cusift_keypoints`` struct."""

    _fields_ = [
        ("count", c_int),
        ("capacity", c_int),
        ("host", POINTER(cusift_keypoint)),
        ("device", POINTER(cusift_keypoint)),
    ]


class cusift_image(Structure):
    """Mirror of the C ``cusift_image`` struct."""

    _fields_ = [
        ("pixels", POINTER(c_float)),
        ("width", c_int),
        ("height", c_int),
    ]


class cusift_canvas(Structure):
    """Mirror of the C ``cusift_canvas`` struct."""

    _fields_ = [
        ("origin_u", c_int),
        ("origin_v", c_int),
        ("canvas_width", c_int),
        ("canvas_height", c_int),
        ("corners1", (c_float * 2) * 4),
        ("corners2", (c_float * 2) * 4),
    ]


class cusift_error_record(Structure):
    """Mirror of the C ``cusift_error_record`` struct."""

    _fields_ = [
        ("severity", c_int),
        ("function", c_char_p),
        ("file", c_char_p),
        ("line", c_int),
        ("cuda_error", c_int),
        ("cuda_error_name", c_char_p),
        ("message", c_char_p),
        ("thread_id", c_ulonglong),
        ("timestamp_ms", c_longlong),
    ]


# void (*)(const cusift_error_record *record, void *user_data)
cusift_error_callback = CFUNCTYPE(None, POINTER(cusift_error_record), c_void_p)


class cusift_extract_options(Structure):
    """Mirror of the C ``cusift_extract_options`` struct."""

    _fields_ = [
        ("contrast_threshold", c_float),
        ("lowest_scale", c_float),
        ("highest_scale", c_float),
        ("edge_threshold", c_float),
        ("initial_blur", c_float),
        ("max_keypoints", c_int),
        ("num_octaves", c_int),
        ("scale_suppression_radius", c_float),
    ]


class cusift_homography_options(Structure):
    """Mirror of the C ``cusift_homography_options`` struct."""

    _fields_ = [
        ("ransac_iterations", c_int),
        ("min_score", c_float),
        ("max_ambiguity", c_float),
        ("inlier_threshold", c_float),
        ("refine_iterations", c_int),
        ("refine_min_score", c_float),
        ("refine_max_ambiguity", c_float),
        ("refine_inlier_threshold", c_float),
        ("seed", c_uint),
        ("model", c_int),
        ("attempts", c_int),
        ("goal", c_int),
    ]


# -- Default DLL search paths ------------------------------------------------

_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_DLL_CANDIDATES = [
    # Typical CMake build output next to the Python package
    _PKG_DIR.parent.parent / "build" / "Release" / "cusift.dll",
    _PKG_DIR.parent.parent / "build" / "Debug" / "cusift.dll",
    _PKG_DIR.parent.parent / "build" / "cusift.dll",
    _PKG_DIR.parent.parent / "install" / "bin" / "cusift.dll",
    # Linux / macOS
    _PKG_DIR.parent.parent / "build" / "libcusift.so",
    _PKG_DIR.parent.parent / "build" / "Release" / "libcusift.so",
    _PKG_DIR.parent.parent / "build" / "libcusift.dylib",
]


def _find_default_dll() -> Path:
    """Try common locations next to the source tree."""
    for p in _DEFAULT_DLL_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not auto-locate the cusift shared library.  "
        "Build with -DCUSIFT_BUILD_SHARED=ON or pass the path explicitly."
    )


def load_library(dll_path: str | Path | None = None) -> ctypes.CDLL:
    """Load the cusift shared library and declare all function signatures.

    Parameters
    ----------
    dll_path : str | Path | None
        Explicit path to ``cusift.dll`` / ``libcusift.so``.
        When *None*, the function searches common build directories.

    Returns
    -------
    ctypes.CDLL
        The loaded library handle with all argtypes/restypes configured.
    """
    if dll_path is None:
        dll_path = _find_default_dll()
    dll_path = Path(dll_path).resolve()

    # Windows needs CUDA runtime DLLs on the search path
    if sys.platform == "win32":
        os.add_dll_directory(str(dll_path.parent))

    lib = ctypes.CDLL(str(dll_path))

    # -- Bind every exported function -------------------------------------

    # Version and errors
    lib.cusift_version.restype = None
    lib.cusift_version.argtypes = [POINTER(c_int), POINTER(c_int), POINTER(c_int)]
    lib.cusift_last_error.restype = c_int
    lib.cusift_last_error.argtypes = [POINTER(cusift_error_record)]

    # Option defaults
    lib.cusift_default_extract_options.restype = None
    lib.cusift_default_extract_options.argtypes = [POINTER(cusift_extract_options)]
    lib.cusift_default_homography_options.restype = None
    lib.cusift_default_homography_options.argtypes = [POINTER(cusift_homography_options)]

    # Initialisation
    lib.cusift_init.restype = None
    lib.cusift_init.argtypes = []

    # Feature extraction
    lib.cusift_extract.restype = None
    lib.cusift_extract.argtypes = [
        POINTER(cusift_image),
        POINTER(cusift_keypoints),
        POINTER(cusift_extract_options),
    ]

    # Matching
    lib.cusift_match.restype = None
    lib.cusift_match.argtypes = [POINTER(cusift_keypoints), POINTER(cusift_keypoints)]

    # Homography
    lib.cusift_estimate_homography.restype = None
    lib.cusift_estimate_homography.argtypes = [
        POINTER(cusift_keypoints),
        POINTER(c_float),
        POINTER(c_int),
        POINTER(cusift_homography_options),
    ]

    # Image warping
    lib.cusift_warp.restype = None
    lib.cusift_warp.argtypes = [
        POINTER(cusift_image),  # image1
        POINTER(cusift_image),  # image2
        POINTER(c_float),  # homography
        POINTER(cusift_image),  # warped_image1  (out)
        POINTER(cusift_image),  # warped_image2  (out)
        c_bool,            # use_gpu
    ]

    # Canvas geometry (no warp)
    lib.cusift_compute_canvas.restype = None
    lib.cusift_compute_canvas.argtypes = [
        POINTER(c_float),          # homography
        c_int, c_int,              # image_width1, image_height1
        c_int, c_int,              # image_width2, image_height2
        POINTER(cusift_canvas),   # geometry (out)
    ]

    # Process-local device reset after a sticky CUDA error
    lib.cusift_device_reset.restype = None
    lib.cusift_device_reset.argtypes = []

    # Error log file and callback
    lib.cusift_set_error_log.restype = None
    lib.cusift_set_error_log.argtypes = [c_char_p, c_size_t]
    lib.cusift_set_error_callback.restype = None
    lib.cusift_set_error_callback.argtypes = [cusift_error_callback, c_void_p]
    lib.cusift_error_log_dropped.restype = c_ulonglong
    lib.cusift_error_log_dropped.argtypes = []

    # Cleanup
    lib.cusift_keypoints_free.restype = None
    lib.cusift_keypoints_free.argtypes = [POINTER(cusift_keypoints)]

    lib.cusift_image_free.restype = None
    lib.cusift_image_free.argtypes = [POINTER(cusift_image)]

    # Save / serialise
    lib.cusift_keypoints_save_json.restype = None
    lib.cusift_keypoints_save_json.argtypes = [ctypes.c_char_p, POINTER(cusift_keypoints)]

    # -- Convenience combo functions --------------------------------------

    lib.cusift_extract_and_match.restype = None
    lib.cusift_extract_and_match.argtypes = [
        POINTER(cusift_image),
        POINTER(cusift_image),
        POINTER(cusift_keypoints),
        POINTER(cusift_keypoints),
        POINTER(cusift_extract_options),
    ]

    lib.cusift_align.restype = None
    lib.cusift_align.argtypes = [
        POINTER(cusift_image),
        POINTER(cusift_image),
        POINTER(cusift_keypoints),
        POINTER(cusift_keypoints),
        POINTER(cusift_extract_options),
        POINTER(cusift_homography_options),
        POINTER(c_float),   # homography (out)
        POINTER(c_int),     # inliers (out)
    ]

    lib.cusift_align_and_warp.restype = None
    lib.cusift_align_and_warp.argtypes = [
        POINTER(cusift_image),
        POINTER(cusift_image),
        POINTER(cusift_keypoints),
        POINTER(cusift_keypoints),
        POINTER(cusift_extract_options),
        POINTER(cusift_homography_options),
        POINTER(c_float),        # homography (out)
        POINTER(c_int),          # inliers (out)
        POINTER(cusift_image),   # warped1 (out)
        POINTER(cusift_image),   # warped2 (out)
    ]


    return lib
