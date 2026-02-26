"""
Low-level ctypes bindings for the cusift C shared library.

This module mirrors all structs and function signatures from ``cusift.h``.
Users should prefer the high-level API in :mod:`cusift.cusift`.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import POINTER, Structure, c_bool, c_char, c_float, c_int, c_uint
from pathlib import Path

# -- ctypes struct mirrors of cusift.h ----------------------------------------


class SiftPoint(Structure):
    """Mirror of the C ``SiftPoint`` struct."""

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
    """Mirror of the C ``SiftData`` struct."""

    _fields_ = [
        ("numPts", c_int),
        ("maxPts", c_int),
        ("h_data", POINTER(SiftPoint)),
        ("d_data", POINTER(SiftPoint)),
    ]


class Image_t(Structure):
    """Mirror of the C ``Image_t`` struct."""

    _fields_ = [
        ("host_img_", POINTER(c_float)),
        ("width_", c_int),
        ("height_", c_int),
    ]


class ExtractSiftOptions_t(Structure):
    """Mirror of the C ``ExtractSiftOptions_t`` struct."""

    _fields_ = [
        ("thresh_", c_float),
        ("lowest_scale_", c_float),
        ("edge_thresh_", c_float),
        ("init_blur_", c_float),
        ("max_keypoints_", c_int),
        ("num_octaves_", c_int),
    ]


class FindHomographyOptions_t(Structure):
    """Mirror of the C ``FindHomographyOptions_t`` struct."""

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

    # Error helpers
    lib.CusiftGetLastErrorString.restype = None
    lib.CusiftGetLastErrorString.argtypes = [
        POINTER(c_int),
        c_char * 256,
        c_char * 256,
    ]

    lib.CusiftHadError.restype = c_int
    lib.CusiftHadError.argtypes = []

    # Initialisation
    lib.InitializeCudaSift.restype = None
    lib.InitializeCudaSift.argtypes = []

    # Feature extraction
    lib.ExtractSiftFromImage.restype = None
    lib.ExtractSiftFromImage.argtypes = [
        POINTER(Image_t),
        POINTER(SiftData),
        POINTER(ExtractSiftOptions_t),
    ]

    # Matching
    lib.MatchSiftData.restype = None
    lib.MatchSiftData.argtypes = [POINTER(SiftData), POINTER(SiftData)]

    # Homography
    lib.FindHomography.restype = None
    lib.FindHomography.argtypes = [
        POINTER(SiftData),
        POINTER(c_float),
        POINTER(c_int),
        POINTER(FindHomographyOptions_t),
    ]

    # Image warping
    lib.WarpImages.restype = None
    lib.WarpImages.argtypes = [
        POINTER(Image_t),  # image1
        POINTER(Image_t),  # image2
        POINTER(c_float),  # homography
        POINTER(Image_t),  # warped_image1  (out)
        POINTER(Image_t),  # warped_image2  (out)
        c_bool,            # useGPU
    ]

    # Cleanup
    lib.DeleteSiftData.restype = None
    lib.DeleteSiftData.argtypes = [POINTER(SiftData)]

    lib.FreeImage.restype = None
    lib.FreeImage.argtypes = [POINTER(Image_t)]

    # Save / serialise
    lib.SaveSiftData.restype = None
    lib.SaveSiftData.argtypes = [ctypes.c_char_p, POINTER(SiftData)]

    # -- Convenience combo functions --------------------------------------

    lib.ExtractAndMatchSift.restype = None
    lib.ExtractAndMatchSift.argtypes = [
        POINTER(Image_t),
        POINTER(Image_t),
        POINTER(SiftData),
        POINTER(SiftData),
        POINTER(ExtractSiftOptions_t),
    ]

    lib.ExtractAndMatchAndFindHomography.restype = None
    lib.ExtractAndMatchAndFindHomography.argtypes = [
        POINTER(Image_t),
        POINTER(Image_t),
        POINTER(SiftData),
        POINTER(SiftData),
        POINTER(c_float),
        POINTER(c_int),
        POINTER(ExtractSiftOptions_t),
        POINTER(FindHomographyOptions_t),
    ]

    lib.ExtractAndMatchAndFindHomographyAndWarp.restype = None
    lib.ExtractAndMatchAndFindHomographyAndWarp.argtypes = [
        POINTER(Image_t),
        POINTER(Image_t),
        POINTER(SiftData),
        POINTER(SiftData),
        POINTER(c_float),
        POINTER(c_int),
        POINTER(ExtractSiftOptions_t),
        POINTER(FindHomographyOptions_t),
        POINTER(Image_t),
        POINTER(Image_t),
    ]

    return lib
