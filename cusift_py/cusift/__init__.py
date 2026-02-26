"""
cusift – A Python interface to the CuSIFT GPU-accelerated SIFT library.

Example
-------
>>> from cusift import CuSift
>>> sift = CuSift()                           # uses default DLL search
>>> kp1 = sift.extract("photo1.png")
>>> kp2 = sift.extract("photo2.png")
>>> matches = sift.match(kp1, kp2)
>>> H, n_inliers = sift.find_homography(kp1)
"""

from cusift.cusift import (
    CuSift,
    CuSiftError,
    ExtractOptions,
    HomographyOptions,
    Keypoint,
    KeypointList,
    MatchResult,
)

__all__ = [
    "CuSift",
    "CuSiftError",
    "ExtractOptions",
    "HomographyOptions",
    "Keypoint",
    "KeypointList",
    "MatchResult",
]
