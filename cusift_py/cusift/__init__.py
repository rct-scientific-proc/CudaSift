"""
cusift – A Python interface to the CuSIFT GPU-accelerated SIFT library.

Example
-------
>>> from cusift import CuSift
>>> sift = CuSift()                           # uses default DLL search
>>> kp1 = sift.extract("photo1.png")
>>> kp2 = sift.extract("photo2.png")
>>> matches = sift.match(kp1, kp2)
>>> H, n_inliers = sift.estimate_homography(kp1)
"""

from cusift.cusift import (
    CuSift,
    CuSiftError,
    ErrorRecord,
    ExtractOptions,
    GOAL_MAX_INLIERS,
    GOAL_MIN_IDENTITY_DISTANCE,
    HomographyOptions,
    Keypoint,
    Keypoints,
    MatchResult,
    MODEL_HOMOGRAPHY,
    MODEL_SIMILARITY,
    Canvas,
)

__all__ = [
    "CuSift",
    "CuSiftError",
    "ErrorRecord",
    "ExtractOptions",
    "GOAL_MAX_INLIERS",
    "GOAL_MIN_IDENTITY_DISTANCE",
    "HomographyOptions",
    "Keypoint",
    "Keypoints",
    "MatchResult",
    "MODEL_HOMOGRAPHY",
    "MODEL_SIMILARITY",
    "Canvas",
]
