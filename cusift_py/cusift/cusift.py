"""
High-level Pythonic interface to the CuSIFT GPU-accelerated SIFT library.

All public symbols are re-exported from the package ``__init__.py``.
"""

from __future__ import annotations

import ctypes
from ctypes import POINTER, byref, c_bool, c_float, c_int
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from cusift._bindings import (
    ExtractSiftOptions_t,
    FindHomographyOptions_t,
    Image_t,
    SiftData,
    SiftPoint,
    load_library,
)


# -- Exceptions ---------------------------------------------------------------


class CuSiftError(Exception):
    """Raised when the C library reports an error."""

    def __init__(self, message: str, filename: str = "", line: int = 0):
        self.filename = filename
        self.line = line
        super().__init__(message)


def _check_error(lib: ctypes.CDLL) -> None:
    """Query the library error flag; raise :class:`CuSiftError` if set."""
    if lib.CusiftHadError():
        line = c_int(0)
        fname = (ctypes.c_char * 256)()
        msg = (ctypes.c_char * 256)()
        lib.CusiftGetLastErrorString(byref(line), fname, msg)
        raise CuSiftError(
            msg.value.decode("utf-8", errors="replace"),
            filename=fname.value.decode("utf-8", errors="replace"),
            line=line.value,
        )


# -- Data classes -------------------------------------------------------------


@dataclass
class Keypoint:
    """A single SIFT keypoint with its 128-d descriptor."""

    x: float
    y: float
    scale: float
    sharpness: float
    edgeness: float
    orientation: float
    score: float
    ambiguity: float
    match: int
    match_x: float
    match_y: float
    match_error: float
    subsampling: float
    descriptor: np.ndarray = field(repr=False)  # shape (128,)

    @classmethod
    def _from_sift_point(cls, pt: SiftPoint) -> "Keypoint":
        desc = np.ctypeslib.as_array(pt.data, shape=(128,)).copy()
        return cls(
            x=pt.xpos,
            y=pt.ypos,
            scale=pt.scale,
            sharpness=pt.sharpness,
            edgeness=pt.edgeness,
            orientation=pt.orientation,
            score=pt.score,
            ambiguity=pt.ambiguity,
            match=pt.match,
            match_x=pt.match_xpos,
            match_y=pt.match_ypos,
            match_error=pt.match_error,
            subsampling=pt.subsampling,
            descriptor=desc,
        )


@dataclass
class MatchResult:
    """A single matched keypoint pair.

    After calling :meth:`CuSift.match`, each entry describes a
    correspondence found between the two sets of keypoints.
    """

    query_index: int
    """Index into the *first* keypoint list (query)."""

    match_index: int
    """Index stored in ``SiftPoint.match`` (index into the second set, or -1 if unmatched)."""

    x1: float
    """x position of the query keypoint."""

    y1: float
    """y position of the query keypoint."""

    x2: float
    """x position of the matched keypoint (from ``match_xpos``)."""

    y2: float
    """y position of the matched keypoint (from ``match_ypos``)."""

    error: float
    """Match error (L2 descriptor distance ratio)."""

    score: float
    """Match score of the query keypoint."""

    ambiguity: float
    """Match ambiguity of the query keypoint."""


@dataclass
class ExtractOptions:
    """Parameters for SIFT feature extraction.

    See ``ExtractSiftOptions_t`` in ``cusift.h`` for full documentation.
    """

    thresh: float = 3.0
    """Contrast threshold for DoG extrema (higher = fewer, more stable keypoints)."""

    lowest_scale: float = 0.0
    """Minimum feature scale in pixels (0.0 keeps all scales)."""

    edge_thresh: float = 10.0
    """Edge rejection threshold (ratio of principal curvatures)."""

    init_blur: float = 1.0
    """Assumed blur (sigma) of the input image."""

    max_keypoints: int = 32768
    """Maximum number of keypoints returned."""

    num_octaves: int = 5
    """Number of octave levels in the scale-space pyramid."""

    def _to_ctypes(self) -> ExtractSiftOptions_t:
        return ExtractSiftOptions_t(
            thresh_=self.thresh,
            lowest_scale_=self.lowest_scale,
            edge_thresh_=self.edge_thresh,
            init_blur_=self.init_blur,
            max_keypoints_=self.max_keypoints,
            num_octaves_=self.num_octaves,
        )


@dataclass
class HomographyOptions:
    """Parameters for RANSAC homography estimation.

    See ``FindHomographyOptions_t`` in ``cusift.h`` for full documentation.
    """

    num_loops: int = 10000
    min_score: float = 0.0
    max_ambiguity: float = 0.80
    thresh: float = 5.0
    improve_num_loops: int = 5
    improve_min_score: float = 0.0
    improve_max_ambiguity: float = 0.80
    improve_thresh: float = 3.0
    seed: int = 0

    def _to_ctypes(self) -> FindHomographyOptions_t:
        return FindHomographyOptions_t(
            num_loops_=self.num_loops,
            min_score_=self.min_score,
            max_ambiguity_=self.max_ambiguity,
            thresh_=self.thresh,
            improve_num_loops_=self.improve_num_loops,
            improve_min_score_=self.improve_min_score,
            improve_max_ambiguity_=self.improve_max_ambiguity,
            improve_thresh_=self.improve_thresh,
            seed_=self.seed,
        )


# -- Helper: build an Image_t from numpy -------------------------------------


def _make_image_t(pixels: np.ndarray, width: int, height: int) -> tuple[Image_t, np.ndarray]:
    """Wrap a contiguous float32 array in an ``Image_t``.

    Returns the struct *and* the backing array (to prevent GC).
    """
    if pixels.dtype != np.float32:
        pixels = pixels.astype(np.float32)
    pixels = np.ascontiguousarray(pixels)
    img = Image_t()
    img.host_img_ = pixels.ctypes.data_as(POINTER(c_float))
    img.width_ = width
    img.height_ = height
    return img, pixels


def _load_image_grayscale(path: Union[str, Path]) -> tuple[np.ndarray, int, int]:
    """Load *path* as a grayscale float32 image.  Returns ``(pixels, w, h)``."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for loading images from file paths.  "
            "Install it with:  pip install Pillow"
        ) from exc
    img = Image.open(path).convert("L")
    w, h = img.size
    arr = np.asarray(img, dtype=np.float32)
    return np.ascontiguousarray(arr), w, h


# -- Keypoint list with SiftData handle ---------------------------------------


class KeypointList(list):
    """A ``list[Keypoint]`` that also carries the underlying ``SiftData``.

    Users interact with this exactly like a normal list.  Internally the
    C ``SiftData`` struct is kept alive so it can be passed to
    :meth:`CuSift.match` and :meth:`CuSift.find_homography` without
    re-uploading descriptors to the GPU.

    Call :meth:`free` (or use as a context manager) to release GPU
    memory early.  Otherwise it is freed when the object is garbage
    collected.
    """

    def __init__(self, keypoints: List[Keypoint], sift_data: SiftData, lib: ctypes.CDLL):
        super().__init__(keypoints)
        self._sift_data = sift_data
        self._lib = lib
        self._freed = False

    # -- resource management --------------------------------------------------

    def free(self) -> None:
        """Release the underlying ``SiftData`` GPU/host memory."""
        if not self._freed:
            self._lib.DeleteSiftData(byref(self._sift_data))
            self._freed = True

    def __del__(self) -> None:
        self.free()

    def __enter__(self) -> "KeypointList":
        return self

    def __exit__(self, *exc) -> None:
        self.free()


# -- Main class ---------------------------------------------------------------


class CuSift:
    """High-level interface to the CuSIFT library.

    Parameters
    ----------
    dll_path : str | Path | None
        Explicit path to ``cusift.dll`` / ``libcusift.so``.
        When *None* the library is located automatically.

    Example
    -------
    >>> sift = CuSift()
    >>> keypoints = sift.extract("photo.png")
    >>> print(f"Found {len(keypoints)} SIFT features")
    """

    def __init__(self, dll_path: Optional[Union[str, Path]] = None):
        self._lib = load_library(dll_path)
        self._lib.InitializeCudaSift()
        _check_error(self._lib)

    # -- Feature extraction -----------------------------------------------

    def extract(
        self,
        image: Union[str, Path, np.ndarray],
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        options: Optional[ExtractOptions] = None,
    ) -> KeypointList:
        """Extract SIFT keypoints from an image.

        Parameters
        ----------
        image : str | Path | numpy.ndarray
            Either a file path (loaded as grayscale via Pillow) or a 2-D
            ``float32`` numpy array of shape ``(height, width)`` with pixel
            values in ``[0, 255]``.
        width, height : int, optional
            Required only when *image* is a 1-D array.  For 2-D arrays
            and file paths these are inferred automatically.
        options : ExtractOptions, optional
            Extraction parameters.  Uses :class:`ExtractOptions` defaults
            when not provided.

        Returns
        -------
        KeypointList
            Detected SIFT features (behaves like ``list[Keypoint]``).
            Retains the underlying ``SiftData`` so it can be passed
            directly to :meth:`match`.

        Raises
        ------
        CuSiftError
            If the underlying C library reports an error.
        """
        # -- Resolve pixel data -------------------------------------------
        if isinstance(image, (str, Path)):
            pixels, w, h = _load_image_grayscale(image)
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                h, w = image.shape
                pixels = image
            elif image.ndim == 1:
                if width is None or height is None:
                    raise ValueError(
                        "width and height must be supplied for 1-D arrays"
                    )
                w, h = width, height
                pixels = image
            else:
                raise ValueError(
                    f"Expected a 1-D or 2-D array, got shape {image.shape}"
                )
        else:
            raise TypeError(
                f"image must be a file path or numpy array, got {type(image)}"
            )

        # Override w/h if explicitly given
        if width is not None:
            w = width
        if height is not None:
            h = height

        # -- Build ctypes arguments ---------------------------------------
        img_ct, _pixels_ref = _make_image_t(pixels, w, h)
        sift_data = SiftData()
        opts_ct = (options or ExtractOptions())._to_ctypes()

        # -- Call the C function ------------------------------------------
        self._lib.ExtractSiftFromImage(
            byref(img_ct), byref(sift_data), byref(opts_ct)
        )
        _check_error(self._lib)

        # -- Convert results ----------------------------------------------
        keypoints: List[Keypoint] = []
        for i in range(sift_data.numPts):
            keypoints.append(Keypoint._from_sift_point(sift_data.h_data[i]))

        # Wrap in KeypointList (owns the SiftData; freed on GC or .free())
        return KeypointList(keypoints, sift_data, self._lib)

    # -- Feature matching -------------------------------------------------

    def match(
        self,
        kp1: KeypointList,
        kp2: KeypointList,
    ) -> List[MatchResult]:
        """Match SIFT features between two keypoint sets.

        Calls the C ``MatchSiftData`` function.  Match results are written
        into the ``match``, ``match_xpos``, ``match_ypos``, and
        ``match_error`` fields of *kp1*'s underlying ``SiftData``.  This
        method then reads those fields back and returns a list of
        :class:`MatchResult` for every keypoint in *kp1* that was
        successfully matched (``match >= 0``).

        The ``Keypoint`` objects inside *kp1* are also updated in-place
        so their ``match``, ``match_x``, ``match_y``, and ``match_error``
        attributes reflect the new correspondences.

        Parameters
        ----------
        kp1 : KeypointList
            Query keypoints (returned by :meth:`extract`).
        kp2 : KeypointList
            Target keypoints (returned by :meth:`extract`).

        Returns
        -------
        list[MatchResult]
            One entry per matched correspondence (unmatched keypoints
            are omitted).

        Raises
        ------
        CuSiftError
            If the underlying C library reports an error.
        TypeError
            If *kp1* or *kp2* are not :class:`KeypointList` instances.
        """
        if not isinstance(kp1, KeypointList) or not isinstance(kp2, KeypointList):
            raise TypeError(
                "match() requires KeypointList objects returned by extract()"
            )
        if kp1._freed or kp2._freed:
            raise RuntimeError(
                "Cannot match: underlying SiftData has been freed"
            )

        # -- Call the C function ------------------------------------------
        self._lib.MatchSiftData(
            byref(kp1._sift_data), byref(kp2._sift_data)
        )
        _check_error(self._lib)

        # -- Read back results and build MatchResult list -----------------
        results: List[MatchResult] = []
        sd = kp1._sift_data
        for i in range(sd.numPts):
            pt = sd.h_data[i]
            # Update the Python-side Keypoint as well
            kp1[i].match = pt.match
            kp1[i].match_x = pt.match_xpos
            kp1[i].match_y = pt.match_ypos
            kp1[i].match_error = pt.match_error
            kp1[i].score = pt.score
            kp1[i].ambiguity = pt.ambiguity

            if pt.match >= 0:
                results.append(
                    MatchResult(
                        query_index=i,
                        match_index=pt.match,
                        x1=pt.xpos,
                        y1=pt.ypos,
                        x2=pt.match_xpos,
                        y2=pt.match_ypos,
                        error=pt.match_error,
                        score=pt.score,
                        ambiguity=pt.ambiguity,
                    )
                )

        return results

    # -- Homography estimation --------------------------------------------

    def find_homography(
        self,
        kp: KeypointList,
        *,
        options: Optional[HomographyOptions] = None,
    ) -> tuple[np.ndarray, int]:
        """Estimate a homography from matched SIFT features.

        Calls the C ``FindHomography`` function.  The keypoints in *kp*
        must already contain valid match information (i.e. you should call
        :meth:`match` first).

        Parameters
        ----------
        kp : KeypointList
            Keypoints with match data (returned by :meth:`extract`,
            after calling :meth:`match`).
        options : HomographyOptions, optional
            RANSAC / refinement parameters.  Uses
            :class:`HomographyOptions` defaults when not provided.

        Returns
        -------
        (homography, num_inliers) : tuple[numpy.ndarray, int]
            *homography* is a ``(3, 3)`` float32 array in row-major order.
            *num_inliers* is the number of inlier matches used to compute
            the homography.

        Raises
        ------
        CuSiftError
            If the underlying C library reports an error.
        TypeError
            If *kp* is not a :class:`KeypointList`.
        """
        if not isinstance(kp, KeypointList):
            raise TypeError(
                "find_homography() requires a KeypointList returned by extract()"
            )
        if kp._freed:
            raise RuntimeError(
                "Cannot find homography: underlying SiftData has been freed"
            )

        # -- Build ctypes arguments ---------------------------------------
        homography = (c_float * 9)()
        num_matches = c_int(0)
        opts_ct = (options or HomographyOptions())._to_ctypes()

        # -- Call the C function ------------------------------------------
        self._lib.FindHomography(
            byref(kp._sift_data),
            homography,
            byref(num_matches),
            byref(opts_ct),
        )
        _check_error(self._lib)

        # -- Convert to numpy ---------------------------------------------
        H = np.ctypeslib.as_array(homography, shape=(9,)).copy().reshape(3, 3)

        return H, num_matches.value

    # -- Image warping ----------------------------------------------------

    def warp_images(
        self,
        image1: Union[str, Path, np.ndarray],
        image2: Union[str, Path, np.ndarray],
        homography: np.ndarray,
        *,
        use_gpu: bool = True,
        width1: Optional[int] = None,
        height1: Optional[int] = None,
        width2: Optional[int] = None,
        height2: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Warp two images using a homography so they are aligned.

        Calls the C ``WarpImages`` function.  The homography should be
        the output of :meth:`find_homography`.

        Parameters
        ----------
        image1 : str | Path | numpy.ndarray
            First input image (file path or 2-D float32 array).
        image2 : str | Path | numpy.ndarray
            Second input image (file path or 2-D float32 array).
        homography : numpy.ndarray
            A ``(3, 3)`` float32 homography matrix in row-major order.
        use_gpu : bool, optional
            Whether to use GPU acceleration for warping (default *True*).
        width1, height1 : int, optional
            Dimensions override for *image1* (only needed for 1-D arrays).
        width2, height2 : int, optional
            Dimensions override for *image2* (only needed for 1-D arrays).

        Returns
        -------
        (warped1, warped2) : tuple[numpy.ndarray, numpy.ndarray]
            Two 2-D ``float32`` arrays of shape ``(height, width)``
            containing the warped images.  The caller owns these arrays;
            the underlying C memory is copied and then freed.

        Raises
        ------
        CuSiftError
            If the underlying C library reports an error.
        """
        # -- Resolve pixel data for image 1 -------------------------------
        if isinstance(image1, (str, Path)):
            pix1, w1, h1 = _load_image_grayscale(image1)
        elif isinstance(image1, np.ndarray):
            if image1.ndim == 2:
                h1, w1 = image1.shape
            elif image1.ndim == 1:
                if width1 is None or height1 is None:
                    raise ValueError(
                        "width1 and height1 must be supplied for 1-D arrays"
                    )
                w1, h1 = width1, height1
            else:
                raise ValueError(
                    f"Expected a 1-D or 2-D array for image1, got shape {image1.shape}"
                )
            pix1 = image1
        else:
            raise TypeError(
                f"image1 must be a file path or numpy array, got {type(image1)}"
            )
        if width1 is not None:
            w1 = width1
        if height1 is not None:
            h1 = height1

        # -- Resolve pixel data for image 2 -------------------------------
        if isinstance(image2, (str, Path)):
            pix2, w2, h2 = _load_image_grayscale(image2)
        elif isinstance(image2, np.ndarray):
            if image2.ndim == 2:
                h2, w2 = image2.shape
            elif image2.ndim == 1:
                if width2 is None or height2 is None:
                    raise ValueError(
                        "width2 and height2 must be supplied for 1-D arrays"
                    )
                w2, h2 = width2, height2
            else:
                raise ValueError(
                    f"Expected a 1-D or 2-D array for image2, got shape {image2.shape}"
                )
            pix2 = image2
        else:
            raise TypeError(
                f"image2 must be a file path or numpy array, got {type(image2)}"
            )
        if width2 is not None:
            w2 = width2
        if height2 is not None:
            h2 = height2

        # -- Build ctypes arguments ---------------------------------------
        img1_ct, _pix1_ref = _make_image_t(pix1, w1, h1)
        img2_ct, _pix2_ref = _make_image_t(pix2, w2, h2)

        H_flat = np.ascontiguousarray(homography.ravel(), dtype=np.float32)
        h_ct = H_flat.ctypes.data_as(POINTER(c_float))

        warped1_ct = Image_t()
        warped2_ct = Image_t()

        # -- Call the C function ------------------------------------------
        self._lib.WarpImages(
            byref(img1_ct),
            byref(img2_ct),
            h_ct,
            byref(warped1_ct),
            byref(warped2_ct),
            c_bool(use_gpu),
        )
        _check_error(self._lib)

        # -- Copy warped pixels into numpy arrays -------------------------
        #    The C library allocates warped_image.host_img_ with malloc();
        #    we copy the data and then free the C-side buffer via FreeImage().
        n1 = warped1_ct.width_ * warped1_ct.height_
        n2 = warped2_ct.width_ * warped2_ct.height_

        warped1 = np.ctypeslib.as_array(warped1_ct.host_img_, shape=(n1,)).copy()
        warped1 = warped1.reshape(warped1_ct.height_, warped1_ct.width_)

        warped2 = np.ctypeslib.as_array(warped2_ct.host_img_, shape=(n2,)).copy()
        warped2 = warped2.reshape(warped2_ct.height_, warped2_ct.width_)

        # Free the C-allocated pixel buffers via the library's own free
        self._lib.FreeImage(byref(warped1_ct))
        self._lib.FreeImage(byref(warped2_ct))

        return warped1, warped2

