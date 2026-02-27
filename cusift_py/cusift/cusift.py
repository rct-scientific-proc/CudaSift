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


def _resolve_image_arg(
    image: Union[str, Path, np.ndarray],
    width: Optional[int] = None,
    height: Optional[int] = None,
    label: str = "image",
) -> tuple[np.ndarray, int, int]:
    """Turn an image argument into ``(pixels, width, height)``.

    Accepts a file path (loaded via Pillow), a 2-D numpy array
    ``(H, W)``, or a 1-D array (requires explicit *width*/*height*).

    Returns
    -------
    (pixels, w, h) : tuple[numpy.ndarray, int, int]
    """
    if isinstance(image, (str, Path)):
        pixels, w, h = _load_image_grayscale(image)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            h, w = image.shape
            pixels = image
        elif image.ndim == 1:
            if width is None or height is None:
                raise ValueError(
                    f"width and height must be supplied for 1-D arrays ({label})"
                )
            w, h = width, height
            pixels = image
        else:
            raise ValueError(
                f"Expected a 1-D or 2-D array for {label}, got shape {image.shape}"
            )
    else:
        raise TypeError(
            f"{label} must be a file path or numpy array, got {type(image)}"
        )

    # Explicit overrides
    if width is not None:
        w = width
    if height is not None:
        h = height

    return pixels, w, h


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
        pixels, w, h = _resolve_image_arg(image, width, height)

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
        # -- Resolve pixel data -------------------------------------------
        pix1, w1, h1 = _resolve_image_arg(image1, width1, height1, "image1")
        pix2, w2, h2 = _resolve_image_arg(image2, width2, height2, "image2")

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

    # -- Visualisation ----------------------------------------------------

    @staticmethod
    def draw_keypoints(
        image: Union[str, Path, np.ndarray],
        keypoints: KeypointList,
        output: Union[str, Path],
        *,
        radius_scale: float = 3.0,
        orientation_color: str = "red",
        line_width: int = 1,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        """Draw keypoints on an image and save to *output*.

        Each keypoint is drawn as a circle (scaled by ``scale``) with
        an orientation tick line.  The circle colour is derived from the
        keypoint's ``subsampling`` value — the set of unique subsampling values is mapped
        across an HSV hue ramp so that different octaves are visually
        distinct.

        Parameters
        ----------
        image : str | Path | numpy.ndarray
            Source image (file path or 2-D float32 array).
        keypoints : KeypointList
            Keypoints to draw (returned by :meth:`extract`).
        output : str | Path
            File path for the annotated image (e.g. ``"features.png"``).
        radius_scale : float
            Multiplier applied to each keypoint's ``scale`` to determine
            the circle radius (default 3.0).
        orientation_color : str
            Colour for the orientation tick (default ``"red"``).
        line_width : int
            Stroke width for circles and ticks (default 1).
        width, height : int, optional
            Dimension overrides when *image* is a 1-D array.
        """
        import colorsys
        import math

        try:
            from PIL import Image, ImageDraw
        except ImportError as exc:
            raise ImportError(
                "Pillow is required for draw_keypoints.  "
                "Install it with:  pip install Pillow"
            ) from exc

        # -- Build a subsampling → colour mapping -------------------------
        subsample = set(sorted({kp.subsampling for kp in keypoints}))
        if len(subsample) <= 1:
            scale_color = {s: (0, 255, 0) for s in subsample}  # fallback green
        else:
            scale_color = {}
            for idx, s in enumerate(subsample):
                hue = idx / len(subsample)  # 0 … ~1 spread across the hue wheel
                r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
                scale_color[s] = (int(r * 255), int(g * 255), int(b * 255))

        # -- Load / convert to a PIL Image --------------------------------
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pixels, w, h = _resolve_image_arg(image, width, height, "image")
            arr = np.nan_to_num(pixels, nan=0.0)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            if arr.ndim == 1:
                arr = arr.reshape(h, w)
            img = Image.fromarray(arr, mode="L").convert("RGB")
        else:
            raise TypeError(
                f"image must be a file path or numpy array, got {type(image)}"
            )

        draw = ImageDraw.Draw(img)

        for kp in keypoints:
            x, y = kp.x, kp.y
            s = kp.scale * radius_scale
            orient = kp.orientation
            rgb = scale_color[kp.subsampling]

            # Circle at keypoint location
            draw.ellipse(
                [x - s, y - s, x + s, y + s],
                outline=rgb,
                width=line_width,
            )

            # Orientation tick
            dx = s * math.cos(orient)
            dy = s * math.sin(orient)
            draw.line(
                [(x, y), (x + dx, y + dy)],
                fill=orientation_color,
                width=line_width,
            )

        img.save(str(output))

    @staticmethod
    def draw_descriptors(
        image: Union[str, Path, np.ndarray],
        keypoints: KeypointList,
        min_sampling: float,
        output: Union[str, Path],
        *,
        patch_size: int = 256,
        grid_size: int = 256,
        spoke_color: str = "cyan",
        bg_color: str = "black",
        grid_line_color: str = "#444444",
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        """Visualise a single keypoint's 128-d SIFT descriptor.

        Produces a side-by-side image: the left half shows the image
        patch centred on the keypoint (with a crosshair and circle),
        and the right half shows the 4x4 grid of 8-bin orientation
        histograms (rose/polar plots) that make up the descriptor.

        Parameters
        ----------
        image : str | Path | numpy.ndarray
            Source image (file path or 2-D float32 array).
        keypoints : KeypointList
            Keypoints returned by :meth:`extract`.
        min_sampling : float
            Minimum sampling rate for keypoints to be drawn.
        output : str | Path
            File path for saving the descriptors that pass the sampling threshold. We create sub-directories.
        patch_size : int
            Size in pixels of the image-patch panel (default 256).
        grid_size : int
            Size in pixels of the histogram-grid panel (default 256).
        spoke_color : str
            Colour for histogram spoke lines (default ``"cyan"``).
        bg_color : str
            Background colour for the histogram panel (default ``"black"``).
        grid_line_color : str
            Colour for the 4x4 grid lines (default ``"#444444"``).
        width, height : int, optional
            Dimension overrides when *image* is a 1-D array.
        """
        import math

        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as exc:
            raise ImportError(
                "Pillow is required for draw_descriptor.  "
                "Install it with:  pip install Pillow"
            ) from exc

        for index, kp in enumerate(keypoints):
            if kp.subsampling < min_sampling:
                continue  # skip keypoints below the sampling threshold
            kp = keypoints[index]
            desc = kp.descriptor  # shape (128,)

            # -- Load the source image ----------------------------------------
            if isinstance(image, (str, Path)):
                src = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                pixels, w, h = _resolve_image_arg(image, width, height, "image")
                arr = np.nan_to_num(pixels, nan=0.0)
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                if arr.ndim == 1:
                    arr = arr.reshape(h, w)
                src = Image.fromarray(arr, mode="L").convert("RGB")
            else:
                raise TypeError(
                    f"image must be a file path or numpy array, got {type(image)}"
                )

            # -- Left panel: image patch centred on keypoint ------------------
            # Crop a square region around the keypoint, then resize.
            radius = max(kp.scale * 6, 16)  # visible patch radius in src pixels
            cx, cy = kp.x, kp.y
            left = int(cx - radius)
            top = int(cy - radius)
            right = int(cx + radius)
            bottom = int(cy + radius)

            # Pad if the crop box extends outside the image
            img_w, img_h = src.size
            pad_l = max(0, -left)
            pad_t = max(0, -top)
            pad_r = max(0, right - img_w)
            pad_b = max(0, bottom - img_h)

            crop_left = max(left, 0)
            crop_top = max(top, 0)
            crop_right = min(right, img_w)
            crop_bottom = min(bottom, img_h)

            patch = src.crop((crop_left, crop_top, crop_right, crop_bottom))
            if pad_l or pad_t or pad_r or pad_b:
                padded = Image.new("RGB", (right - left, bottom - top), (0, 0, 0))
                padded.paste(patch, (pad_l, pad_t))
                patch = padded

            patch = patch.resize((patch_size, patch_size), Image.LANCZOS)
            draw_patch = ImageDraw.Draw(patch)

            # Draw crosshair at centre
            mid = patch_size // 2
            ch_len = 10
            draw_patch.line(
                [(mid - ch_len, mid), (mid + ch_len, mid)], fill="red", width=1
            )
            draw_patch.line(
                [(mid, mid - ch_len), (mid, mid + ch_len)], fill="red", width=1
            )
            # Draw circle showing the keypoint scale
            scale_r = (kp.scale * 3) / (2 * radius) * patch_size
            draw_patch.ellipse(
                [mid - scale_r, mid - scale_r, mid + scale_r, mid + scale_r],
                outline="lime",
                width=1,
            )

            # -- Right panel: 4x4 histogram grid -----------------------------
            hist_panel = Image.new("RGB", (grid_size, grid_size), bg_color)
            draw_hist = ImageDraw.Draw(hist_panel)

            cell = grid_size / 4  # cell size in pixels
            half = cell / 2

            # Reshape descriptor: 4 rows x 4 cols x 8 orientations
            bins = desc.reshape(4, 4, 8)
            max_val = bins.max() if bins.max() > 0 else 1.0  # normalise

            # Draw grid lines
            for i in range(1, 4):
                coord = int(i * cell)
                draw_hist.line(
                    [(coord, 0), (coord, grid_size)],
                    fill=grid_line_color,
                    width=1,
                )
                draw_hist.line(
                    [(0, coord), (grid_size, coord)],
                    fill=grid_line_color,
                    width=1,
                )

            # Draw 8-bin rose plot in each cell
            for row in range(4):
                for col in range(4):
                    cx_h = col * cell + half
                    cy_h = row * cell + half
                    max_spoke = half * 0.85  # leave a small margin

                    for b in range(8):
                        angle = b * (2 * math.pi / 8) - math.pi / 2  # 0 = up
                        magnitude = bins[row, col, b] / max_val
                        spoke_len = magnitude * max_spoke

                        ex = cx_h + spoke_len * math.cos(angle)
                        ey = cy_h + spoke_len * math.sin(angle)
                        draw_hist.line(
                            [(cx_h, cy_h), (ex, ey)],
                            fill=spoke_color,
                            width=2,
                        )

            # -- Combine panels side-by-side ----------------------------------
            total_w = patch_size + grid_size
            total_h = max(patch_size, grid_size)
            combined = Image.new("RGB", (total_w, total_h), (0, 0, 0))
            combined.paste(patch, (0, 0))
            combined.paste(hist_panel, (patch_size, 0))

            # Add a thin separator line
            draw_combined = ImageDraw.Draw(combined)
            draw_combined.line(
                [(patch_size, 0), (patch_size, total_h)],
                fill="white",
                width=1,
            )

            # Label the panels
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except (IOError, OSError):
                font = ImageFont.load_default()

            draw_combined.text(
                (4, 4),
                f"Keypoint #{index}  scale={kp.scale:.2f}  sub={kp.subsampling:.0f}",
                fill="yellow",
                font=font,
            )
            draw_combined.text(
                (patch_size + 4, 4),
                "4\u00d74\u00d78 Descriptor",
                fill="yellow",
                font=font,
            )

            # -- Save the combined image --------------------------------------
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Sub-directory based on subsampling value (e.g. "output/sub_1.0/desc_0.png")
            sub_dir = output_path.parent / f"sub_{kp.subsampling:.1f}"
            sub_dir.mkdir(parents=True, exist_ok=True)

            # Filename based on keypoint index and (x,y) position
            filename = f"desc_{index}_x{int(kp.x)}_y{int(kp.y)}.png"
            combined.save(sub_dir / filename)
    