"""
High-level Pythonic interface to the CuSIFT GPU-accelerated SIFT library.

All public symbols are re-exported from the package ``__init__.py``.
"""

from __future__ import annotations

import ctypes
import weakref
from ctypes import POINTER, byref, c_bool, c_float, c_int
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np

from cusift._bindings import (
    cusift_error_callback,
    cusift_error_record,
    cusift_extract_options,
    cusift_homography_options,
    cusift_image,
    cusift_keypoints,
    cusift_keypoint,
    cusift_canvas,
    load_library,
)

# Limits mirrored from the C library.  The C side rejects these too, but a
# Python ValueError is friendlier than a CuSiftError from inside the call.
_MAX_KEYPOINTS_LIMIT = (2**31 - 1) // ctypes.sizeof(cusift_keypoint)
_MAX_RANSAC_LOOPS = 65535 * 16
_MIN_IMAGE_DIM = 8
_MAX_IMAGE_DIM = 65535


# -- Exceptions ---------------------------------------------------------------


class CuSiftError(Exception):
    """Raised when the C library reports an error."""

    def __init__(self, message: str, filename: str = "", line: int = 0):
        self.filename = filename
        self.line = line
        super().__init__(message)


# -- Homography goal constants ------------------------------------------------

GOAL_MAX_INLIERS = 0
"""Maximise the number of RANSAC inliers."""

GOAL_MIN_IDENTITY_DISTANCE = 1
"""Minimise the Frobenius distance between the top-left 2x2 block and the identity matrix."""


# -- Geometric model constants ------------------------------------------------

MODEL_HOMOGRAPHY = 0
"""Full 8-DOF projective homography (DLT + RANSAC from 4-point samples)."""

MODEL_SIMILARITY = 1
"""4-DOF similarity transform (rotation + uniform scale + translation)."""


def _check_error(lib: ctypes.CDLL) -> None:
    """Raise :class:`CuSiftError` if the last library call on this thread failed."""
    rec = cusift_error_record()
    if lib.cusift_last_error(byref(rec)):
        def s(b):
            return b.decode("utf-8", errors="replace") if b else ""
        raise CuSiftError(s(rec.message), filename=s(rec.file), line=int(rec.line))


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
    def _from_sift_point(cls, pt: cusift_keypoint) -> "Keypoint":
        desc = np.ctypeslib.as_array(pt.descriptor, shape=(128,)).copy()
        return cls(
            x=pt.x,
            y=pt.y,
            scale=pt.scale,
            sharpness=pt.sharpness,
            edgeness=pt.edgeness,
            orientation=pt.orientation,
            score=pt.score,
            ambiguity=pt.ambiguity,
            match=pt.match,
            match_x=pt.match_x,
            match_y=pt.match_y,
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
    """Index stored in ``cusift_keypoint.match`` (index into the second set, or -1 if unmatched)."""

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
class ErrorRecord:
    """One error or warning reported by the library (see ``cusift_error_record``).

    ``severity`` is 0 for an error (the call failed) and 1 for a warning (the
    call completed in a degraded way).
    """

    severity: int
    function: str
    file: str
    line: int
    cuda_error: int
    cuda_error_name: str
    message: str
    thread_id: int
    timestamp_ms: int

    @classmethod
    def _from_ctypes(cls, r: cusift_error_record) -> "ErrorRecord":
        def s(b: Optional[bytes]) -> str:
            return b.decode("utf-8", errors="replace") if b else ""
        return cls(severity=int(r.severity), function=s(r.function), file=s(r.file), line=int(r.line),
                   cuda_error=int(r.cuda_error), cuda_error_name=s(r.cuda_error_name), message=s(r.message),
                   thread_id=int(r.thread_id), timestamp_ms=int(r.timestamp_ms))


@dataclass
class Canvas:
    """Where the two inputs land on the common canvas (see ``cusift_canvas``).

    Returned by :meth:`CuSift.compute_canvas`.  The canvas is image 1's
    frame extended to hold image 2; canvas pixel ``(0, 0)`` is image-1
    coordinate ``origin``.  Corners are in canvas pixel coordinates, in the
    order ``(0, 0), (w-1, 0), (w-1, h-1), (0, h-1)`` of the source image.
    """

    origin: tuple[int, int]
    """``(origin_u, origin_v)``: image-1 coordinate of canvas pixel (0, 0); both <= 0."""

    canvas_size: tuple[int, int]
    """``(width, height)`` of the warped images."""

    corners1: np.ndarray
    """``(4, 2)`` float32 array: image-1 corners on the canvas (integers)."""

    corners2: np.ndarray
    """``(4, 2)`` float32 array: image-2 corners on the canvas (sub-pixel)."""

    @classmethod
    def _from_ctypes(cls, g: cusift_canvas) -> "Canvas":
        return cls(
            origin=(int(g.origin_u), int(g.origin_v)),
            canvas_size=(int(g.canvas_width), int(g.canvas_height)),
            corners1=np.array([[g.corners1[i][0], g.corners1[i][1]] for i in range(4)], dtype=np.float32),
            corners2=np.array([[g.corners2[i][0], g.corners2[i][1]] for i in range(4)], dtype=np.float32),
        )


@dataclass
class ExtractOptions:
    """Parameters for SIFT extraction.

    Field names and defaults are those of ``cusift_extract_options`` and
    ``cusift_default_extract_options()`` in ``cusift.h``, which documents each.
    """

    contrast_threshold: float = 2.0
    lowest_scale: float = 0.0
    highest_scale: float = float("inf")
    edge_threshold: float = 10.0
    initial_blur: float = 1.0
    max_keypoints: int = 32768
    num_octaves: int = 5
    scale_suppression_radius: float = 0.0

    def _to_ctypes(self) -> cusift_extract_options:
        if not (1 <= int(self.max_keypoints) <= _MAX_KEYPOINTS_LIMIT):
            raise ValueError(
                f"max_keypoints must be in [1, {_MAX_KEYPOINTS_LIMIT}], got {self.max_keypoints}"
            )
        return cusift_extract_options(
            contrast_threshold=self.contrast_threshold,
            lowest_scale=self.lowest_scale,
            highest_scale=self.highest_scale,
            edge_threshold=self.edge_threshold,
            initial_blur=self.initial_blur,
            max_keypoints=self.max_keypoints,
            num_octaves=self.num_octaves,
            scale_suppression_radius=self.scale_suppression_radius,
        )


@dataclass
class HomographyOptions:
    """Parameters for homography estimation.

    Field names and defaults are those of ``cusift_homography_options`` and
    ``cusift_default_homography_options()`` in ``cusift.h``, which documents
    each.  ``attempts`` > 1 runs the estimate several times and keeps one by
    ``goal`` (:data:`GOAL_MAX_INLIERS` or :data:`GOAL_MIN_IDENTITY_DISTANCE`).
    """

    ransac_iterations: int = 10000
    min_score: float = 0.0
    max_ambiguity: float = 0.8
    inlier_threshold: float = 3.0
    refine_iterations: int = 5
    refine_min_score: float = 0.0
    refine_max_ambiguity: float = 0.8
    refine_inlier_threshold: float = 2.0
    seed: int = 0
    model: int = MODEL_HOMOGRAPHY
    attempts: int = 1
    goal: int = GOAL_MAX_INLIERS

    def _to_ctypes(self) -> cusift_homography_options:
        if not (1 <= int(self.ransac_iterations) <= _MAX_RANSAC_LOOPS):
            raise ValueError(
                f"ransac_iterations must be in [1, {_MAX_RANSAC_LOOPS}], got {self.ransac_iterations}"
            )
        if int(self.refine_iterations) < 0:
            raise ValueError(f"refine_iterations must be >= 0, got {self.refine_iterations}")
        if int(self.attempts) < 1:
            raise ValueError(f"attempts must be >= 1, got {self.attempts}")
        return cusift_homography_options(
            ransac_iterations=self.ransac_iterations,
            min_score=self.min_score,
            max_ambiguity=self.max_ambiguity,
            inlier_threshold=self.inlier_threshold,
            refine_iterations=self.refine_iterations,
            refine_min_score=self.refine_min_score,
            refine_max_ambiguity=self.refine_max_ambiguity,
            refine_inlier_threshold=self.refine_inlier_threshold,
            seed=self.seed,
            model=self.model,
            attempts=self.attempts,
            goal=self.goal,
        )


# -- Helper: build an cusift_image from numpy -------------------------------------


def _make_image_t(pixels: np.ndarray, width: int, height: int) -> tuple[cusift_image, np.ndarray]:
    """Wrap a contiguous float32 array in an ``cusift_image``.

    Returns the struct *and* the backing array (to prevent GC).
    """
    if pixels.dtype != np.float32:
        pixels = pixels.astype(np.float32)
    pixels = np.ascontiguousarray(pixels)
    img = cusift_image()
    img.pixels = pixels.ctypes.data_as(POINTER(c_float))
    img.width = width
    img.height = height
    return img, pixels


def _check_dims(label: str, width: int, height: int) -> tuple[int, int]:
    """Validate an image size against the library's [8, 65535] limit.

    The ``cusift_memory_*`` functions do not validate their arguments or set
    the error state: a zero dimension reaches ``log2(0)`` and a negative one
    wraps to a huge ``size_t``.  So the checks are done here, with the same
    bounds the extraction and warp entry points enforce.
    """
    w, h = int(width), int(height)
    if not (_MIN_IMAGE_DIM <= w <= _MAX_IMAGE_DIM and _MIN_IMAGE_DIM <= h <= _MAX_IMAGE_DIM):
        raise ValueError(
            f"{label}: width and height must be in [{_MIN_IMAGE_DIM}, {_MAX_IMAGE_DIM}], got {w}x{h}"
        )
    return w, h


def _check_capacity(label: str, capacity: int) -> int:
    """Validate a keypoint capacity against the same bound as ``max_keypoints``."""
    c = int(capacity)
    if not (1 <= c <= _MAX_KEYPOINTS_LIMIT):
        raise ValueError(f"{label} must be in [1, {_MAX_KEYPOINTS_LIMIT}], got {capacity}")
    return c


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

    # The C library reads exactly width*height floats from the buffer, so a
    # mismatch here would be an out-of-bounds read of the numpy allocation.
    w, h = int(w), int(h)
    if w <= 0 or h <= 0:
        raise ValueError(f"{label}: width and height must be positive, got {w}x{h}")
    if w * h != pixels.size:
        raise ValueError(
            f"{label}: width*height ({w}*{h}={w * h}) does not match the "
            f"number of pixels in the array ({pixels.size})"
        )

    return pixels, w, h


# -- Keypoint list with cusift_keypoints handle ---------------------------------------


class Keypoints(list):
    """A ``list[Keypoint]`` that also carries the underlying ``cusift_keypoints``.

    Users interact with this exactly like a normal list.  Internally the
    C ``cusift_keypoints`` struct is kept alive so it can be passed to
    :meth:`CuSift.match` and :meth:`CuSift.estimate_homography` without
    re-uploading descriptors to the GPU.

    Call :meth:`free` (or use as a context manager) to release GPU
    memory early.  Otherwise it is freed when the object is garbage
    collected.
    """

    def __init__(self, keypoints: List[Keypoint], sift_data: cusift_keypoints, lib: ctypes.CDLL):
        super().__init__(keypoints)
        self._sift_data = sift_data
        self._lib = lib
        # Use weakref.finalize instead of __del__ so cleanup is robust against
        # reference cycles (tracebacks, REPL `_`, Optuna trial state) and runs
        # reliably even at interpreter shutdown.
        self._finalizer = weakref.finalize(
            self, Keypoints._cleanup, lib, sift_data
        )

    @staticmethod
    def _cleanup(lib: ctypes.CDLL, sift_data: cusift_keypoints) -> None:
        # `byref` and `lib` may not be available during interpreter shutdown;
        # swallow any exception so the finalizer never reports an error.
        try:
            lib.cusift_keypoints_free(ctypes.byref(sift_data))
        except Exception:
            pass

    # -- resource management --------------------------------------------------

    @property
    def _freed(self) -> bool:
        return not self._finalizer.alive

    def free(self) -> None:
        """Release the underlying ``cusift_keypoints`` GPU/host memory."""
        # finalize.__call__ runs the cleanup atomically and marks it inactive;
        # subsequent calls are no-ops.
        self._finalizer()

    def __enter__(self) -> "Keypoints":
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
        self._error_cb = None  # keeps the ctypes callback object alive while registered
        self._lib.cusift_init()
        _check_error(self._lib)

    # -- Version and error sinks --------------------------------------------

    def version(self) -> tuple[int, int, int]:
        """(major, minor, patch) of the loaded library (C ``cusift_version``)."""
        major, minor, patch = c_int(0), c_int(0), c_int(0)
        self._lib.cusift_version(byref(major), byref(minor), byref(patch))
        return major.value, minor.value, patch.value

    def set_error_log(self, path: Optional[Union[str, Path]] = "", max_bytes: int = 0) -> None:
        """Append every library error and warning to a file.

        Calls the C ``cusift_set_error_log`` function.  ``path=None`` turns the
        file off, ``""`` (the default) uses ``<temp dir>/cusift-errors.log``,
        anything else is the file to append to.  ``max_bytes`` is the rotation
        cap (0 = 1 MiB): when exceeded the file is renamed to ``<path>.1`` and
        restarted.  Logging never raises and never affects the call being logged.
        """
        if path is None:
            self._lib.cusift_set_error_log(None, 0)
        else:
            self._lib.cusift_set_error_log(str(path).encode("utf-8"), int(max_bytes))

    def set_error_callback(self, callback: Optional[Callable[[ErrorRecord], None]]) -> None:
        """Deliver every library error and warning to *callback* as well.

        Calls the C ``cusift_set_error_callback`` function.  The callback receives
        an :class:`ErrorRecord` and runs on the thread where the error occurred.
        An exception raised inside it is printed by ctypes and otherwise
        ignored.  Pass ``None`` to remove the callback.  Process-wide: the last
        registration wins, whichever :class:`CuSift` instance made it.
        """
        if callback is None:
            self._lib.cusift_set_error_callback(cusift_error_callback(0), None)
            self._error_cb = None
            return

        def trampoline(rec_ptr, _user):
            callback(ErrorRecord._from_ctypes(rec_ptr.contents))

        cb = cusift_error_callback(trampoline)
        self._lib.cusift_set_error_callback(cb, None)
        self._error_cb = cb

    def error_log_dropped(self) -> int:
        """Number of log records that could not be written to the error log."""
        return int(self._lib.cusift_error_log_dropped())

    def reset_device(self) -> None:
        """Recover from a sticky CUDA error by resetting this process's context.

        Calls the C ``cusift_device_reset`` function.  The reset is process-local
        (other processes sharing the GPU are unaffected), but it destroys every
        device allocation of this process: free every :class:`Keypoints`
        first and do not reuse any GPU-resident result afterwards.

        Raises
        ------
        CuSiftError
            If the device is still unusable after the reset.
        """
        self._lib.cusift_device_reset()
        _check_error(self._lib)

    # -- Feature extraction -----------------------------------------------

    def extract(
        self,
        image: Union[str, Path, np.ndarray],
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        options: Optional[ExtractOptions] = None,
    ) -> Keypoints:
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
        Keypoints
            Detected SIFT features (behaves like ``list[Keypoint]``).
            Retains the underlying ``cusift_keypoints`` so it can be passed
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
        sift_data = cusift_keypoints()
        opts_ct = (options or ExtractOptions())._to_ctypes()

        # -- Call the C function ------------------------------------------
        self._lib.cusift_extract(
            byref(img_ct), byref(sift_data), byref(opts_ct)
        )
        try:
            _check_error(self._lib)

            # -- Convert results ------------------------------------------
            keypoints: List[Keypoint] = []
            for i in range(sift_data.count):
                keypoints.append(Keypoint._from_sift_point(sift_data.host[i]))

            # Wrap in Keypoints (owns the cusift_keypoints; freed on GC or .free())
            return Keypoints(keypoints, sift_data, self._lib)
        except:
            self._lib.cusift_keypoints_free(byref(sift_data))
            raise

    # -- Feature matching -------------------------------------------------

    def match(
        self,
        kp1: Keypoints,
        kp2: Keypoints,
    ) -> List[MatchResult]:
        """Match SIFT features between two keypoint sets.

        Calls the C ``cusift_match`` function.  Match results are written
        into the ``match``, ``match_xpos``, ``match_ypos``, and
        ``match_error`` fields of *kp1*'s underlying ``cusift_keypoints``.  This
        method then reads those fields back and returns a list of
        :class:`MatchResult` for every keypoint in *kp1* that was
        successfully matched (``match >= 0``).

        The ``Keypoint`` objects inside *kp1* are also updated in-place
        so their ``match``, ``match_x``, ``match_y``, and ``match_error``
        attributes reflect the new correspondences.

        Parameters
        ----------
        kp1 : Keypoints
            Query keypoints (returned by :meth:`extract`).
        kp2 : Keypoints
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
            If *kp1* or *kp2* are not :class:`Keypoints` instances.
        """
        if not isinstance(kp1, Keypoints) or not isinstance(kp2, Keypoints):
            raise TypeError(
                "match() requires Keypoints objects returned by extract()"
            )
        if kp1._freed or kp2._freed:
            raise RuntimeError(
                "Cannot match: underlying cusift_keypoints has been freed"
            )

        # -- Call the C function ------------------------------------------
        self._lib.cusift_match(
            byref(kp1._sift_data), byref(kp2._sift_data)
        )
        _check_error(self._lib)

        # -- Read back results and build MatchResult list -----------------
        results: List[MatchResult] = []
        sd = kp1._sift_data
        for i in range(sd.count):
            pt = sd.host[i]
            # Update the Python-side Keypoint as well
            kp1[i].match = pt.match
            kp1[i].match_x = pt.match_x
            kp1[i].match_y = pt.match_y
            kp1[i].match_error = pt.match_error
            kp1[i].score = pt.score
            kp1[i].ambiguity = pt.ambiguity

            if pt.match >= 0:
                results.append(
                    MatchResult(
                        query_index=i,
                        match_index=pt.match,
                        x1=pt.x,
                        y1=pt.y,
                        x2=pt.match_x,
                        y2=pt.match_y,
                        error=pt.match_error,
                        score=pt.score,
                        ambiguity=pt.ambiguity,
                    )
                )

        return results

    # -- Homography estimation --------------------------------------------

    def estimate_homography(
        self,
        kp: Keypoints,
        *,
        options: Optional[HomographyOptions] = None,
    ) -> tuple[np.ndarray, int]:
        """Estimate a homography from matched SIFT features.

        Calls the C ``cusift_estimate_homography`` function.  The keypoints in *kp*
        must already contain valid match information (i.e. you should call
        :meth:`match` first).

        Parameters
        ----------
        kp : Keypoints
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
            If *kp* is not a :class:`Keypoints`.
        """
        if not isinstance(kp, Keypoints):
            raise TypeError(
                "estimate_homography() requires a Keypoints returned by extract()"
            )
        if kp._freed:
            raise RuntimeError(
                "Cannot find homography: underlying cusift_keypoints has been freed"
            )

        # -- Build ctypes arguments ---------------------------------------
        homography = (c_float * 9)()
        num_matches = c_int(0)
        opts_ct = (options or HomographyOptions())._to_ctypes()

        # -- Call the C function ------------------------------------------
        self._lib.cusift_estimate_homography(
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

    def warp(
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

        Calls the C ``cusift_warp`` function.  The homography should be
        the output of :meth:`estimate_homography`.

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

        H_flat = np.ascontiguousarray(np.asarray(homography, dtype=np.float32).ravel())
        if H_flat.size != 9:
            raise ValueError(
                f"homography must have 9 elements (3x3), got shape {np.shape(homography)}"
            )
        h_ct = H_flat.ctypes.data_as(POINTER(c_float))

        warped1_ct = cusift_image()
        warped2_ct = cusift_image()

        # -- Call the C function ------------------------------------------
        self._lib.cusift_warp(
            byref(img1_ct),
            byref(img2_ct),
            h_ct,
            byref(warped1_ct),
            byref(warped2_ct),
            c_bool(use_gpu),
        )

        # -- Copy warped pixels into numpy arrays -------------------------
        #    The C library allocates warped_image.pixels with malloc();
        #    we copy the data and then free the C-side buffer via cusift_image_free().
        #    cusift_image_free() is a no-op when pixels is NULL, so it is safe to
        #    call unconditionally even if the C call errored out.
        try:
            _check_error(self._lib)

            n1 = warped1_ct.width * warped1_ct.height
            n2 = warped2_ct.width * warped2_ct.height

            warped1 = np.ctypeslib.as_array(warped1_ct.pixels, shape=(n1,)).copy()
            warped1 = warped1.reshape(warped1_ct.height, warped1_ct.width)

            warped2 = np.ctypeslib.as_array(warped2_ct.pixels, shape=(n2,)).copy()
            warped2 = warped2.reshape(warped2_ct.height, warped2_ct.width)

            return warped1, warped2
        finally:
            # Free the C-allocated pixel buffers via the library's own free
            self._lib.cusift_image_free(byref(warped1_ct))
            self._lib.cusift_image_free(byref(warped2_ct))

    def compute_canvas(
        self,
        homography: np.ndarray,
        width1: int,
        height1: int,
        width2: int,
        height2: int,
    ) -> Canvas:
        """Compute the canvas that :meth:`warp` would produce, without warping.

        Calls the C ``cusift_compute_canvas`` function: a constant-time,
        host-only computation that reports the canvas size, its origin in
        image-1 coordinates, and where the four corners of each input land
        on the canvas.  The values are exactly those the warp uses.

        Parameters
        ----------
        homography : numpy.ndarray
            A ``(3, 3)`` homography matrix (image 1 to image 2), as returned
            by :meth:`estimate_homography`.
        width1, height1 : int
            Size of image 1 in pixels.
        width2, height2 : int
            Size of image 2 in pixels.

        Returns
        -------
        Canvas

        Raises
        ------
        CuSiftError
            If the homography is non-finite or singular, a dimension is not
            positive, or the canvas would exceed twice the larger input in
            either axis (the same limit the warp enforces).  For the
            canvas-size case the rejected geometry is attached to the
            exception as ``err.geometry``.
        """
        H_flat = np.ascontiguousarray(np.asarray(homography, dtype=np.float32).ravel())
        if H_flat.size != 9:
            raise ValueError(
                f"homography must have 9 elements (3x3), got shape {np.shape(homography)}"
            )
        h_ct = H_flat.ctypes.data_as(POINTER(c_float))
        geom_ct = cusift_canvas()

        self._lib.cusift_compute_canvas(
            h_ct,
            c_int(int(width1)), c_int(int(height1)),
            c_int(int(width2)), c_int(int(height2)),
            byref(geom_ct),
        )
        try:
            _check_error(self._lib)
        except CuSiftError as err:
            err.geometry = Canvas._from_ctypes(geom_ct)  # type: ignore[attr-defined]
            raise
        return Canvas._from_ctypes(geom_ct)

    # -- Device memory estimates ------------------------------------------
    #
    # Host-only and allocation-free: each returns the peak device memory, in
    # bytes, a stage would need, so a caller can check it against the free
    # memory of a small board before running anything.  The C functions do
    # not validate their arguments, so these methods do (ValueError).

    def memory_extract(
        self,
        width: int,
        height: int,
        options: Optional[ExtractOptions] = None,
    ) -> int:
        """Peak device memory, in bytes, that :meth:`extract` needs for an image
        of this size (C ``cusift_memory_extract``): keypoint buffer, image copy,
        scale-space pyramid, sort temporaries and context.
        """
        w, h = _check_dims("image", width, height)
        opts_ct = (options or ExtractOptions())._to_ctypes()
        return int(self._lib.cusift_memory_extract(w, h, byref(opts_ct)))

    def memory_match(self, capacity1: int, capacity2: int) -> int:
        """Device memory, in bytes, held by two keypoint sets of these
        capacities during :meth:`match` (C ``cusift_memory_match``).

        The capacity of a set is the ``max_keypoints`` it was extracted with.
        """
        c1 = _check_capacity("capacity1", capacity1)
        c2 = _check_capacity("capacity2", capacity2)
        return int(self._lib.cusift_memory_match(c1, c2))

    def memory_homography(
        self,
        capacity: int,
        options: Optional[HomographyOptions] = None,
    ) -> int:
        """Device memory, in bytes, that :meth:`estimate_homography` needs for a
        keypoint set of this capacity: the set plus the RANSAC temporaries
        (C ``cusift_memory_homography``).
        """
        c = _check_capacity("capacity", capacity)
        opts_ct = (options or HomographyOptions())._to_ctypes()
        return int(self._lib.cusift_memory_homography(c, byref(opts_ct)))

    def memory_warp(self, width1: int, height1: int, width2: int, height2: int) -> int:
        """Peak device memory, in bytes, of a GPU :meth:`warp` (C ``cusift_memory_warp``).

        Assumes the worst-case canvas of twice the larger input in each axis;
        :meth:`compute_canvas` gives the exact canvas once the homography is known.
        """
        w1, h1 = _check_dims("image1", width1, height1)
        w2, h2 = _check_dims("image2", width2, height2)
        return int(self._lib.cusift_memory_warp(w1, h1, w2, h2))

    def memory_pipeline(
        self,
        width1: int,
        height1: int,
        width2: int,
        height2: int,
        *,
        extract_options: Optional[ExtractOptions] = None,
        homography_options: Optional[HomographyOptions] = None,
    ) -> int:
        """Peak device memory, in bytes, across :meth:`align_and_warp`: the
        maximum over its stages (C ``cusift_memory_pipeline``).
        """
        w1, h1 = _check_dims("image1", width1, height1)
        w2, h2 = _check_dims("image2", width2, height2)
        eo = (extract_options or ExtractOptions())._to_ctypes()
        ho = (homography_options or HomographyOptions())._to_ctypes()
        return int(self._lib.cusift_memory_pipeline(w1, h1, w2, h2, byref(eo), byref(ho)))

    # -- Convenience combo pipelines --------------------------------------

    def extract_and_match(
        self,
        image1: Union[str, Path, np.ndarray],
        image2: Union[str, Path, np.ndarray],
        *,
        width1: Optional[int] = None,
        height1: Optional[int] = None,
        width2: Optional[int] = None,
        height2: Optional[int] = None,
        options: Optional[ExtractOptions] = None,
    ) -> tuple[Keypoints, Keypoints, List[MatchResult]]:
        """Extract SIFT features from two images and match them in one call.

        This is a convenience wrapper around the C ``cusift_extract_and_match``
        function, which fuses :meth:`extract` and :meth:`match` into a
        single GPU-accelerated pipeline call, avoiding an extra
        host ↔ device round-trip.

        Parameters
        ----------
        image1 : str | Path | numpy.ndarray
            First image (file path or 2-D ``float32`` array).
        image2 : str | Path | numpy.ndarray
            Second image (file path or 2-D ``float32`` array).
        width1, height1 : int, optional
            Dimension overrides for *image1* (only needed for 1-D arrays).
        width2, height2 : int, optional
            Dimension overrides for *image2* (only needed for 1-D arrays).
        options : ExtractOptions, optional
            Extraction parameters applied to *both* images.  Uses
            :class:`ExtractOptions` defaults when not provided.

        Returns
        -------
        (kp1, kp2, matches) : tuple[Keypoints, Keypoints, list[MatchResult]]
            *kp1* and *kp2* are the keypoints extracted from each image.
            *matches* contains one :class:`MatchResult` per successfully
            matched correspondence (unmatched keypoints are omitted).
            Both ``Keypoints`` objects own their underlying
            ``cusift_keypoints``; call ``.free()`` or use them as context managers
            to release GPU memory early.

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
        sift_data1 = cusift_keypoints()
        sift_data2 = cusift_keypoints()
        opts_ct = (options or ExtractOptions())._to_ctypes()

        # -- Call the C function ------------------------------------------
        self._lib.cusift_extract_and_match(
            byref(img1_ct),
            byref(img2_ct),
            byref(sift_data1),
            byref(sift_data2),
            byref(opts_ct),
        )
        kp1 = None
        kp2 = None
        try:
            _check_error(self._lib)

            # -- Convert keypoints ----------------------------------------
            kps1: List[Keypoint] = []
            for i in range(sift_data1.count):
                kps1.append(Keypoint._from_sift_point(sift_data1.host[i]))
            kp1 = Keypoints(kps1, sift_data1, self._lib)

            kps2: List[Keypoint] = []
            for i in range(sift_data2.count):
                kps2.append(Keypoint._from_sift_point(sift_data2.host[i]))
            kp2 = Keypoints(kps2, sift_data2, self._lib)

            # -- Read back match results from sift_data1 ------------------
            matches: List[MatchResult] = []
            for i in range(sift_data1.count):
                pt = sift_data1.host[i]
                if pt.match >= 0:
                    matches.append(
                        MatchResult(
                            query_index=i,
                            match_index=pt.match,
                            x1=pt.x,
                            y1=pt.y,
                            x2=pt.match_x,
                            y2=pt.match_y,
                            error=pt.match_error,
                            score=pt.score,
                            ambiguity=pt.ambiguity,
                        )
                    )

            return kp1, kp2, matches
        except:
            if kp1 is not None:
                kp1.free()
            else:
                self._lib.cusift_keypoints_free(byref(sift_data1))
            if kp2 is not None:
                kp2.free()
            else:
                self._lib.cusift_keypoints_free(byref(sift_data2))
            raise

    def _collect_pair(self, sift_data1, sift_data2):
        """Wrap two filled keypoint sets; free them if anything goes wrong."""
        kp1 = kp2 = None
        try:
            _check_error(self._lib)
            kp1 = Keypoints([Keypoint._from_sift_point(sift_data1.host[i]) for i in range(sift_data1.count)], sift_data1, self._lib)
            kp2 = Keypoints([Keypoint._from_sift_point(sift_data2.host[i]) for i in range(sift_data2.count)], sift_data2, self._lib)
            matches: List[MatchResult] = []
            for i in range(sift_data1.count):
                pt = sift_data1.host[i]
                if pt.match >= 0:
                    matches.append(MatchResult(query_index=i, match_index=pt.match, x1=pt.x, y1=pt.y,
                                               x2=pt.match_x, y2=pt.match_y, error=pt.match_error,
                                               score=pt.score, ambiguity=pt.ambiguity))
            return kp1, kp2, matches
        except BaseException:
            if kp1 is not None:
                kp1.free()
            else:
                self._lib.cusift_keypoints_free(byref(sift_data1))
            if kp2 is not None:
                kp2.free()
            else:
                self._lib.cusift_keypoints_free(byref(sift_data2))
            raise

    def align(
        self,
        image1: Union[str, Path, np.ndarray],
        image2: Union[str, Path, np.ndarray],
        *,
        width1: Optional[int] = None,
        height1: Optional[int] = None,
        width2: Optional[int] = None,
        height2: Optional[int] = None,
        extract_options: Optional[ExtractOptions] = None,
        homography_options: Optional[HomographyOptions] = None,
    ) -> tuple[Keypoints, Keypoints, List[MatchResult], np.ndarray, int]:
        """Extract, match and estimate the homography from image 1 to image 2.

        Calls the C ``cusift_align`` function.  Several attempts and the
        selection goal are set through ``homography_options`` (``attempts``,
        ``goal``).

        Returns
        -------
        (kp1, kp2, matches, homography, num_inliers)
            The keypoints of each image (free them with ``.free()``), the
            match list, the ``(3, 3)`` float32 homography and the RANSAC
            inlier count.

        Raises
        ------
        CuSiftError
            If the underlying C library reports an error.
        """
        pix1, w1, h1 = _resolve_image_arg(image1, width1, height1, "image1")
        pix2, w2, h2 = _resolve_image_arg(image2, width2, height2, "image2")
        img1_ct, _pix1_ref = _make_image_t(pix1, w1, h1)
        img2_ct, _pix2_ref = _make_image_t(pix2, w2, h2)
        sift_data1 = cusift_keypoints()
        sift_data2 = cusift_keypoints()
        ext_ct = (extract_options or ExtractOptions())._to_ctypes()
        hom_ct = (homography_options or HomographyOptions())._to_ctypes()
        homography = (c_float * 9)()
        inliers = c_int(0)

        self._lib.cusift_align(
            byref(img1_ct), byref(img2_ct), byref(sift_data1), byref(sift_data2),
            byref(ext_ct), byref(hom_ct), homography, byref(inliers),
        )
        kp1, kp2, matches = self._collect_pair(sift_data1, sift_data2)
        H = np.ctypeslib.as_array(homography, shape=(9,)).copy().reshape(3, 3)
        return kp1, kp2, matches, H, inliers.value

    def align_and_warp(
        self,
        image1: Union[str, Path, np.ndarray],
        image2: Union[str, Path, np.ndarray],
        *,
        width1: Optional[int] = None,
        height1: Optional[int] = None,
        width2: Optional[int] = None,
        height2: Optional[int] = None,
        extract_options: Optional[ExtractOptions] = None,
        homography_options: Optional[HomographyOptions] = None,
    ) -> tuple[Keypoints, Keypoints, List[MatchResult], np.ndarray, int, np.ndarray, np.ndarray]:
        """:meth:`align` followed by the GPU warp of both images onto the common canvas.

        Calls the C ``cusift_align_and_warp`` function; the images uploaded
        for extraction are reused as warp sources.

        Returns
        -------
        (kp1, kp2, matches, homography, num_inliers, warped1, warped2)
            As :meth:`align`, plus the two ``float32`` canvases of equal shape
            (NaN where no source covers a pixel).  The caller owns the arrays.

        Raises
        ------
        CuSiftError
            If the underlying C library reports an error.
        """
        pix1, w1, h1 = _resolve_image_arg(image1, width1, height1, "image1")
        pix2, w2, h2 = _resolve_image_arg(image2, width2, height2, "image2")
        img1_ct, _pix1_ref = _make_image_t(pix1, w1, h1)
        img2_ct, _pix2_ref = _make_image_t(pix2, w2, h2)
        sift_data1 = cusift_keypoints()
        sift_data2 = cusift_keypoints()
        ext_ct = (extract_options or ExtractOptions())._to_ctypes()
        hom_ct = (homography_options or HomographyOptions())._to_ctypes()
        homography = (c_float * 9)()
        inliers = c_int(0)
        warped1_ct = cusift_image()
        warped2_ct = cusift_image()

        self._lib.cusift_align_and_warp(
            byref(img1_ct), byref(img2_ct), byref(sift_data1), byref(sift_data2),
            byref(ext_ct), byref(hom_ct), homography, byref(inliers),
            byref(warped1_ct), byref(warped2_ct),
        )
        try:
            kp1, kp2, matches = self._collect_pair(sift_data1, sift_data2)
            H = np.ctypeslib.as_array(homography, shape=(9,)).copy().reshape(3, 3)
            n1 = warped1_ct.width * warped1_ct.height
            n2 = warped2_ct.width * warped2_ct.height
            warped1 = np.ctypeslib.as_array(warped1_ct.pixels, shape=(n1,)).copy().reshape(warped1_ct.height, warped1_ct.width)
            warped2 = np.ctypeslib.as_array(warped2_ct.pixels, shape=(n2,)).copy().reshape(warped2_ct.height, warped2_ct.width)
            return kp1, kp2, matches, H, inliers.value, warped1, warped2
        finally:
            self._lib.cusift_image_free(byref(warped1_ct))
            self._lib.cusift_image_free(byref(warped2_ct))

    # -- Visualisation ----------------------------------------------------

    @staticmethod
    def draw_keypoints(
        image: Union[str, Path, np.ndarray],
        keypoints: Keypoints,
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
        keypoints : Keypoints
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
    def draw_matches(
        image1: Union[str, Path],
        image2: Union[str, Path],
        matches: List[MatchResult],
        output: Union[str, Path],
        *,
        line_width: int = 1,
        point_radius: int = 3,
    ) -> None:
        """Draw matched keypoint pairs on a side-by-side canvas and save.

        Each correspondence is drawn as a coloured line connecting the
        keypoint positions in the two images.  Colours cycle through an
        HSV hue ramp so individual matches are visually distinct.

        Parameters
        ----------
        image1 : str | Path
            Path to the first (left) image.
        image2 : str | Path
            Path to the second (right) image.
        matches : list[MatchResult]
            Match correspondences (returned by :meth:`match` or one of
            the combo pipeline methods).
        output : str | Path
            File path for the annotated image (e.g. ``"matches.png"``).
        line_width : int
            Stroke width for the connecting lines (default 1).
        point_radius : int
            Radius of the keypoint dots (default 3).
        """
        import colorsys

        try:
            from PIL import Image, ImageDraw
        except ImportError as exc:
            raise ImportError(
                "Pillow is required for draw_matches.  "
                "Install it with:  pip install Pillow"
            ) from exc

        img1 = Image.open(image1).convert("RGB")
        img2 = Image.open(image2).convert("RGB")

        w1, h1 = img1.size
        w2, h2 = img2.size
        canvas_w = w1 + w2
        canvas_h = max(h1, h2)

        canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))
        canvas.paste(img1, (0, 0))
        canvas.paste(img2, (w1, 0))

        draw = ImageDraw.Draw(canvas)

        n = len(matches)
        for idx, m in enumerate(matches):
            hue = idx / max(n, 1)
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            color = (int(r * 255), int(g * 255), int(b * 255))

            x1, y1 = m.x1, m.y1
            x2, y2 = m.x2 + w1, m.y2  # offset into the right panel
            draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)
            draw.ellipse(
                [x1 - point_radius, y1 - point_radius,
                 x1 + point_radius, y1 + point_radius],
                fill=color,
            )
            draw.ellipse(
                [x2 - point_radius, y2 - point_radius,
                 x2 + point_radius, y2 + point_radius],
                fill=color,
            )

        canvas.save(str(output))

    @staticmethod
    def draw_descriptors(
        image: Union[str, Path, np.ndarray],
        keypoints: Keypoints,
        min_sampling: float,
        output_dir: Union[str, Path],
        *,
        patch_size: int = 256,
        grid_size: int = 256,
        spoke_color: str = "cyan",
        bg_color: str = "black",
        grid_line_color: str = "#444444",
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        """Visualise SIFT descriptors for keypoints above a sampling threshold.

        For each qualifying keypoint, produces a side-by-side image: the
        left half shows the image patch centred on the keypoint (with a
        crosshair and circle), and the right half shows the 4×4 grid of
        8-bin orientation histograms (rose/polar plots) that make up the
        descriptor.

        Images are saved into subdirectories of *output_dir* organised by
        subsampling value::

            output_dir/
                sub_1.0/
                    desc_42_x310_y205.png
                sub_2.0/
                    ...

        Parameters
        ----------
        image : str | Path | numpy.ndarray
            Source image (file path or 2-D float32 array).
        keypoints : Keypoints
            Keypoints returned by :meth:`extract`.
        min_sampling : float
            Minimum subsampling value; keypoints below this are skipped.
        output_dir : str | Path
            Root directory for saving descriptor images.  Subdirectories
            named ``sub_<value>`` are created automatically.
        patch_size : int
            Size in pixels of the image-patch panel (default 256).
        grid_size : int
            Size in pixels of the histogram-grid panel (default 256).
        spoke_color : str
            Colour for histogram spoke lines (default ``"cyan"``).
        bg_color : str
            Background colour for the histogram panel (default ``"black"``).
        grid_line_color : str
            Colour for the 4×4 grid lines (default ``"#444444"``).
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

        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        # Source image is decoded once, lazily, on the first qualifying keypoint
        # and then reused for every patch (cropping does not mutate it). This
        # avoids re-decoding the whole image on every loop iteration.
        src = None

        for index, kp in enumerate(keypoints):
            if kp.subsampling < min_sampling:
                continue  # skip keypoints below the sampling threshold
            kp = keypoints[index]
            desc = kp.descriptor  # shape (128,)

            # -- Load the source image (once) ---------------------------------
            if src is None:
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
            sub_dir = output_root / f"sub_{kp.subsampling:.1f}"
            sub_dir.mkdir(parents=True, exist_ok=True)

            filename = f"desc_{index}_x{int(kp.x)}_y{int(kp.y)}.png"
            combined.save(sub_dir / filename)
    