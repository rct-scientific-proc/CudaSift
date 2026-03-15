

import json
import math
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QWheelEvent
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from cusift.cusift import CuSift, ExtractOptions, Keypoint

# -- Types --------------------------------------------------------------------

@dataclass
class DetectedKeypoint:
    """Detector-agnostic keypoint."""
    x: float
    y: float
    scale: float

PointList = List[Dict[str, float]]  # [{"x": ..., "y": ...}, ...]
PerImageResults = Dict[str, dict]   # image_name -> {tp, fp, fn, keypoints}

MATCH_RADIUS = 10.0  # pixels – maximum distance for a keypoint to "match" a ground-truth point


# -- Helpers ------------------------------------------------------------------

def _load_ground_truth(path: str) -> Dict[str, PointList]:
    """Load a JSON file mapping image filenames to lists of {x, y} coordinates."""
    with open(path, "r") as f:
        data = json.load(f)
    # Expected format: {"image1.png": [{"x": ..., "y": ...}, ...], ...}
    return data


def _match_files(gt_names: List[str], image_dir: str) -> Dict[str, Path]:
    """Match ground-truth filenames to actual files in *image_dir*."""
    dir_path = Path(image_dir)
    available = {p.name: p for p in dir_path.iterdir() if p.is_file()}
    matched: Dict[str, Path] = {}
    for name in gt_names:
        if name in available:
            matched[name] = available[name]
        else:
            stem = Path(name).stem
            for avail_name, avail_path in available.items():
                if Path(avail_name).stem == stem:
                    matched[name] = avail_path
                    break
    return matched


def _classify_keypoints(
    keypoints: List[DetectedKeypoint],
    gt_points: PointList,
    radius: float = MATCH_RADIUS,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Classify keypoints into true positives, false positives, false negatives.

    Returns (true_positives, false_positives, false_negatives).
    Each entry is a dict with at least 'x' and 'y'.
    """
    gt_matched = [False] * len(gt_points)
    tp: List[dict] = []
    fp: List[dict] = []

    for kp in keypoints:
        best_dist = float("inf")
        best_idx = -1
        for i, gt in enumerate(gt_points):
            d = math.hypot(kp.x - gt["x"], kp.y - gt["y"])
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_dist <= radius and not gt_matched[best_idx]:
            gt_matched[best_idx] = True
            tp.append({"x": kp.x, "y": kp.y, "gt_x": gt_points[best_idx]["x"], "gt_y": gt_points[best_idx]["y"]})
        else:
            fp.append({"x": kp.x, "y": kp.y})

    fn = [{"x": gt["x"], "y": gt["y"]} for i, gt in enumerate(gt_points) if not gt_matched[i]]
    return tp, fp, fn


def _numpy_to_qpixmap(arr: np.ndarray) -> QPixmap:
    """Convert a grayscale or RGB numpy array to QPixmap."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        h, w = arr.shape
        qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
    else:
        h, w, ch = arr.shape
        if ch == 3:
            qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
        else:
            qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
    return QPixmap.fromImage(qimg.copy())


# -- Detector abstraction -------------------------------------------------------

class Detector(ABC):
    """Base class for keypoint detectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable detector name shown in the combo box."""

    @abstractmethod
    def build_param_widget(self) -> QWidget:
        """Return a QWidget containing detector-specific parameter controls."""

    @abstractmethod
    def extract(self, image: np.ndarray) -> List[DetectedKeypoint]:
        """Detect keypoints in a grayscale float32 image (H×W, [0-255])."""

    @abstractmethod
    def get_params(self) -> dict:
        """Return current parameters as a JSON-serialisable dict."""


# -- CuSIFT detector -----------------------------------------------------------

class CuSiftDetector(Detector):
    name = "CuSIFT"

    def __init__(self):
        self._sift: Optional[CuSift] = None
        self._widget: Optional[QWidget] = None
        # Spin boxes stored after build
        self._spins: Dict[str, QDoubleSpinBox] = {}
        self._spin_ints: Dict[str, QSpinBox] = {}

    def build_param_widget(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        def _dbl(name, lo, hi, dec, val):
            s = QDoubleSpinBox(); s.setRange(lo, hi); s.setDecimals(dec); s.setValue(val)
            self._spins[name] = s; form.addRow(f"{name}:", s)

        def _int(name, lo, hi, val):
            s = QSpinBox(); s.setRange(lo, hi); s.setValue(val)
            self._spin_ints[name] = s; form.addRow(f"{name}:", s)

        _dbl("Threshold", 0, 100, 2, 3.0)
        _dbl("Lowest Scale", 0, 100, 2, 0.0)
        _dbl("Highest Scale", 0, 100000, 2, 99999.0)
        _dbl("Edge Threshold", 0, 100, 2, 10.0)
        _dbl("Init Blur", 0, 10, 2, 1.0)
        _int("Max Keypoints", 1, 131072, 32768)
        _int("Num Octaves", 1, 10, 5)
        _dbl("Scale Suppression Radius", 0, 50, 2, 0.0)

        self._widget = w
        return w

    def extract(self, image: np.ndarray) -> List[DetectedKeypoint]:
        if self._sift is None:
            self._sift = CuSift()
        opts = ExtractOptions(
            thresh=self._spins["Threshold"].value(),
            lowest_scale=self._spins["Lowest Scale"].value(),
            highest_scale=self._spins["Highest Scale"].value(),
            edge_thresh=self._spins["Edge Threshold"].value(),
            init_blur=self._spins["Init Blur"].value(),
            max_keypoints=self._spin_ints["Max Keypoints"].value(),
            num_octaves=self._spin_ints["Num Octaves"].value(),
            scale_suppression_radius=self._spins["Scale Suppression Radius"].value(),
        )
        kp_list = self._sift.extract(image, options=opts)
        return [DetectedKeypoint(x=k.x, y=k.y, scale=k.scale) for k in kp_list]

    def get_params(self) -> dict:
        return {
            "threshold": self._spins["Threshold"].value(),
            "lowest_scale": self._spins["Lowest Scale"].value(),
            "highest_scale": self._spins["Highest Scale"].value(),
            "edge_threshold": self._spins["Edge Threshold"].value(),
            "init_blur": self._spins["Init Blur"].value(),
            "max_keypoints": self._spin_ints["Max Keypoints"].value(),
            "num_octaves": self._spin_ints["Num Octaves"].value(),
            "scale_suppression_radius": self._spins["Scale Suppression Radius"].value(),
        }


# -- OpenCV detectors ----------------------------------------------------------

class _OpenCVDetector(Detector):
    """Base for OpenCV Feature2D-based detectors."""

    def __init__(self):
        self._widget: Optional[QWidget] = None

    @abstractmethod
    def _create_feature2d(self):
        """Return a cv2.Feature2D instance with current params."""

    def extract(self, image: np.ndarray) -> List[DetectedKeypoint]:
        import cv2
        if image.dtype != np.uint8:
            img_u8 = np.clip(image, 0, 255).astype(np.uint8)
        else:
            img_u8 = image
        detector = self._create_feature2d()
        kps = detector.detect(img_u8, None)
        return [DetectedKeypoint(x=k.pt[0], y=k.pt[1], scale=k.size) for k in kps]


class OrbDetector(_OpenCVDetector):
    name = "ORB"

    def __init__(self):
        super().__init__()
        self._spins: Dict[str, QSpinBox] = {}
        self._dspins: Dict[str, QDoubleSpinBox] = {}

    def build_param_widget(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        s = QSpinBox(); s.setRange(1, 100000); s.setValue(500)
        self._spins["nFeatures"] = s; form.addRow("Max Features:", s)

        s = QDoubleSpinBox(); s.setRange(1.0, 3.0); s.setDecimals(2); s.setValue(1.2)
        self._dspins["scaleFactor"] = s; form.addRow("Scale Factor:", s)

        s = QSpinBox(); s.setRange(1, 20); s.setValue(8)
        self._spins["nLevels"] = s; form.addRow("Num Levels:", s)

        s = QSpinBox(); s.setRange(0, 100); s.setValue(31)
        self._spins["edgeThreshold"] = s; form.addRow("Edge Threshold:", s)

        s = QSpinBox(); s.setRange(2, 100); s.setValue(31)
        self._spins["patchSize"] = s; form.addRow("Patch Size:", s)

        self._widget = w
        return w

    def _create_feature2d(self):
        import cv2
        return cv2.ORB_create(
            nfeatures=self._spins["nFeatures"].value(),
            scaleFactor=self._dspins["scaleFactor"].value(),
            nlevels=self._spins["nLevels"].value(),
            edgeThreshold=self._spins["edgeThreshold"].value(),
            patchSize=self._spins["patchSize"].value(),
        )

    def get_params(self) -> dict:
        return {k: s.value() for k, s in {**self._spins, **self._dspins}.items()}


class AkazeDetector(_OpenCVDetector):
    name = "AKAZE"

    def __init__(self):
        super().__init__()
        self._dspins: Dict[str, QDoubleSpinBox] = {}
        self._spins: Dict[str, QSpinBox] = {}

    def build_param_widget(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        s = QDoubleSpinBox(); s.setRange(0.0001, 1.0); s.setDecimals(4); s.setValue(0.001)
        self._dspins["threshold"] = s; form.addRow("Threshold:", s)

        s = QSpinBox(); s.setRange(1, 10); s.setValue(4)
        self._spins["nOctaves"] = s; form.addRow("Num Octaves:", s)

        s = QSpinBox(); s.setRange(1, 10); s.setValue(4)
        self._spins["nOctaveLayers"] = s; form.addRow("Octave Layers:", s)

        self._widget = w
        return w

    def _create_feature2d(self):
        import cv2
        return cv2.AKAZE_create(
            threshold=self._dspins["threshold"].value(),
            nOctaves=self._spins["nOctaves"].value(),
            nOctaveLayers=self._spins["nOctaveLayers"].value(),
        )

    def get_params(self) -> dict:
        return {k: s.value() for k, s in {**self._dspins, **self._spins}.items()}


class BriskDetector(_OpenCVDetector):
    name = "BRISK"

    def __init__(self):
        super().__init__()
        self._spins: Dict[str, QSpinBox] = {}
        self._dspins: Dict[str, QDoubleSpinBox] = {}

    def build_param_widget(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        s = QSpinBox(); s.setRange(1, 200); s.setValue(30)
        self._spins["thresh"] = s; form.addRow("Threshold:", s)

        s = QSpinBox(); s.setRange(0, 10); s.setValue(3)
        self._spins["octaves"] = s; form.addRow("Octaves:", s)

        s = QDoubleSpinBox(); s.setRange(0.0, 10.0); s.setDecimals(1); s.setValue(1.0)
        self._dspins["patternScale"] = s; form.addRow("Pattern Scale:", s)

        self._widget = w
        return w

    def _create_feature2d(self):
        import cv2
        return cv2.BRISK_create(
            thresh=self._spins["thresh"].value(),
            octaves=self._spins["octaves"].value(),
            patternScale=self._dspins["patternScale"].value(),
        )

    def get_params(self) -> dict:
        return {k: s.value() for k, s in {**self._spins, **self._dspins}.items()}


class KazeDetector(_OpenCVDetector):
    name = "KAZE"

    def __init__(self):
        super().__init__()
        self._dspins: Dict[str, QDoubleSpinBox] = {}
        self._spins: Dict[str, QSpinBox] = {}

    def build_param_widget(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        s = QDoubleSpinBox(); s.setRange(0.0001, 1.0); s.setDecimals(4); s.setValue(0.001)
        self._dspins["threshold"] = s; form.addRow("Threshold:", s)

        s = QSpinBox(); s.setRange(1, 10); s.setValue(4)
        self._spins["nOctaves"] = s; form.addRow("Num Octaves:", s)

        s = QSpinBox(); s.setRange(1, 10); s.setValue(4)
        self._spins["nOctaveLayers"] = s; form.addRow("Octave Layers:", s)

        self._widget = w
        return w

    def _create_feature2d(self):
        import cv2
        return cv2.KAZE_create(
            threshold=self._dspins["threshold"].value(),
            nOctaves=self._spins["nOctaves"].value(),
            nOctaveLayers=self._spins["nOctaveLayers"].value(),
        )

    def get_params(self) -> dict:
        return {k: s.value() for k, s in {**self._dspins, **self._spins}.items()}


# -- Zoomable Image Widget -----------------------------------------------------

class ZoomableImageWidget(QWidget):
    """Widget that displays a QPixmap with mouse-wheel zoom and drag-to-pan."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._zoom: float = 1.0
        self._offset: QPointF = QPointF(0, 0)  # pan offset in widget coords
        self._dragging: bool = False
        self._drag_start: QPointF = QPointF()
        self._offset_start: QPointF = QPointF()
        self.setMouseTracking(True)
        self._placeholder_text = "Load a JSON file and image directory to begin."

    def set_pixmap(self, pixmap: QPixmap, fit: bool = True) -> None:
        self._pixmap = pixmap
        if fit:
            self.fit_to_view()
        else:
            self.update()

    def set_placeholder(self, text: str) -> None:
        self._placeholder_text = text
        self._pixmap = None
        self.update()

    def fit_to_view(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            return
        vw, vh = self.width(), self.height()
        pw, ph = self._pixmap.width(), self._pixmap.height()
        if pw == 0 or ph == 0:
            return
        self._zoom = min(vw / pw, vh / ph)
        # Center the image
        self._offset = QPointF(
            (vw - pw * self._zoom) / 2,
            (vh - ph * self._zoom) / 2,
        )
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        if self._pixmap is None or self._pixmap.isNull():
            painter.drawText(self.rect(), Qt.AlignCenter, self._placeholder_text)
            painter.end()
            return
        painter.translate(self._offset)
        painter.scale(self._zoom, self._zoom)
        painter.drawPixmap(0, 0, self._pixmap)
        painter.end()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._pixmap is None:
            return
        old_zoom = self._zoom
        factor = 1.25 if event.angleDelta().y() > 0 else 1 / 1.25
        self._zoom = max(0.01, min(self._zoom * factor, 100.0))
        # Zoom towards the mouse cursor position
        mouse_pos = QPointF(event.pos())
        self._offset = mouse_pos - (self._zoom / old_zoom) * (mouse_pos - self._offset)
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._pixmap is not None:
            self._dragging = True
            self._drag_start = QPointF(event.pos())
            self._offset_start = QPointF(self._offset)
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging:
            delta = QPointF(event.pos()) - self._drag_start
            self._offset = self._offset_start + delta
            self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._dragging = False
            self.setCursor(Qt.ArrowCursor)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Re-fit when the widget is first shown or resized with an image loaded
        if self._pixmap is not None:
            self.fit_to_view()


# -- Main Window --------------------------------------------------------------

class SiftEvalApp(QWidget):
    # All available detectors — add new ones here
    DETECTORS: List[type] = [CuSiftDetector, OrbDetector, AkazeDetector, BriskDetector, KazeDetector]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Keypoint Detector Evaluator")
        self.resize(1400, 900)

        # State
        self._gt_data: Dict[str, PointList] = {}
        self._image_paths: Dict[str, Path] = {}       # name -> file path
        self._results: PerImageResults = {}
        self._current_image_name: Optional[str] = None

        # Detector instances (created once, kept alive for the session)
        self._detectors: List[Detector] = [cls() for cls in self.DETECTORS]

        self._build_ui()

    @property
    def _active_detector(self) -> Detector:
        return self._detectors[self._combo_detector.currentIndex()]

    # -- UI construction -------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)

        # Toolbar
        toolbar = QHBoxLayout()
        self._btn_load_json = QPushButton("Load JSON")
        self._btn_load_json.clicked.connect(self._on_load_json)
        self._btn_load_images = QPushButton("Select Image Directory")
        self._btn_load_images.clicked.connect(self._on_load_images)
        self._btn_load_images.setEnabled(False)

        self._combo_detector = QComboBox()
        for det in self._detectors:
            self._combo_detector.addItem(det.name)
        self._combo_detector.currentIndexChanged.connect(self._on_detector_changed)

        toolbar.addWidget(self._btn_load_json)
        toolbar.addWidget(self._btn_load_images)
        toolbar.addWidget(QLabel("Detector:"))
        toolbar.addWidget(self._combo_detector)
        toolbar.addStretch()
        root.addLayout(toolbar)

        # Main splitter: list | image | params
        splitter = QSplitter(Qt.Horizontal)

        # -- Left: image list --
        self._list_widget = QListWidget()
        self._list_widget.currentTextChanged.connect(self._on_image_selected)
        splitter.addWidget(self._list_widget)

        # -- Center: image display --
        self._image_view = ZoomableImageWidget()
        splitter.addWidget(self._image_view)

        # -- Right: detector parameters + evaluation --
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Stacked widget for detector-specific parameter panels
        param_group = QGroupBox("Detector Options")
        param_inner = QVBoxLayout(param_group)
        self._param_stack = QStackedWidget()
        for det in self._detectors:
            self._param_stack.addWidget(det.build_param_widget())
        param_inner.addWidget(self._param_stack)
        right_layout.addWidget(param_group)

        # Match radius
        radius_group = QGroupBox("Evaluation")
        radius_form = QFormLayout()
        self._spin_radius = QDoubleSpinBox()
        self._spin_radius.setRange(1.0, 200.0)
        self._spin_radius.setDecimals(1)
        self._spin_radius.setValue(MATCH_RADIUS)
        radius_form.addRow("Match Radius (px):", self._spin_radius)
        self._spin_beta = QDoubleSpinBox()
        self._spin_beta.setRange(0.01, 100.0)
        self._spin_beta.setDecimals(2)
        self._spin_beta.setValue(1.0)
        radius_form.addRow("F-beta (β):", self._spin_beta)
        radius_group.setLayout(radius_form)
        right_layout.addWidget(radius_group)

        # Run button
        self._btn_run = QPushButton("Run Detector")
        self._btn_run.setEnabled(False)
        self._btn_run.clicked.connect(self._on_run_detector)
        right_layout.addWidget(self._btn_run)

        # Save button
        self._btn_save = QPushButton("Save Results")
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._on_save_results)
        right_layout.addWidget(self._btn_save)

        # -- Metrics (in the right panel, below buttons) --
        metrics_group = QGroupBox("Metrics")
        metrics_form = QFormLayout()
        self._lbl_precision = QLabel("—")
        self._lbl_recall = QLabel("—")
        self._lbl_fbeta = QLabel("—")
        self._lbl_tp = QLabel("—")
        self._lbl_fp = QLabel("—")
        self._lbl_fn = QLabel("—")
        self._lbl_fp_per_img = QLabel("—")
        metrics_form.addRow("Precision:", self._lbl_precision)
        metrics_form.addRow("Recall:", self._lbl_recall)
        metrics_form.addRow("F-beta:", self._lbl_fbeta)
        metrics_form.addRow("TP:", self._lbl_tp)
        metrics_form.addRow("FP:", self._lbl_fp)
        metrics_form.addRow("FN:", self._lbl_fn)
        metrics_form.addRow("FP / Image:", self._lbl_fp_per_img)
        metrics_group.setLayout(metrics_form)
        right_layout.addWidget(metrics_group)

        right_layout.addStretch()
        splitter.addWidget(right_panel)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 1)
        root.addWidget(splitter, stretch=1)

        # Status bar
        self._status = QStatusBar()
        self._status.setMaximumHeight(22)
        root.addWidget(self._status)

    # -- Slots -----------------------------------------------------------------

    def _on_detector_changed(self, index: int):
        self._param_stack.setCurrentIndex(index)
        self._status.showMessage(f"Detector: {self._active_detector.name}", 3000)

    def _on_load_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Ground-Truth JSON", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            self._gt_data = _load_ground_truth(path)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to load JSON:\n{exc}")
            return
        self._btn_load_images.setEnabled(True)
        self._status.showMessage(f"Loaded {len(self._gt_data)} images from JSON.", 5000)

    def _on_load_images(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if not dir_path:
            return
        self._image_paths = _match_files(list(self._gt_data.keys()), dir_path)
        self._list_widget.clear()
        for name in self._gt_data:
            self._list_widget.addItem(name)
        self._btn_run.setEnabled(True)
        matched = len(self._image_paths)
        total = len(self._gt_data)
        self._status.showMessage(f"Matched {matched}/{total} images from directory.", 5000)

    def _on_image_selected(self, name: str):
        if not name:
            return
        self._current_image_name = name
        self._display_image(name)

    def _on_run_detector(self):
        if not self._gt_data:
            return

        detector = self._active_detector
        radius = self._spin_radius.value()
        self._results.clear()

        # Build list of images to process
        work_items = [(name, gt) for name, gt in self._gt_data.items() if name in self._image_paths]
        total = len(work_items)

        # Set up progress dialog
        progress = QProgressDialog(f"Running {detector.name} extraction...", "Cancel", 0, total, self)
        progress.setWindowTitle(f"{detector.name} Progress")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        def _utc_stamp() -> str:
            return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        print(f"[{_utc_stamp()}]: Starting {detector.name} extraction on {total} images")

        total_tp = total_fp = total_fn = 0

        for idx, (name, gt_points) in enumerate(work_items):
            if progress.wasCanceled():
                print(f"[{_utc_stamp()}]: Cancelled by user after {idx}/{total} images")
                break

            progress.setLabelText(f"Processing {name} ({idx + 1}/{total})")
            progress.setValue(idx)
            QApplication.processEvents()

            print(f"[{_utc_stamp()}]: [{idx + 1}/{total}] Extracting: {name}")

            pil_img = Image.open(str(self._image_paths[name])).convert("L")
            img_arg = np.array(pil_img, dtype=np.float32)
            try:
                keypoints = detector.extract(img_arg)
            except Exception as exc:
                print(f"[{_utc_stamp()}]: ERROR on {name}: {exc}")
                self._status.showMessage(f"Error on {name}: {exc}")
                continue

            tp, fp, fn = _classify_keypoints(keypoints, gt_points, radius)
            self._results[name] = {
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "keypoints": [{"x": k.x, "y": k.y, "scale": k.scale} for k in keypoints],
            }
            total_tp += len(tp)
            total_fp += len(fp)
            total_fn += len(fn)

            print(f"[{_utc_stamp()}]: [{idx + 1}/{total}] {name}: {len(keypoints)} kps, TP={len(tp)} FP={len(fp)} FN={len(fn)}")

        progress.setValue(total)
        progress.close()

        print(f"[{_utc_stamp()}]: {detector.name} extraction complete — TP={total_tp} FP={total_fp} FN={total_fn}")

        # Update metrics
        num_images = sum(1 for n in self._gt_data if n in self._image_paths)
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        beta = self._spin_beta.value()
        beta2 = beta * beta
        if precision + recall > 0:
            fbeta = (1 + beta2) * precision * recall / (beta2 * precision + recall)
        else:
            fbeta = 0.0
        fp_per_img = total_fp / num_images if num_images > 0 else 0.0

        self._lbl_tp.setText(str(total_tp))
        self._lbl_fp.setText(str(total_fp))
        self._lbl_fn.setText(str(total_fn))
        self._lbl_precision.setText(f"{precision:.4f}")
        self._lbl_recall.setText(f"{recall:.4f}")
        self._lbl_fbeta.setText(f"{fbeta:.4f}")
        self._lbl_fp_per_img.setText(f"{fp_per_img:.2f}")

        self._btn_save.setEnabled(True)
        self._status.showMessage(f"{detector.name} extraction complete.", 5000)

        # Refresh current image view
        if self._current_image_name:
            self._display_image(self._current_image_name)

    def _display_image(self, name: str):
        """Display image with annotated ground-truth and SIFT keypoints."""
        if name in self._image_paths:
            pil_img = Image.open(str(self._image_paths[name])).convert("RGB")
            arr = np.array(pil_img)
        else:
            self._image_view.set_placeholder(f"No image file matched for '{name}'.")
            return

        pixmap = _numpy_to_qpixmap(arr)

        gt_points = self._gt_data.get(name, [])
        result = self._results.get(name)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        if result is None:
            # No SIFT run yet – just draw blue circles for ground-truth
            pen = QPen(QColor(0, 120, 255), 2)
            painter.setPen(pen)
            for pt in gt_points:
                painter.drawEllipse(int(pt["x"]) - 5, int(pt["y"]) - 5, 10, 10)
        else:
            # Draw true positives in green
            pen_tp = QPen(QColor(0, 220, 0), 2)
            painter.setPen(pen_tp)
            for pt in result["true_positives"]:
                painter.drawEllipse(int(pt["x"]) - 6, int(pt["y"]) - 6, 12, 12)

            # Draw false negatives in red (ground-truth points that weren't matched)
            pen_fn = QPen(QColor(255, 0, 0), 2)
            painter.setPen(pen_fn)
            for pt in result["false_negatives"]:
                painter.drawEllipse(int(pt["x"]) - 5, int(pt["y"]) - 5, 10, 10)

            # Draw ground-truth in blue (underneath, as reference)
            pen_gt = QPen(QColor(0, 120, 255), 1)
            painter.setPen(pen_gt)
            for pt in gt_points:
                painter.drawEllipse(int(pt["x"]) - 4, int(pt["y"]) - 4, 8, 8)

        painter.end()

        self._image_view.set_pixmap(pixmap)

    def _on_save_results(self):
        if not self._results:
            QMessageBox.information(self, "Nothing to save", "Run a detector first.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save Results", "detector_results.json", "JSON Files (*.json)")
        if not path:
            return

        detector = self._active_detector

        output = {
            "detector": detector.name,
            "detector_parameters": detector.get_params(),
            "match_radius": self._spin_radius.value(),
            "images": {},
        }

        total_tp = total_fp = total_fn = 0
        for name, res in self._results.items():
            n_tp = len(res["true_positives"])
            n_fp = len(res["false_positives"])
            n_fn = len(res["false_negatives"])
            total_tp += n_tp
            total_fp += n_fp
            total_fn += n_fn
            output["images"][name] = {
                "true_positives": res["true_positives"],
                "false_positives": res["false_positives"],
                "false_negatives": res["false_negatives"],
                "num_tp": n_tp,
                "num_fp": n_fp,
                "num_fn": n_fn,
            }

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        output["aggregate_metrics"] = {
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "precision": precision,
            "recall": recall,
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        self._status.showMessage(f"Results saved to {path}", 5000)


def main():
    app = QApplication(sys.argv)
    window = SiftEvalApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()


