# nuitka-project: --standalone
# nuitka-project: --enable-plugin=pyside6
# nuitka-project: --include-package=onnxruntime
# nuitka-project: --include-package-data=onnxruntime
# nuitka-project: --include-package=cv2
# nuitka-project: --windows-console-mode=disable
# nuitka-project: --windows-icon-from-ico=app.ico
# nuitka-project: --output-filename=FemurXpert

from __future__ import annotations

import logging
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QIcon,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes
else:
    ctypes = None  # type: ignore[assignment]
    wintypes = None  # type: ignore[assignment]

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] (%(name)s) %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("femurxpert")

# ============================================================================
# Global Configuration and Constants
# ============================================================================

RES_TYPE: str = "XRAYMODEL"
RES_NAME: str = "MODEL"

CONFIDENCE_THRESHOLD: float = 0.368
MASK_OPACITY: float = 0.30

MODEL_CLASSES: list[str] = [
    "Greater Trochanteric Fracture",
    "Intertrochanteric Fracture",
    "Lesser Trochanteric Fracture",
    "Femoral Neck Fracture",
    "Subtrochanteric Fracture",
]

HEALTHY_LABEL: str = "Healthy (No Fracture)"
HEALTHY_COLOR: str = "#a3be8c"

CLASS_PALETTE: list[str] = [
    "#88c0d0",  # Greater Trochanter - Frost Ice Blue
    "#bf616a",  # Intertrochanteric   - Crimson / Coral
    "#ebcb8b",  # Lesser Trochanter   - Amber Gold
    "#d08770",  # Neck                - Orange / Coral
    "#b48ead",  # Subtrochanteric     - Soft Violet
]


def get_class_color(class_id: int) -> QColor:
    return QColor(CLASS_PALETTE[class_id % len(CLASS_PALETTE)])


def resolve_asset_path(filename: str | Path) -> Path:
    """Robust, multi-target asset path resolver.

    Searches across:
    1. Direct filesystem path if existent.
    2. Module directory (Nuitka standalone distribution or onefile temp folder).
    3. Executable directory (external files placed alongside the compiled .exe).
    4. Current working directory.
    5. sys.argv[0] directory.
    6. Legacy packaging root (sys._MEIPASS).
    """
    target = Path(filename)
    target_name = target.name

    candidates: list[Path] = [
        target,
        Path(__file__).resolve().parent / target_name,
        Path(sys.executable).resolve().parent / target_name,
        Path.cwd() / target_name,
    ]

    if sys.argv and sys.argv[0]:
        candidates.append(Path(sys.argv[0]).resolve().parent / target_name)

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / target_name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    return Path(__file__).resolve().parent / target_name


MODEL_FILE: Path = resolve_asset_path("best.onnx")
ICON_FILE: Path = resolve_asset_path("app.ico")

# ============================================================================
# PE Resource Utilities (Embedded Model Loader & Injector)
# ============================================================================


def load_embedded_model() -> bytes | None:
    """Read the ONNX model embedded in the Windows PE resource section."""
    if sys.platform != "win32" or ctypes is None or wintypes is None:
        return None

    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        k32.GetModuleHandleW.restype = ctypes.c_void_p
        k32.FindResourceW.argtypes = [
            ctypes.c_void_p,
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
        ]
        k32.FindResourceW.restype = ctypes.c_void_p
        k32.SizeofResource.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        k32.SizeofResource.restype = wintypes.DWORD
        k32.LoadResource.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        k32.LoadResource.restype = ctypes.c_void_p
        k32.LockResource.argtypes = [ctypes.c_void_p]
        k32.LockResource.restype = ctypes.c_void_p

        hmod = k32.GetModuleHandleW(None)
        hres = k32.FindResourceW(hmod, RES_NAME, RES_TYPE)
        if not hres:
            return None

        size = k32.SizeofResource(hmod, hres)
        res_handle = k32.LoadResource(hmod, hres)
        ptr = k32.LockResource(res_handle)
        if not ptr or not size:
            return None

        return ctypes.string_at(ptr, size)
    except Exception as exc:  # noqa: BLE001
        logger.debug("PE resource loading error: %s", exc)
        return None


def extract_pe_overlay(data: bytes) -> bytes:
    """Extract trailing overlay bytes appended past the last PE raw section."""
    if len(data) < 0x40 or data[:2] != b"MZ":
        return b""

    e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
    if len(data) < e_lfanew + 24 or data[e_lfanew : e_lfanew + 4] != b"PE\0\0":
        return b""

    num_sections = struct.unpack_from("<H", data, e_lfanew + 6)[0]
    size_of_opt_hdr = struct.unpack_from("<H", data, e_lfanew + 20)[0]
    sec_hdr_start = e_lfanew + 24 + size_of_opt_hdr

    max_raw_end = 0
    for i in range(num_sections):
        offset = sec_hdr_start + i * 40
        if offset + 40 > len(data):
            break
        size_of_raw = struct.unpack_from("<I", data, offset + 16)[0]
        ptr_to_raw = struct.unpack_from("<I", data, offset + 20)[0]
        max_raw_end = max(max_raw_end, ptr_to_raw + size_of_raw)

    if max_raw_end > 0 and len(data) > max_raw_end:
        return data[max_raw_end:]
    return b""


def embed_model_in_exe(exe_path: str | Path, model_path: str | Path) -> None:
    """Embed an ONNX model into a Windows executable PE resource."""
    if sys.platform != "win32" or ctypes is None or wintypes is None:
        raise RuntimeError("Embedding model resources is only supported on Windows.")

    target = Path(exe_path).resolve()
    model = Path(model_path).resolve()

    if not target.is_file():
        raise FileNotFoundError(f"Target executable not found: {target}")
    if not model.is_file():
        raise FileNotFoundError(f"Model file not found: {model}")

    exe_bytes = target.read_bytes()
    overlay = extract_pe_overlay(exe_bytes)
    data = model.read_bytes()

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.BeginUpdateResourceW.argtypes = [wintypes.LPCWSTR, wintypes.BOOL]
    k32.BeginUpdateResourceW.restype = ctypes.c_void_p
    k32.UpdateResourceW.argtypes = [
        ctypes.c_void_p,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.WORD,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    k32.UpdateResourceW.restype = wintypes.BOOL
    k32.EndUpdateResourceW.argtypes = [ctypes.c_void_p, wintypes.BOOL]
    k32.EndUpdateResourceW.restype = wintypes.BOOL

    h = k32.BeginUpdateResourceW(str(target), False)
    if not h:
        raise ctypes.WinError(ctypes.get_last_error())

    buf = ctypes.create_string_buffer(data, len(data))
    if not k32.UpdateResourceW(h, RES_TYPE, RES_NAME, 0, buf, len(data)):
        err = ctypes.get_last_error()
        k32.EndUpdateResourceW(h, True)
        raise ctypes.WinError(err)

    if not k32.EndUpdateResourceW(h, False):
        raise ctypes.WinError(ctypes.get_last_error())

    if overlay:
        with open(target, "ab") as f:
            f.write(overlay)

    logger.info(
        "Embedded %d model bytes into %s (preserved %d overlay bytes).",
        len(data),
        target.name,
        len(overlay),
    )


# ============================================================================
# Inference Engine (Segmenter)
# ============================================================================


class Segmenter:
    def __init__(self, n_classes: int, model_source: Path | str | bytes) -> None:
        self.nc = n_classes
        model_input: str | bytes = (
            bytes(model_source)
            if isinstance(model_source, (bytes, bytearray))
            else str(model_source)
        )

        so = ort.SessionOptions()
        so.log_severity_level = 3
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        avail = ort.get_available_providers()
        attempts = [
            ["DmlExecutionProvider", "CPUExecutionProvider"],
            ["CPUExecutionProvider"],
        ]

        sess: ort.InferenceSession | None = None
        for prov in attempts:
            if prov[0] not in avail:
                continue
            try:
                sess = ort.InferenceSession(model_input, so, providers=prov)
                logger.info("Initialized ONNX Runtime with provider: %s", prov[0])
                break
            except (RuntimeError, ValueError, OSError) as exc:
                logger.debug("Failed provider attempt %s: %s", prov[0], exc)
                continue

        if sess is None:
            raise RuntimeError("No usable inference provider found for ONNX Runtime.")

        self.sess: ort.InferenceSession = sess

        inp = self.sess.get_inputs()[0]
        self.inp: str = inp.name

        in_h = inp.shape[2] if len(inp.shape) > 2 else 640
        in_w = inp.shape[3] if len(inp.shape) > 3 else 640
        self.in_h: int = (
            int(in_h) if isinstance(in_h, (int, np.integer)) and in_h > 0 else 640
        )
        self.in_w: int = (
            int(in_w) if isinstance(in_w, (int, np.integer)) and in_w > 0 else 640
        )

        out_shape = self.sess.get_outputs()[0].shape
        if len(out_shape) > 1 and isinstance(out_shape[1], (int, np.integer)):
            n_ch = int(out_shape[1])
            expected_ch = 4 + self.nc + 32
            if n_ch != expected_ch:
                raise RuntimeError(
                    f"Model output mismatch: expected {self.nc} classes ({expected_ch} channels), "
                    f"model provides {n_ch} channels."
                )

    def warm_up(self) -> None:
        z = np.zeros((1, 3, self.in_h, self.in_w), dtype=np.float32)
        self.sess.run(None, {self.inp: z})

    def predict(
        self, rgb: np.ndarray, conf: float, iou: float = 0.7, max_det: int = 300
    ) -> list[
        tuple[tuple[float, float, float, float], float, int, list[tuple[float, float]]]
    ]:
        orig_h, orig_w = rgb.shape[:2]
        sh, sw = self.in_h, self.in_w
        r = min(sh / orig_h, sw / orig_w)
        nw, nh = round(orig_w * r), round(orig_h * r)
        left = round((sw - nw) / 2 - 0.1)
        top = round((sh - nh) / 2 - 0.1)

        canvas = np.full((sh, sw, 3), 114, dtype=np.uint8)
        canvas[top : top + nh, left : left + nw] = cv2.resize(
            rgb, (nw, nh), interpolation=cv2.INTER_LINEAR
        )
        blob = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None], dtype=np.float32)
        blob *= 1.0 / 255.0

        raw_outputs = self.sess.run(None, {self.inp: blob})
        out: np.ndarray = np.asarray(raw_outputs[0])
        protos: np.ndarray = np.asarray(raw_outputs[1])

        pred = out[0].T
        nm = int(protos.shape[1])

        scores = pred[:, 4 : 4 + self.nc]
        cf, cls = scores.max(1), scores.argmax(1)
        k = cf > conf
        pred, cf, cls = pred[k], cf[k], cls[k]
        if len(cf) == 0:
            return []

        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

        nb = xyxy + cls[:, None] * 4096.0
        tl = np.stack(
            [nb[:, 0], nb[:, 1], nb[:, 2] - nb[:, 0], nb[:, 3] - nb[:, 1]], axis=1
        )
        idx = np.asarray(
            cv2.dnn.NMSBoxes(tl.tolist(), cf.tolist(), float(conf), iou)
        ).reshape(-1)[:max_det]

        mh, mw = int(protos.shape[2]), int(protos.shape[3])
        pm = protos[0].reshape(nm, -1)
        xs, ys = np.arange(mw)[None, :], np.arange(mh)[:, None]
        res: list[
            tuple[
                tuple[float, float, float, float], float, int, list[tuple[float, float]]
            ]
        ] = []

        for i in idx:
            logits = (pred[i, 4 + self.nc :] @ pm).reshape(mh, mw)
            x1, y1, x2, y2 = xyxy[i] * np.array([mw / sw, mh / sh, mw / sw, mh / sh])
            logits = np.where(
                (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2), logits, 0.0
            )
            m = cv2.resize(logits, (sw, sh), interpolation=cv2.INTER_LINEAR) > 0

            poly: list[tuple[float, float]] = []
            cnts, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if cnts:
                c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
                c[:, 0] = np.clip((c[:, 0] - left) / r, 0, orig_w)
                c[:, 1] = np.clip((c[:, 1] - top) / r, 0, orig_h)
                if len(c) > 2:
                    poly = [(float(pt[0]), float(pt[1])) for pt in c]

            b = (xyxy[i] - np.array([left, top, left, top])) / r
            b = np.clip(b, 0, [orig_w, orig_h, orig_w, orig_h])
            box_coords = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            res.append((box_coords, float(cf[i]), int(cls[i]), poly))

        return res


# ============================================================================
# Data Models and Background Workers
# ============================================================================


@dataclass
class DetectionItem:
    box: tuple[float, float, float, float]
    score: float
    class_id: int
    class_name: str
    polygon: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class InferenceOutput:
    items: list[DetectionItem]
    inference_time_ms: float
    is_segmentation: bool


class ModelLoader(QThread):
    ready = Signal(object)
    failed = Signal(str)

    def __init__(self, model_path: Path, n_classes: int) -> None:
        super().__init__()
        self.model_path = model_path
        self.n_classes = n_classes

    def run(self) -> None:
        try:
            embedded_bytes = load_embedded_model()
            if embedded_bytes is not None:
                seg = Segmenter(self.n_classes, embedded_bytes)
            else:
                path = (
                    self.model_path
                    if self.model_path.is_file()
                    else resolve_asset_path(self.model_path.name)
                )
                if path.is_file():
                    seg = Segmenter(self.n_classes, path)
                else:
                    raise FileNotFoundError(
                        f"Model not found in PE resource or at: {self.model_path}"
                    )

            seg.warm_up()
            self.ready.emit(seg)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"Could not initialize analysis model: {exc}")


class InferenceWorker(QThread):
    finished_signal = Signal(int, object)
    failed_signal = Signal(int, str)

    def __init__(
        self, seg: Segmenter, image_rgb: np.ndarray, conf: float, job_id: int
    ) -> None:
        super().__init__()
        self.seg = seg
        self.image_rgb = image_rgb
        self.conf = conf
        self.job_id = job_id

    def run(self) -> None:
        try:
            t0 = time.perf_counter()
            dets = self.seg.predict(self.image_rgb, self.conf)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            items = [
                DetectionItem(
                    box=b,
                    score=s,
                    class_id=c,
                    class_name=MODEL_CLASSES[c]
                    if 0 <= c < len(MODEL_CLASSES)
                    else f"Class_{c}",
                    polygon=p,
                )
                for b, s, c, p in dets
            ]
            self.finished_signal.emit(
                self.job_id,
                InferenceOutput(
                    items=items,
                    inference_time_ms=elapsed_ms,
                    is_segmentation=True,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self.failed_signal.emit(self.job_id, f"Inference error: {exc}")


# ============================================================================
# Custom GUI Widgets
# ============================================================================


class SpinnerWidget(QWidget):
    def __init__(
        self, parent: QWidget | None = None, size: int = 22, color: str = "#88c0d0"
    ) -> None:
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._angle = 0
        self._color = QColor(color)
        self._track_color = QColor(68, 77, 94, 90)
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._step)

    def start(self) -> None:
        self._angle = 0
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def _step(self) -> None:
        self._angle = (self._angle + 6) % 360
        self.update()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        cx, cy = self.width() / 2.0, self.height() / 2.0
        r = min(cx, cy) - 2.5
        rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)

        painter.setPen(
            QPen(self._track_color, 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        )
        painter.drawEllipse(rect)

        painter.setPen(
            QPen(self._color, 2.4, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        )
        painter.drawArc(rect, int(-self._angle * 16), 100 * 16)


class RoundResetButton(QPushButton):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(44, 44)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Reset View (Zoom, Pan & Windowing)")
        self._hovered = False

    def enterEvent(self, event: Any) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: Any) -> None:
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(1, 1, -1, -1)
        bg = (
            QColor(35, 40, 49, 245)
            if self.isDown()
            else (QColor(42, 48, 59, 235) if self._hovered else QColor(35, 40, 49, 210))
        )
        border = QColor("#88c0d0") if self._hovered else QColor("#444d5e")

        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(border, 1.5))
        painter.drawEllipse(rect)

        fg = QColor("#88c0d0") if self._hovered else QColor("#e5e9f0")
        cx, cy = self.width() / 2.0, self.height() / 2.0
        r = 9.0
        arc_rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)

        painter.setPen(
            QPen(
                fg,
                2.0,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        )
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(arc_rect, 40 * 16, 270 * 16)

        tip_x = cx + r * 0.766
        tip_y = cy - r * 0.643
        arrow = QPolygonF(
            [
                QPointF(tip_x - 1, tip_y - 4.5),
                QPointF(tip_x + 3.5, tip_y + 0.5),
                QPointF(tip_x - 3.5, tip_y + 2.5),
            ]
        )
        painter.setPen(QPen(fg, 1.0))
        painter.setBrush(QBrush(fg))
        painter.drawPolygon(arrow)


class RoundEyeButton(QPushButton):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(44, 44)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Toggle Label & Overlay Visibility")
        self._hovered = False
        self._eye_open = True

    def is_eye_open(self) -> bool:
        return self._eye_open

    def set_eye_open(self, open_state: bool) -> None:
        if self._eye_open != open_state:
            self._eye_open = open_state
            self.update()

    def enterEvent(self, event: Any) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event: Any) -> None:
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(1, 1, -1, -1)
        bg = (
            QColor(35, 40, 49, 245)
            if self.isDown()
            else (QColor(42, 48, 59, 235) if self._hovered else QColor(35, 40, 49, 210))
        )
        border = QColor("#88c0d0") if self._hovered else QColor("#444d5e")

        painter.setBrush(QBrush(bg))
        painter.setPen(QPen(border, 1.5))
        painter.drawEllipse(rect)

        fg = (
            (QColor("#88c0d0") if self._hovered else QColor("#e5e9f0"))
            if self._eye_open
            else (QColor("#bf616a") if self._hovered else QColor("#7b889b"))
        )

        cx, cy = self.width() / 2.0, self.height() / 2.0

        path = QPainterPath()
        path.moveTo(cx - 9.5, cy)
        path.quadTo(cx, cy - 7.5, cx + 9.5, cy)
        path.quadTo(cx, cy + 7.5, cx - 9.5, cy)

        painter.setPen(
            QPen(
                fg,
                1.8,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        )
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        painter.setBrush(QBrush(fg))
        painter.drawEllipse(QPointF(cx, cy), 3.0, 3.0)

        if not self._eye_open:
            painter.setPen(
                QPen(
                    fg,
                    2.0,
                    Qt.PenStyle.SolidLine,
                    Qt.PenCapStyle.RoundCap,
                )
            )
            painter.drawLine(
                QPointF(cx - 10.0, cy - 7.5),
                QPointF(cx + 10.0, cy + 7.5),
            )


class CenteredLoadingOverlay(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("centeredLoadingOverlay")
        self.setFixedSize(180, 52)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setColor(QColor(0, 0, 0, 180))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 10, 18, 10)
        layout.setSpacing(12)

        self.spinner = SpinnerWidget(self, size=22, color="#88c0d0")
        layout.addWidget(self.spinner)

        self.label = QLabel("Analyzing...")
        self.label.setObjectName("loadingLabel")
        layout.addWidget(self.label)

        self.hide()

    def show_loading(self, text: str = "Analyzing...") -> None:
        self.label.setText(text)
        self.spinner.start()
        self.show()
        self.raise_()

    def hide_loading(self) -> None:
        self.spinner.stop()
        self.hide()


class LabelLegendPanel(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("legendPanel")

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 160))
        shadow.setOffset(0, 3)
        self.setGraphicsEffect(shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 16, 12)
        layout.setSpacing(6)

        self.labels: list[QLabel] = []
        for name in MODEL_CLASSES:
            lbl = QLabel(f"⚈ {name}")
            lbl.setObjectName("anatomyLabel")
            lbl.setStyleSheet("color: #6c7689; font-size: 12px; font-weight: 500;")
            layout.addWidget(lbl)
            self.labels.append(lbl)

        self.lbl_healthy = QLabel(f"⚈ {HEALTHY_LABEL}")
        self.lbl_healthy.setObjectName("healthyLabel")
        self.lbl_healthy.setStyleSheet(
            "color: #6c7689; font-size: 12px; font-weight: 500;"
        )
        layout.addWidget(self.lbl_healthy)

    def reset_labels(self) -> None:
        for lbl in self.labels:
            lbl.setStyleSheet("color: #6c7689; font-size: 12px; font-weight: 500;")
        self.lbl_healthy.setStyleSheet(
            "color: #6c7689; font-size: 12px; font-weight: 500;"
        )

    def update_detected(self, detected_indices: set[int]) -> None:
        if not detected_indices:
            for lbl in self.labels:
                lbl.setStyleSheet("color: #6c7689; font-size: 12px; font-weight: 500;")
            self.lbl_healthy.setStyleSheet(
                f"color: {HEALTHY_COLOR}; font-size: 12px; font-weight: 700;"
            )
            return

        self.lbl_healthy.setStyleSheet(
            "color: #6c7689; font-size: 12px; font-weight: 500;"
        )
        for idx, lbl in enumerate(self.labels):
            if idx in detected_indices:
                col = CLASS_PALETTE[idx % len(CLASS_PALETTE)]
                lbl.setStyleSheet(f"color: {col}; font-size: 12px; font-weight: 700;")
            else:
                lbl.setStyleSheet("color: #6c7689; font-size: 12px; font-weight: 500;")


# ============================================================================
# Main Viewport
# ============================================================================


class XRayGraphicsView(QGraphicsView):
    file_dropped = Signal(str)
    view_resized = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.graphics_scene = QGraphicsScene(self)
        self.setScene(self.graphics_scene)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self.setAcceptDrops(True)

        self.raw_image_rgb: np.ndarray | None = None
        self.pixmap_item = QGraphicsPixmapItem()
        self.pixmap_item.setZValue(0)
        self.graphics_scene.addItem(self.pixmap_item)

        self.contrast: float = 1.0
        self.brightness: float = 0.0

        self.is_dragging_contrast: bool = False
        self.is_dragging_brightness: bool = False
        self.is_panning: bool = False
        self.last_mouse_pos = QPoint()

        self.overlay_items: list[QGraphicsItem] = []
        self.detection_items: list[DetectionItem] = []
        self._overlay_visible: bool = True

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.view_resized.emit()

    def set_image(self, image_rgb: np.ndarray) -> None:
        self.raw_image_rgb = image_rgb
        self.detection_items.clear()
        self.clear_overlay()
        self.reset_window_level()
        self.fit_to_screen()

    def update_display_pixmap(self) -> None:
        if self.raw_image_rgb is None:
            return

        c = float(self.contrast)
        b = float(self.brightness)

        lut = np.arange(256, dtype=np.float32)
        lut = np.clip((lut - 128.0) * c + 128.0 + b, 0.0, 255.0).astype(np.uint8)

        adjusted_rgb = np.ascontiguousarray(cv2.LUT(self.raw_image_rgb, lut))
        h, w, ch = adjusted_rgb.shape

        qimg = QImage(adjusted_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.pixmap_item.setPixmap(QPixmap.fromImage(qimg))
        self.graphics_scene.setSceneRect(0, 0, w, h)

    def set_detections(self, items: list[DetectionItem]) -> None:
        self.detection_items = items
        self.render_overlay()

    def clear_overlay(self) -> None:
        for item in self.overlay_items:
            self.graphics_scene.removeItem(item)
        self.overlay_items.clear()

    def set_overlay_visible(self, visible: bool) -> None:
        self._overlay_visible = visible
        for item in self.overlay_items:
            item.setVisible(visible)

    def render_overlay(self) -> None:
        self.clear_overlay()
        if not self.detection_items:
            return

        for item in self.detection_items:
            if item.score < CONFIDENCE_THRESHOLD:
                continue

            color = get_class_color(item.class_id)
            if item.polygon:
                qpoly = QPolygonF([QPointF(x, y) for x, y in item.polygon])
                poly_item = QGraphicsPolygonItem(qpoly)
                pen = QPen(color, 2.0)
                pen.setCosmetic(True)
                poly_item.setPen(pen)

                fill_color = QColor(
                    color.red(),
                    color.green(),
                    color.blue(),
                    int(255 * MASK_OPACITY),
                )
                poly_item.setBrush(QBrush(fill_color))
                poly_item.setZValue(10)
                poly_item.setVisible(self._overlay_visible)
                self.graphics_scene.addItem(poly_item)
                self.overlay_items.append(poly_item)

    def reset_window_level(self) -> None:
        self.contrast = 1.0
        self.brightness = 0.0
        self.update_display_pixmap()

    def fit_to_screen(self) -> None:
        if self.raw_image_rgb is None:
            return
        self.resetTransform()
        rect = self.graphics_scene.sceneRect()
        if not rect.isEmpty():
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def reset_all_view(self) -> None:
        self.reset_window_level()
        self.fit_to_screen()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self.raw_image_rgb is None:
            return
        angle = event.angleDelta().y()
        if angle == 0:
            return
        factor = 1.15 if angle > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self.last_mouse_pos = event.pos()
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging_contrast = True
        elif event.button() == Qt.MouseButton.RightButton:
            self.is_dragging_brightness = True
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.raw_image_rgb is None:
            super().mouseMoveEvent(event)
            return

        delta = event.pos() - self.last_mouse_pos
        self.last_mouse_pos = event.pos()

        if self.is_dragging_contrast:
            diff = delta.x() - delta.y()
            self.contrast = float(np.clip(self.contrast + diff * 0.008, 0.1, 5.0))
            self.update_display_pixmap()
        elif self.is_dragging_brightness:
            diff = delta.x() - delta.y()
            self.brightness = float(
                np.clip(self.brightness + diff * 0.5, -150.0, 150.0)
            )
            self.update_display_pixmap()
        elif self.is_panning:
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging_contrast = False
        elif event.button() == Qt.MouseButton.RightButton:
            self.is_dragging_brightness = False
        elif event.button() == Qt.MouseButton.MiddleButton:
            self.is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                ext = Path(url.toLocalFile()).suffix.lower()
                if ext in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                local_path = url.toLocalFile()
                ext = Path(local_path).suffix.lower()
                if ext in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}:
                    self.file_dropped.emit(local_path)
                    event.acceptProposedAction()
                    return
        event.ignore()


# ============================================================================
# Main Window Coordinator
# ============================================================================


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("FemurXpert")
        self.resize(1200, 800)

        if ICON_FILE.is_file():
            self.setWindowIcon(QIcon(str(ICON_FILE)))

        self.segmenter: Segmenter | None = None
        self._job_id: int = 0
        self._pending_rgb: np.ndarray | None = None
        self._workers: set[QThread] = set()
        self._labels_visible: bool = True

        self.viewer = XRayGraphicsView(self)
        self.setCentralWidget(self.viewer)

        self._init_floating_ui()
        self._apply_theme()

        self.viewer.file_dropped.connect(self._handle_file_dropped)
        self.viewer.view_resized.connect(self._position_floating_widgets)

        self.loading_overlay.show_loading("Loading model...")
        self.loader = ModelLoader(MODEL_FILE, len(MODEL_CLASSES))
        self.loader.ready.connect(self._on_model_ready)
        self.loader.failed.connect(self._on_model_failed)
        self.loader.start()

    def _init_floating_ui(self) -> None:
        vp = self.viewer.viewport()

        self.btn_reset = RoundResetButton(vp)
        self.btn_reset.clicked.connect(self.viewer.reset_all_view)

        reset_shadow = QGraphicsDropShadowEffect(self.btn_reset)
        reset_shadow.setBlurRadius(16)
        reset_shadow.setColor(QColor(0, 0, 0, 160))
        reset_shadow.setOffset(0, 3)
        self.btn_reset.setGraphicsEffect(reset_shadow)

        self.btn_eye = RoundEyeButton(vp)
        self.btn_eye.clicked.connect(self._toggle_labels_visibility)

        eye_shadow = QGraphicsDropShadowEffect(self.btn_eye)
        eye_shadow.setBlurRadius(16)
        eye_shadow.setColor(QColor(0, 0, 0, 160))
        eye_shadow.setOffset(0, 3)
        self.btn_eye.setGraphicsEffect(eye_shadow)

        self.legend_panel = LabelLegendPanel(vp)
        self.loading_overlay = CenteredLoadingOverlay(vp)
        self._position_floating_widgets()

    def _position_floating_widgets(self) -> None:
        vp = self.viewer.viewport()
        vw_h = vp.height()
        vw_w = vp.width()

        btn_y = max(20, vw_h - self.btn_reset.height() - 20)
        self.btn_reset.move(20, btn_y)
        self.btn_eye.move(74, btn_y)

        self.legend_panel.adjustSize()
        self.legend_panel.move(20, 20)

        cx = max(0, (vw_w - self.loading_overlay.width()) // 2)
        cy = max(0, (vw_h - self.loading_overlay.height()) // 2)
        self.loading_overlay.move(cx, cy)

    def _toggle_labels_visibility(self) -> None:
        self._labels_visible = not self._labels_visible
        self.btn_eye.set_eye_open(self._labels_visible)
        self.legend_panel.setVisible(self._labels_visible)
        self.viewer.set_overlay_visible(self._labels_visible)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._position_floating_widgets()

    def _apply_theme(self) -> None:
        qss = """
        QMainWindow { background-color: #1e222a; }
        QGraphicsView { background-color: #181b20; border: none; }
        QFrame#legendPanel {
            background-color: rgba(35, 40, 49, 0.92);
            border: 1px solid #444d5e;
            border-radius: 8px;
        }
        QFrame#centeredLoadingOverlay {
            background-color: rgba(35, 40, 49, 0.94);
            border: 1px solid #444d5e;
            border-radius: 26px;
        }
        QLabel#loadingLabel {
            color: #eceff4;
            font-size: 13px;
            font-weight: 600;
            letter-spacing: 0.4px;
        }
        QWidget {
            color: #d8dee9;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }
        """
        self.setStyleSheet(qss)

    def _on_model_ready(self, seg: Segmenter) -> None:
        self.segmenter = seg
        self.loading_overlay.hide_loading()
        if self._pending_rgb is not None:
            rgb = self._pending_rgb
            self._pending_rgb = None
            self._start_automatic_inference(rgb)

    def _on_model_failed(self, msg: str) -> None:
        self.loading_overlay.hide_loading()
        QMessageBox.critical(self, "Model Initialization Error", msg)

    def _handle_file_dropped(self, path_str: str) -> None:
        path = Path(path_str)
        if not path.is_file():
            return

        try:
            with open(path, "rb") as f:
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)

            img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError("Could not decode image with OpenCV.")

            if img.dtype != np.uint8:
                img_float = img.astype(np.float32)
                img_min = float(np.min(img_float))
                img_max = float(np.max(img_float))
                if img_max > img_min:
                    img = ((img_float - img_min) / (img_max - img_min) * 255.0).astype(
                        np.uint8
                    )
                else:
                    img = np.zeros_like(img, dtype=np.uint8)

            if len(img.shape) == 2:
                rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(img.shape) == 3:
                if img.shape[2] == 4:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                elif img.shape[2] == 3:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    raise ValueError(f"Unsupported channel dimension: {img.shape[2]}")
            else:
                raise ValueError(f"Unsupported matrix shape: {img.shape}")

            self.viewer.set_image(rgb)
            self.legend_panel.reset_labels()
            self._start_automatic_inference(rgb)

        except (ValueError, OSError, RuntimeError, cv2.error) as exc:
            QMessageBox.critical(
                self, "Error Loading Image", f"Failed to load {path.name}: {exc}"
            )

    def _start_automatic_inference(self, rgb_image: np.ndarray) -> None:
        if self.segmenter is None:
            self._pending_rgb = rgb_image
            return

        for worker in list(self._workers):
            worker.quit()
        self._workers.clear()

        self.loading_overlay.show_loading("Analyzing X-ray...")
        self._job_id += 1
        current_job = self._job_id

        worker = InferenceWorker(
            self.segmenter, rgb_image, CONFIDENCE_THRESHOLD, current_job
        )
        worker.finished_signal.connect(self._on_inference_completed)
        worker.failed_signal.connect(self._on_inference_failed)
        worker.finished.connect(lambda w=worker: self._workers.discard(w))
        self._workers.add(worker)
        worker.start()

    def _on_inference_completed(self, job_id: int, output: InferenceOutput) -> None:
        if job_id != self._job_id:
            return

        self.loading_overlay.hide_loading()
        self.viewer.set_detections(output.items)

        detected_indices = {
            item.class_id for item in output.items if item.score >= CONFIDENCE_THRESHOLD
        }
        self.legend_panel.update_detected(detected_indices)

    def _on_inference_failed(self, job_id: int, error_msg: str) -> None:
        if job_id != self._job_id:
            return

        self.loading_overlay.hide_loading()
        QMessageBox.critical(self, "Inference Failure", error_msg)

    def closeEvent(self, event: Any) -> None:
        for t in list(self._workers):
            t.quit()
            t.wait(1000)
        if self.loader.isRunning():
            self.loader.quit()
            self.loader.wait(1000)
        super().closeEvent(event)


# ============================================================================
# Main Dispatcher
# ============================================================================


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "--embed":
        embed_model_in_exe(sys.argv[2], sys.argv[3])
        return

    if (
        len(sys.argv) == 3
        and sys.argv[1].lower().endswith(".exe")
        and sys.argv[2].lower().endswith(".onnx")
    ):
        embed_model_in_exe(sys.argv[1], sys.argv[2])
        return

    if sys.platform == "win32" and ctypes is not None:
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "FemurXpert.DesktopViewer"
            )
        except Exception:  # noqa: BLE001, S110
            pass

    app = QApplication(sys.argv)
    if ICON_FILE.is_file():
        app.setWindowIcon(QIcon(str(ICON_FILE)))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
