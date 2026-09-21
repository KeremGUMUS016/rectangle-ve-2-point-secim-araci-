import copy
import json
import math
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import (
    Qt,
    QPointF,
    QRectF,
    Signal,
    QEvent,
    QObject,
    QThread,
    QProcess,
    QProcessEnvironment,
)
from PySide6.QtGui import (
    QAction,
    QColor,
    QImageReader,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QToolBar,
    QVBoxLayout,
    QWidget,
)


# ============================================================
# AYARLAR
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

AUTOSAVE_FILENAME = ".fidan_pose_annotations.json"
TRAINING_FOLDER = "fidan_pose_training"

CLASS_ID = 0
CLASS_NAME = "fidan"

KEYPOINT_NAMES = [
    "sap_ucu",
    "aci_noktasi",
]

BASE_MODEL = "yolo11m-pose.pt"

TRAIN_IMAGE_SIZE = 1024
PREDICT_IMAGE_SIZE = 1024
TRAIN_EPOCHS = 300
TRAIN_BATCH = 2
TRAIN_PATIENCE = 80
DEFAULT_CONFIDENCE = 0.35

COLORS = {
    "background": QColor("#202124"),
    "bbox": QColor("#00D8FF"),
    "active_bbox": QColor("#FFD740"),
    "selected_bbox": QColor("#FF9800"),
    "model_bbox": QColor("#E040FB"),
    "sap_ucu": QColor("#00E676"),
    "aci_noktasi": QColor("#FF5252"),
    "edge": QColor("#FFD740"),
    "text": QColor("#FFFFFF"),
    "handle": QColor("#FFFFFF"),
    "active_handle": QColor("#FFD740"),
    "draft": QColor("#FF9800"),
    "human_fill": QColor(255, 215, 64, 50),
    "active_fill": QColor(255, 215, 64, 65),
    "model_fill": QColor(224, 64, 251, 45),
}


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def normalize_bbox(bbox):
    x1, y1, x2, y2 = bbox

    return [
        min(x1, x2),
        min(y1, y2),
        max(x1, x2),
        max(y1, y2),
    ]


def point_distance(point_a, point_b):
    return math.hypot(
        point_a[0] - point_b[0],
        point_a[1] - point_b[1],
    )


def normalize_object_metadata(obj):
    result = copy.deepcopy(obj)

    # Eski JSON kayıtlarında bu alanlar yoktu.
    # Eski anotasyonlar elle oluşturulduğu için onaylı kabul edilir.
    result.setdefault("source", "human")
    result.setdefault("reviewed", True)
    result.setdefault("confidence", None)
    result.setdefault("keypoint_confidence", None)
    result.setdefault("model_path", None)

    return result


# ============================================================
# YOLO POSE ÖN ETİKETLEME WORKER
# ============================================================

class PoseInferenceWorker(QObject):
    progress = Signal(int, int, str)
    prediction_ready = Signal(str, list)
    completed = Signal()
    failed = Signal(str)

    def __init__(
        self,
        model_path,
        image_items,
        confidence,
    ):
        super().__init__()

        self.model_path = str(model_path)
        self.image_items = list(image_items)
        self.confidence = float(confidence)
        self.cancel_requested = False

    def cancel(self):
        self.cancel_requested = True

    def run(self):
        try:
            import torch
            from ultralytics import YOLO

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA kullanılamıyor. Uygulamayı .venv "
                    "içindeki Python ile çalıştırın."
                )

            model = YOLO(self.model_path)
            total = len(self.image_items)

            for position, (key, image_path) in enumerate(
                self.image_items,
                start=1,
            ):
                if self.cancel_requested:
                    break

                self.progress.emit(
                    position,
                    total,
                    Path(image_path).name,
                )

                results = model.predict(
                    source=str(image_path),
                    imgsz=PREDICT_IMAGE_SIZE,
                    conf=self.confidence,
                    iou=0.50,
                    device=0,
                    verbose=False,
                )

                predictions = []

                if results:
                    result = results[0]
                    boxes = result.boxes
                    keypoints = result.keypoints

                    if (
                        boxes is not None
                        and keypoints is not None
                        and keypoints.xy is not None
                    ):
                        boxes_xyxy = (
                            boxes.xyxy
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        if boxes.conf is not None:
                            box_confidences = (
                                boxes.conf
                                .detach()
                                .cpu()
                                .numpy()
                            )
                        else:
                            box_confidences = [
                                None
                            ] * len(boxes_xyxy)

                        points_xy = (
                            keypoints.xy
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        points_confidence = None

                        if keypoints.conf is not None:
                            points_confidence = (
                                keypoints.conf
                                .detach()
                                .cpu()
                                .numpy()
                            )

                        for index, bbox in enumerate(
                            boxes_xyxy
                        ):
                            if index >= len(points_xy):
                                continue

                            points = points_xy[index]

                            if len(points) != 2:
                                raise RuntimeError(
                                    "Seçilen model iki keypoint üretmiyor. "
                                    "Yalnız bu fidan veri setiyle eğitilmiş "
                                    "best.pt dosyasını kullanın."
                                )

                            x1, y1, x2, y2 = [
                                float(value)
                                for value in bbox
                            ]

                            grasp = [
                                float(points[0][0]),
                                float(points[0][1]),
                            ]

                            angle = [
                                float(points[1][0]),
                                float(points[1][1]),
                            ]

                            box_confidence = None

                            if (
                                index
                                < len(box_confidences)
                                and box_confidences[index]
                                is not None
                            ):
                                box_confidence = float(
                                    box_confidences[index]
                                )

                            average_keypoint_confidence = None

                            if (
                                points_confidence is not None
                                and index
                                < len(points_confidence)
                            ):
                                values = points_confidence[index]

                                if len(values) > 0:
                                    average_keypoint_confidence = (
                                        float(
                                            sum(values)
                                            / len(values)
                                        )
                                    )

                            predictions.append({
                                "bbox": normalize_bbox([
                                    x1,
                                    y1,
                                    x2,
                                    y2,
                                ]),
                                "grasp": grasp,
                                "angle": angle,
                                "source": "model",
                                "reviewed": False,
                                "confidence": box_confidence,
                                "keypoint_confidence": (
                                    average_keypoint_confidence
                                ),
                                "model_path": self.model_path,
                            })

                self.prediction_ready.emit(
                    key,
                    predictions,
                )

            try:
                del model
                torch.cuda.empty_cache()
            except Exception:
                pass

            self.completed.emit()

        except Exception as error:
            self.failed.emit(
                f"{type(error).__name__}: {error}"
            )


# ============================================================
# ANOTASYON ALANI
# ============================================================

class AnnotationCanvas(QWidget):
    annotations_changed = Signal()
    mode_changed = Signal(str)
    selection_changed = Signal()

    pan_requested = Signal(float, float)
    zoom_requested = Signal(float, QPointF)

    def __init__(self):
        super().__init__()

        self.setMouseTracking(True)
        self.setFocusPolicy(
            Qt.FocusPolicy.StrongFocus
        )

        self.grabGesture(
            Qt.GestureType.PinchGesture
        )

        self.pixmap = None
        self.image_width = 0
        self.image_height = 0

        self.objects = []

        self.selected_index = None
        self.active_annotation_index = None

        # box, select, grasp, angle
        self.mode = "box"
        self.zoom = 1.0

        self.workspace_margin_x = 0
        self.workspace_margin_y = 0

        self.box_start = None
        self.box_current = None

        self.edit_drag_type = None
        self.edit_drag_corner = None
        self.edit_drag_index = None
        self.edit_drag_original_object = None
        self.edit_drag_before_snapshot = None
        self.edit_drag_changed = False

        self.undo_stack = []
        self.redo_stack = []
        self.max_history = 500

        self.setMinimumSize(800, 600)

    # ========================================================
    # GÖRÜNTÜ
    # ========================================================

    def load_image(self, image_path, objects):
        pixmap = QPixmap(str(image_path))

        if pixmap.isNull():
            raise RuntimeError(
                f"Görüntü açılamadı:\n{image_path}"
            )

        self.pixmap = pixmap
        self.image_width = pixmap.width()
        self.image_height = pixmap.height()

        self.objects = [
            normalize_object_metadata(obj)
            for obj in copy.deepcopy(objects)
        ]

        self.selected_index = None
        self.active_annotation_index = None

        self.undo_stack.clear()
        self.redo_stack.clear()

        self.reset_interaction()
        self.set_mode("box")

        self.update_canvas_size()
        self.update()

    def reset_interaction(self):
        self.box_start = None
        self.box_current = None
        self.clear_edit_drag()

    def get_objects(self):
        return copy.deepcopy(self.objects)

    def set_mode(self, mode):
        if mode not in {
            "box",
            "select",
            "grasp",
            "angle",
        }:
            return

        self.mode = mode
        self.mode_changed.emit(mode)
        self.update()

    def set_workspace_margins(
        self,
        width,
        height,
    ):
        self.workspace_margin_x = max(
            0,
            int(width),
        )

        self.workspace_margin_y = max(
            0,
            int(height),
        )

        self.update_canvas_size()
        self.update()

    def update_canvas_size(self):
        if self.pixmap is None:
            return

        scaled_width = int(
            round(self.image_width * self.zoom)
        )

        scaled_height = int(
            round(self.image_height * self.zoom)
        )

        self.setFixedSize(
            max(
                1,
                scaled_width
                + 2 * self.workspace_margin_x,
            ),
            max(
                1,
                scaled_height
                + 2 * self.workspace_margin_y,
            ),
        )

    def image_position(self, event):
        x = (
            event.position().x()
            - self.workspace_margin_x
        ) / self.zoom

        y = (
            event.position().y()
            - self.workspace_margin_y
        ) / self.zoom

        return [
            clamp(x, 0, self.image_width),
            clamp(y, 0, self.image_height),
        ]

    def event_is_over_image(self, event):
        left = self.workspace_margin_x
        top = self.workspace_margin_y

        right = (
            left
            + self.image_width * self.zoom
        )

        bottom = (
            top
            + self.image_height * self.zoom
        )

        position = event.position()

        return (
            left <= position.x() <= right
            and top <= position.y() <= bottom
        )

    # ========================================================
    # NOKTALARI KUTU İÇİNDE TUTMA
    # ========================================================

    def clamp_point_to_bbox(
        self,
        point,
        bbox,
    ):
        x1, y1, x2, y2 = bbox

        return [
            clamp(point[0], x1, x2),
            clamp(point[1], y1, y2),
        ]

    def clamp_object_keypoints(self, obj):
        for key in ("grasp", "angle"):
            point = obj.get(key)

            if point is not None:
                obj[key] = self.clamp_point_to_bbox(
                    point,
                    obj["bbox"],
                )
    def mark_object_as_edited(self, obj):
        if (
            obj.get("source") == "model"
            and not obj.get("reviewed", False)
        ):
            obj["source"] = "model_edited"

    # ========================================================
    # UNDO / REDO
    # ========================================================

    def current_snapshot(self):
        return {
            "objects": copy.deepcopy(
                self.objects
            ),
            "selected_index": self.selected_index,
            "active_annotation_index": (
                self.active_annotation_index
            ),
            "mode": self.mode,
        }

    def push_snapshot(self, snapshot=None):
        if snapshot is None:
            snapshot = self.current_snapshot()

        self.undo_stack.append(
            copy.deepcopy(snapshot)
        )

        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)

        self.redo_stack.clear()

    def restore_snapshot(self, snapshot):
        self.objects = copy.deepcopy(
            snapshot.get("objects", [])
        )

        self.selected_index = snapshot.get(
            "selected_index"
        )

        self.active_annotation_index = snapshot.get(
            "active_annotation_index"
        )

        self.mode = snapshot.get(
            "mode",
            "box",
        )

        if self.selected_index is not None:
            if not (
                0
                <= self.selected_index
                < len(self.objects)
            ):
                self.selected_index = None

        if self.active_annotation_index is not None:
            if not (
                0
                <= self.active_annotation_index
                < len(self.objects)
            ):
                self.active_annotation_index = None

        self.reset_interaction()

        self.mode_changed.emit(self.mode)
        self.selection_changed.emit()
        self.annotations_changed.emit()
        self.update()

    def undo(self):
        if not self.undo_stack:
            return

        self.redo_stack.append(
            self.current_snapshot()
        )

        self.restore_snapshot(
            self.undo_stack.pop()
        )

    def redo(self):
        if not self.redo_stack:
            return

        self.undo_stack.append(
            self.current_snapshot()
        )

        self.restore_snapshot(
            self.redo_stack.pop()
        )

    # ========================================================
    # SİLME / NOKTA YENİLEME
    # ========================================================

    def delete_selected(self):
        if self.selected_index is None:
            return

        if not (
            0
            <= self.selected_index
            < len(self.objects)
        ):
            return

        self.push_snapshot()

        deleted_index = self.selected_index
        self.objects.pop(deleted_index)

        if (
            self.active_annotation_index
            == deleted_index
        ):
            self.active_annotation_index = None
            self.set_mode("box")

        elif (
            self.active_annotation_index is not None
            and self.active_annotation_index
            > deleted_index
        ):
            self.active_annotation_index -= 1

        self.selected_index = None

        self.annotations_changed.emit()
        self.selection_changed.emit()
        self.update()

    def delete_last_template(self):
        if not self.objects:
            return

        self.push_snapshot()

        deleted_index = len(self.objects) - 1
        self.objects.pop()

        if (
            self.active_annotation_index
            == deleted_index
        ):
            self.active_annotation_index = None

        if self.selected_index == deleted_index:
            self.selected_index = None

        self.set_mode("box")

        self.annotations_changed.emit()
        self.selection_changed.emit()
        self.update()

    def restart_points_for_selected(self):
        if self.selected_index is None:
            return

        if not (
            0
            <= self.selected_index
            < len(self.objects)
        ):
            return

        self.push_snapshot()

        obj = self.objects[self.selected_index]
        obj["grasp"] = None
        obj["angle"] = None

        self.active_annotation_index = (
            self.selected_index
        )

        self.set_mode("grasp")

        self.annotations_changed.emit()
        self.update()

    def cancel_active_operation(self):
        self.box_start = None
        self.box_current = None

        self.clear_edit_drag()

        self.active_annotation_index = None
        self.selected_index = None

        self.set_mode("box")

        self.selection_changed.emit()
        self.update()

    # ========================================================
    # ZOOM VE TOUCHPAD
    # ========================================================

    def set_zoom(self, zoom):
        self.zoom = clamp(
            float(zoom),
            0.05,
            8.0,
        )

        self.update_canvas_size()
        self.update()

    def zoom_in(self):
        self.set_zoom(self.zoom * 1.2)

    def zoom_out(self):
        self.set_zoom(self.zoom / 1.2)

    def reset_zoom(self):
        self.set_zoom(1.0)

    def fit_zoom(
        self,
        available_width,
        available_height,
    ):
        if self.pixmap is None:
            return

        if (
            self.image_width <= 0
            or self.image_height <= 0
        ):
            return

        scale_x = (
            available_width
            / self.image_width
        )

        scale_y = (
            available_height
            / self.image_height
        )

        self.set_zoom(
            min(
                max(0.05, scale_x),
                max(0.05, scale_y),
                1.0,
            )
        )

    def wheelEvent(self, event):
        pixel_delta = event.pixelDelta()
        angle_delta = event.angleDelta()

        if (
            event.modifiers()
            & Qt.KeyboardModifier.ControlModifier
        ):
            delta_y = (
                pixel_delta.y()
                if not pixel_delta.isNull()
                else angle_delta.y()
            )

            if delta_y > 0:
                factor = 1.08
            elif delta_y < 0:
                factor = 1 / 1.08
            else:
                event.accept()
                return

            self.zoom_requested.emit(
                factor,
                event.position(),
            )

            event.accept()
            return

        if not pixel_delta.isNull():
            dx = -pixel_delta.x()
            dy = -pixel_delta.y()
        else:
            dx = -angle_delta.x() / 2.0
            dy = -angle_delta.y() / 2.0

            if (
                event.modifiers()
                & Qt.KeyboardModifier.ShiftModifier
            ):
                dx = -angle_delta.y() / 2.0
                dy = 0.0

        self.pan_requested.emit(
            float(dx),
            float(dy),
        )

        event.accept()

    def event(self, event):
        if event.type() == QEvent.Type.Gesture:
            pinch = event.gesture(
                Qt.GestureType.PinchGesture
            )

            if pinch is not None:
                previous_scale = (
                    pinch.lastScaleFactor()
                )

                current_scale = (
                    pinch.scaleFactor()
                )

                if previous_scale > 0:
                    factor = (
                        current_scale
                        / previous_scale
                    )
                else:
                    factor = current_scale

                factor = clamp(
                    factor,
                    0.75,
                    1.25,
                )

                if abs(factor - 1.0) > 0.001:
                    self.zoom_requested.emit(
                        factor,
                        pinch.centerPoint(),
                    )

                event.accept()
                return True

        if event.type() == QEvent.Type.NativeGesture:
            try:
                if (
                    event.gestureType()
                    == Qt.NativeGestureType.ZoomNativeGesture
                ):
                    factor = math.exp(
                        float(event.value())
                    )

                    factor = clamp(
                        factor,
                        0.75,
                        1.25,
                    )

                    self.zoom_requested.emit(
                        factor,
                        event.position(),
                    )

                    event.accept()
                    return True

            except (AttributeError, TypeError):
                pass

        return super().event(event)

    # ========================================================
    # HIT TEST
    # ========================================================

    def bbox_corners(self, bbox):
        x1, y1, x2, y2 = bbox

        return {
            "tl": [x1, y1],
            "tr": [x2, y1],
            "bl": [x1, y2],
            "br": [x2, y2],
        }

    def hit_test_keypoint(self, point):
        radius = 16 / self.zoom

        for index in range(
            len(self.objects) - 1,
            -1,
            -1,
        ):
            obj = self.objects[index]

            for key in ("grasp", "angle"):
                keypoint = obj.get(key)

                if keypoint is None:
                    continue

                if (
                    point_distance(
                        point,
                        keypoint,
                    )
                    <= radius
                ):
                    return index, key

        return None, None

    def hit_test_corner(self, point):
        radius = 16 / self.zoom

        for index in range(
            len(self.objects) - 1,
            -1,
            -1,
        ):
            corners = self.bbox_corners(
                self.objects[index]["bbox"]
            )

            for name, corner in corners.items():
                if (
                    point_distance(
                        point,
                        corner,
                    )
                    <= radius
                ):
                    return index, name

        return None, None

    def hit_test_bbox_interior(self, point):
        for index in range(
            len(self.objects) - 1,
            -1,
            -1,
        ):
            x1, y1, x2, y2 = (
                self.objects[index]["bbox"]
            )

            if (
                x1 <= point[0] <= x2
                and y1 <= point[1] <= y2
            ):
                return index

        return None

    # ========================================================
    # SAĞ TUŞ DÜZENLEME
    # ========================================================

    def begin_edit_drag(
        self,
        index,
        drag_type,
        corner=None,
    ):
        if not (
            0 <= index < len(self.objects)
        ):
            return

        before_snapshot = (
            self.current_snapshot()
        )

        self.selected_index = index
        self.edit_drag_type = drag_type
        self.edit_drag_corner = corner
        self.edit_drag_index = index

        self.edit_drag_original_object = (
            copy.deepcopy(
                self.objects[index]
            )
        )

        self.edit_drag_before_snapshot = (
            before_snapshot
        )

        self.edit_drag_changed = False

        self.selection_changed.emit()
        self.update()

    def clear_edit_drag(self):
        self.edit_drag_type = None
        self.edit_drag_corner = None
        self.edit_drag_index = None
        self.edit_drag_original_object = None
        self.edit_drag_before_snapshot = None
        self.edit_drag_changed = False

    # ========================================================
    # FARE
    # ========================================================

    def mousePressEvent(self, event):
        if self.pixmap is None:
            return

        if not self.event_is_over_image(event):
            return

        point = self.image_position(event)

        if (
            event.button()
            == Qt.MouseButton.RightButton
        ):
            index, key_type = (
                self.hit_test_keypoint(point)
            )

            if index is not None:
                self.begin_edit_drag(
                    index,
                    key_type,
                )
                return

            index, corner = (
                self.hit_test_corner(point)
            )

            if index is not None:
                self.begin_edit_drag(
                    index,
                    "corner",
                    corner,
                )
                return

            return

        if (
            event.button()
            != Qt.MouseButton.LeftButton
        ):
            return

        if self.mode == "box":
            self.box_start = point
            self.box_current = point
            self.update()
            return

        if self.mode == "grasp":
            index = self.active_annotation_index

            if index is None:
                return

            if not (
                0 <= index < len(self.objects)
            ):
                return

            self.push_snapshot()

            obj = self.objects[index]

            obj["grasp"] = (
                self.clamp_point_to_bbox(
                    point,
                    obj["bbox"],
                )
            )

            self.selected_index = index
            self.set_mode("angle")

            self.annotations_changed.emit()
            self.selection_changed.emit()
            self.update()
            return

        if self.mode == "angle":
            index = self.active_annotation_index

            if index is None:
                return

            if not (
                0 <= index < len(self.objects)
            ):
                return

            self.push_snapshot()

            obj = self.objects[index]

            obj["angle"] = (
                self.clamp_point_to_bbox(
                    point,
                    obj["bbox"],
                )
            )

            self.active_annotation_index = None
            self.selected_index = None

            self.set_mode("box")

            self.annotations_changed.emit()
            self.selection_changed.emit()
            self.update()
            return

        if self.mode == "select":
            self.selected_index = (
                self.hit_test_bbox_interior(
                    point
                )
            )

            self.selection_changed.emit()
            self.update()

    def mouseMoveEvent(self, event):
        if self.pixmap is None:
            return

        point = self.image_position(event)

        if (
            self.mode == "box"
            and self.box_start is not None
            and (
                event.buttons()
                & Qt.MouseButton.LeftButton
            )
        ):
            self.box_current = point
            self.update()
            return

        if self.edit_drag_type is None:
            return

        if not (
            event.buttons()
            & Qt.MouseButton.RightButton
        ):
            return

        index = self.edit_drag_index

        if (
            index is None
            or not 0 <= index < len(self.objects)
        ):
            return

        original = (
            self.edit_drag_original_object
        )

        if original is None:
            return

        obj = self.objects[index]

        if self.edit_drag_type == "grasp":
            obj["grasp"] = (
                self.clamp_point_to_bbox(
                    point,
                    obj["bbox"],
                )
            )

        elif self.edit_drag_type == "angle":
            obj["angle"] = (
                self.clamp_point_to_bbox(
                    point,
                    obj["bbox"],
                )
            )

        elif self.edit_drag_type == "corner":
            bbox = list(original["bbox"])
            x, y = point

            if self.edit_drag_corner == "tl":
                bbox[0] = x
                bbox[1] = y

            elif self.edit_drag_corner == "tr":
                bbox[2] = x
                bbox[1] = y

            elif self.edit_drag_corner == "bl":
                bbox[0] = x
                bbox[3] = y

            elif self.edit_drag_corner == "br":
                bbox[2] = x
                bbox[3] = y

            obj["bbox"] = normalize_bbox(
                bbox
            )

            self.clamp_object_keypoints(obj)

        if obj != original:
            self.edit_drag_changed = True

        self.update()

    def mouseReleaseEvent(self, event):
        if (
            event.button()
            == Qt.MouseButton.LeftButton
            and self.mode == "box"
            and self.box_start is not None
        ):
            point = self.image_position(event)

            bbox = normalize_bbox([
                self.box_start[0],
                self.box_start[1],
                point[0],
                point[1],
            ])

            self.box_start = None
            self.box_current = None

            if (
                bbox[2] - bbox[0] < 5
                or bbox[3] - bbox[1] < 5
            ):
                self.update()
                return

            self.push_snapshot()

            self.objects.append({
                "bbox": bbox,
                "grasp": None,
                "angle": None,
                "source": "human",
                "reviewed": True,
                "confidence": None,
                "keypoint_confidence": None,
                "model_path": None,
            })

            new_index = (
                len(self.objects) - 1
            )

            self.selected_index = new_index
            self.active_annotation_index = (
                new_index
            )

            self.set_mode("grasp")

            self.annotations_changed.emit()
            self.selection_changed.emit()
            self.update()
            return

        if (
            event.button()
            == Qt.MouseButton.RightButton
            and self.edit_drag_type is not None
        ):
            changed = self.edit_drag_changed
            snapshot = (
                self.edit_drag_before_snapshot
            )

            if changed and snapshot is not None:
                index = self.edit_drag_index

                if (
                    index is not None
                    and 0 <= index < len(self.objects)
                ):
                    self.mark_object_as_edited(
                        self.objects[index]
                    )

                self.push_snapshot(snapshot)
                self.annotations_changed.emit()


            self.clear_edit_drag()

            self.selection_changed.emit()
            self.update()

    # ========================================================
    # ÇİZİM
    # ========================================================

    def paintEvent(self, event):
        painter = QPainter(self)

        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing,
            True,
        )

        painter.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform,
            True,
        )

        painter.fillRect(
            self.rect(),
            COLORS["background"],
        )

        if self.pixmap is None:
            return

        painter.translate(
            self.workspace_margin_x,
            self.workspace_margin_y,
        )

        painter.scale(
            self.zoom,
            self.zoom,
        )

        painter.drawPixmap(
            0,
            0,
            self.pixmap,
        )

        for index, obj in enumerate(
            self.objects
        ):
            self.draw_object(
                painter,
                obj,
                index,
            )

        if (
            self.mode == "box"
            and self.box_start is not None
            and self.box_current is not None
        ):
            bbox = normalize_bbox([
                self.box_start[0],
                self.box_start[1],
                self.box_current[0],
                self.box_current[1],
            ])

            pen = QPen(
                COLORS["draft"]
            )

            pen.setWidthF(
                2 / self.zoom
            )

            pen.setStyle(
                Qt.PenStyle.DashLine
            )

            painter.setPen(pen)
            painter.setBrush(
                Qt.BrushStyle.NoBrush
            )

            painter.drawRect(
                QRectF(
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                )
            )

    def draw_object(
        self,
        painter,
        obj,
        index,
    ):
        x1, y1, x2, y2 = obj["bbox"]

        is_active = (
            index
            == self.active_annotation_index
        )

        is_selected = (
            index
            == self.selected_index
        )

        source = obj.get(
            "source",
            "human",
        )

        reviewed = bool(
            obj.get("reviewed", True)
        )

        is_model_prediction = (
            source == "model"
            and not reviewed
        )

        is_edited_model = (
            source == "model_edited"
            and not reviewed
        )

        if is_active:
            color = COLORS["active_bbox"]
        elif is_selected:
            color = COLORS["selected_bbox"]
        elif is_model_prediction:
            color = COLORS["model_bbox"]
        elif is_edited_model:
            color = COLORS["active_bbox"]
        else:
            color = COLORS["bbox"]


        bbox_pen = QPen(color)

        bbox_pen.setWidthF(
            (
                3
                if is_active or is_selected
                else 2
            )
            / self.zoom
        )

        painter.setPen(bbox_pen)

        if is_model_prediction:
            painter.setBrush(
                COLORS["model_fill"]
            )
        elif (
            is_edited_model
            or is_active
            or is_selected
        ):
            painter.setBrush(
                COLORS["active_fill"]
            )
        else:
            painter.setBrush(
                COLORS["human_fill"]
            )

        painter.drawRect(
            QRectF(
                x1,
                y1,
                x2 - x1,
                y2 - y1,
            )
        )

        font = painter.font()

        font.setPixelSize(
            max(
                10,
                int(15 / self.zoom),
            )
        )

        painter.setFont(font)
        painter.setPen(color)

        label = CLASS_NAME

        if is_model_prediction:
            confidence = obj.get(
                "confidence"
            )

            if confidence is None:
                label = (
                    f"{CLASS_NAME} [TAHMİN]"
                )
            else:
                label = (
                    f"{CLASS_NAME} [TAHMİN "
                    f"{confidence:.2f}]"
                )
        elif is_edited_model:
            label = (
                f"{CLASS_NAME} [DÜZENLENDİ]"
            )

        painter.drawText(
            QPointF(
                x1 + 4,
                max(15, y1 - 5),
            ),
            label,
        )

        grasp = obj.get("grasp")
        angle = obj.get("angle")

        if (
            grasp is not None
            and angle is not None
        ):
            edge_pen = QPen(
                COLORS["edge"]
            )

            edge_pen.setWidthF(
                3 / self.zoom
            )

            painter.setPen(edge_pen)

            painter.drawLine(
                QPointF(
                    grasp[0],
                    grasp[1],
                ),
                QPointF(
                    angle[0],
                    angle[1],
                ),
            )

            self.draw_arrow_head(
                painter,
                grasp,
                angle,
            )

        if grasp is not None:
            self.draw_keypoint(
                painter,
                grasp,
                COLORS["sap_ucu"],
                "sap_ucu",
            )

        if angle is not None:
            self.draw_keypoint(
                painter,
                angle,
                COLORS["aci_noktasi"],
                "aci_noktasi",
            )

        for corner in self.bbox_corners(
            obj["bbox"]
        ).values():
            self.draw_handle(
                painter,
                corner,
                active=(
                    is_active
                    or is_selected
                ),
            )

    def draw_handle(
        self,
        painter,
        point,
        active,
    ):
        size = (
            11 / self.zoom
            if active
            else 8 / self.zoom
        )

        painter.setPen(
            QPen(
                QColor("#000000"),
                1 / self.zoom,
            )
        )

        painter.setBrush(
            COLORS["active_handle"]
            if active
            else COLORS["handle"]
        )

        painter.drawRect(
            QRectF(
                point[0] - size / 2,
                point[1] - size / 2,
                size,
                size,
            )
        )

    def draw_keypoint(
        self,
        painter,
        point,
        color,
        text,
    ):
        radius = 7 / self.zoom

        painter.setPen(
            QPen(
                COLORS["text"],
                1.5 / self.zoom,
            )
        )

        painter.setBrush(color)

        painter.drawEllipse(
            QPointF(
                point[0],
                point[1],
            ),
            radius,
            radius,
        )

        font = painter.font()

        font.setPixelSize(
            max(
                9,
                int(13 / self.zoom),
            )
        )

        painter.setFont(font)
        painter.setPen(color)

        painter.drawText(
            QPointF(
                point[0]
                + 10 / self.zoom,
                point[1]
                - 8 / self.zoom,
            ),
            text,
        )

    def draw_arrow_head(
        self,
        painter,
        start,
        end,
    ):
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        length = math.hypot(dx, dy)

        if length < 1:
            return

        unit_x = dx / length
        unit_y = dy / length

        arrow_length = 14 / self.zoom
        arrow_width = 7 / self.zoom

        left = [
            end[0]
            - unit_x * arrow_length
            - unit_y * arrow_width,
            end[1]
            - unit_y * arrow_length
            + unit_x * arrow_width,
        ]

        right = [
            end[0]
            - unit_x * arrow_length
            + unit_y * arrow_width,
            end[1]
            - unit_y * arrow_length
            - unit_x * arrow_width,
        ]

        painter.setPen(
            Qt.PenStyle.NoPen
        )

        painter.setBrush(
            COLORS["edge"]
        )

        painter.drawPolygon(
            QPolygonF([
                QPointF(
                    end[0],
                    end[1],
                ),
                QPointF(
                    left[0],
                    left[1],
                ),
                QPointF(
                    right[0],
                    right[1],
                ),
            ])
        )


# ============================================================
# ANA PENCERE
# ============================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "Robotik Fidan Hasadı - YOLO11 Pose Aktif Öğrenme"
        )

        self.resize(1550, 950)

        self.dataset_root = None
        self.image_paths = []
        self.current_index = -1

        self.database = (
            self.empty_database()
        )

        self.loaded_model_path = None

        self.inference_thread = None
        self.inference_worker = None
        self.inference_running = False

        self.training_process = None
        self.training_run_directory = None

        self.canvas = AnnotationCanvas()

        self.scroll = QScrollArea()
        self.scroll.setWidget(self.canvas)
        self.scroll.setWidgetResizable(False)

        self.scroll.setAlignment(
            Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignTop
        )

        self.mode_label = QLabel(
            "Mod: görüntü klasörü bekleniyor"
        )

        self.status_label = QLabel(
            "Bir görüntü klasörü açın."
        )

        self.help_label = QLabel(
            "Sol tuş: kutu → sap_ucu → aci_noktasi | "
            "Sağ sürükle: nokta/köşe düzenle | "
            "S: seçim | K: noktaları yenile | "
            "A/D: fotoğraf | Delete: seçiliyi sil | "
            "X: son şablon | "
            "Mor: onaylanmamış model tahmini"
        )

        self.help_label.setWordWrap(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        self.training_log = QPlainTextEdit()
        self.training_log.setReadOnly(True)

        self.training_log.setMaximumBlockCount(
            5000
        )

        self.training_log.setMaximumHeight(
            220
        )

        self.training_log.setVisible(False)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(
            self.mode_label
        )
        bottom_layout.addStretch()
        bottom_layout.addWidget(
            self.status_label
        )

        main_layout = QVBoxLayout()
        main_layout.addWidget(
            self.help_label
        )
        main_layout.addWidget(
            self.scroll,
            1,
        )
        main_layout.addWidget(
            self.progress_bar
        )
        main_layout.addWidget(
            self.training_log
        )
        main_layout.addLayout(
            bottom_layout
        )

        central_widget = QWidget()
        central_widget.setLayout(
            main_layout
        )

        self.setCentralWidget(
            central_widget
        )

        self.create_toolbar()
        self.create_shortcuts()

        self.canvas.annotations_changed.connect(
            self.on_annotations_changed
        )

        self.canvas.mode_changed.connect(
            self.update_mode_label
        )

        self.canvas.selection_changed.connect(
            self.update_status
        )

        self.canvas.pan_requested.connect(
            self.pan_canvas
        )

        self.canvas.zoom_requested.connect(
            self.zoom_canvas_at_position
        )

    # ========================================================
    # VERİTABANI
    # ========================================================

    def empty_database(self):
        return {
            "version": 8,
            "class_id": CLASS_ID,
            "class_name": CLASS_NAME,
            "keypoint_names": (
                KEYPOINT_NAMES
            ),
            "last_image": None,
            "last_model_path": None,
            "images": {},
        }

    # ========================================================
    # TOOLBAR
    # ========================================================

    def create_toolbar(self):
        toolbar = QToolBar(
            "Araçlar"
        )

        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        self.add_toolbar_button(
            toolbar,
            "Klasör Aç",
            self.open_folder,
        )

        self.add_toolbar_button(
            toolbar,
            "←",
            self.previous_image,
        )

        self.add_toolbar_button(
            toolbar,
            "→",
            self.next_image,
        )

        toolbar.addSeparator()

        self.add_toolbar_button(
            toolbar,
            "Yeni Kutu [B]",
            lambda: self.canvas.set_mode(
                "box"
            ),
        )

        self.add_toolbar_button(
            toolbar,
            "Seçim [S]",
            lambda: self.canvas.set_mode(
                "select"
            ),
        )

        self.add_toolbar_button(
            toolbar,
            "Noktaları Yenile [K]",
            self.canvas.restart_points_for_selected,
        )

        self.add_toolbar_button(
            toolbar,
            "Sil [Delete]",
            self.canvas.delete_selected,
        )

        self.add_toolbar_button(
            toolbar,
            "Sonu Sil [X]",
            self.canvas.delete_last_template,
        )

        toolbar.addSeparator()

        self.add_toolbar_button(
            toolbar,
            "Geri",
            self.canvas.undo,
        )

        self.add_toolbar_button(
            toolbar,
            "İleri",
            self.canvas.redo,
        )

        self.add_toolbar_button(
            toolbar,
            "−",
            self.canvas.zoom_out,
        )

        self.add_toolbar_button(
            toolbar,
            "100%",
            self.canvas.reset_zoom,
        )

        self.add_toolbar_button(
            toolbar,
            "+",
            self.canvas.zoom_in,
        )

        self.add_toolbar_button(
            toolbar,
            "Sığdır",
            self.fit_image,
        )
        self.add_toolbar_button(
            toolbar,
            "Bu Fotoğrafı Ön Etiketle",
            self.prelabel_current_image,
        )
        self.add_toolbar_button(
            toolbar,
            "Onayla ve Sonraki",
            self.approve_current_and_next,
        )
        toolbar.addSeparator()
        self.add_toolbar_button(
            toolbar,
            "Log",
            self.toggle_log,
        )
        self.add_toolbar_button(
            toolbar,
            "YOLO Kutularını İçe Aktar",
            self.import_yolo_boxes,
        )

        self.add_toolbar_button(
            toolbar,
            "Dışa Aktar",
            self.export_dataset,
        )

        toolbar.addSeparator()

        self.add_toolbar_button(
            toolbar,
            "Model Yükle",
            self.load_pose_model,
        )

        self.confidence_spin = (
            QDoubleSpinBox()
        )

        self.confidence_spin.setRange(
            0.01,
            0.99,
        )

        self.confidence_spin.setDecimals(
            2
        )

        self.confidence_spin.setSingleStep(
            0.05
        )

        self.confidence_spin.setValue(
            DEFAULT_CONFIDENCE
        )

        self.confidence_spin.setPrefix(
            "Güven "
        )

        toolbar.addWidget(
            self.confidence_spin
        )

     

        self.add_toolbar_button(
            toolbar,
            "Etiketsizleri Ön Etiketle",
            self.prelabel_all_unlabeled,
        )


        self.add_toolbar_button(
            toolbar,
            "YOLO11m Eğit",
            self.start_yolo11_training,
        )

        

    def add_toolbar_button(
        self,
        toolbar,
        text,
        callback,
    ):
        button = QPushButton(text)
        button.clicked.connect(callback)
        toolbar.addWidget(button)

    # ========================================================
    # KISAYOLLAR
    # ========================================================

    def add_shortcut(
        self,
        sequence,
        callback,
    ):
        action = QAction(self)

        action.setShortcut(
            QKeySequence(sequence)
        )

        action.setShortcutContext(
            Qt.ShortcutContext.ApplicationShortcut
        )

        action.triggered.connect(
            callback
        )

        self.addAction(action)

    def create_shortcuts(self):
        shortcuts = [
            (
                "B",
                lambda: self.canvas.set_mode(
                    "box"
                ),
            ),
            (
                "S",
                lambda: self.canvas.set_mode(
                    "select"
                ),
            ),
            (
                "K",
                self.canvas.restart_points_for_selected,
            ),
            (
                "A",
                self.previous_image,
            ),
            (
                "D",
                self.next_image,
            ),
            (
                "Left",
                self.previous_image,
            ),
            (
                "Right",
                self.next_image,
            ),
            (
                "Ctrl+Z",
                self.canvas.undo,
            ),
            (
                "Ctrl+Y",
                self.canvas.redo,
            ),
            (
                "Ctrl+Shift+Z",
                self.canvas.redo,
            ),
            (
                "Delete",
                self.canvas.delete_selected,
            ),
            (
                "Backspace",
                self.canvas.delete_selected,
            ),
            (
                "X",
                self.canvas.delete_last_template,
            ),
            (
                "Escape",
                self.canvas.cancel_active_operation,
            ),
            (
                "Ctrl++",
                self.canvas.zoom_in,
            ),
            (
                "Ctrl+=",
                self.canvas.zoom_in,
            ),
            (
                "Ctrl+-",
                self.canvas.zoom_out,
            ),
            (
                "Ctrl+0",
                self.canvas.reset_zoom,
            ),
            (
                "Ctrl+S",
                self.save_current_image,
            ),
            (
                "Ctrl+E",
                self.export_dataset,
            ),
        ]

        for sequence, callback in shortcuts:
            self.add_shortcut(
                sequence,
                callback,
            )

    # ========================================================
    # PAN / ZOOM
    # ========================================================

    def update_workspace_margins(self):
        viewport = (
            self.scroll.viewport()
        )

        self.canvas.set_workspace_margins(
            viewport.width(),
            viewport.height(),
        )

    def center_image_in_viewport(self):
        if self.canvas.pixmap is None:
            return

        viewport = (
            self.scroll.viewport()
        )

        scaled_width = (
            self.canvas.image_width
            * self.canvas.zoom
        )

        scaled_height = (
            self.canvas.image_height
            * self.canvas.zoom
        )

        horizontal = (
            self.scroll.horizontalScrollBar()
        )

        vertical = (
            self.scroll.verticalScrollBar()
        )

        horizontal.setValue(
            int(
                self.canvas.workspace_margin_x
                - max(
                    0,
                    (
                        viewport.width()
                        - scaled_width
                    )
                    / 2,
                )
            )
        )

        vertical.setValue(
            int(
                self.canvas.workspace_margin_y
                - max(
                    0,
                    (
                        viewport.height()
                        - scaled_height
                    )
                    / 2,
                )
            )
        )

    def pan_canvas(self, dx, dy):
        horizontal = (
            self.scroll.horizontalScrollBar()
        )

        vertical = (
            self.scroll.verticalScrollBar()
        )

        horizontal.setValue(
            horizontal.value()
            + int(dx)
        )

        vertical.setValue(
            vertical.value()
            + int(dy)
        )

    def zoom_canvas_at_position(
        self,
        factor,
        canvas_position,
    ):
        old_zoom = self.canvas.zoom

        if old_zoom <= 0:
            return

        horizontal = (
            self.scroll.horizontalScrollBar()
        )

        vertical = (
            self.scroll.verticalScrollBar()
        )

        old_horizontal = (
            horizontal.value()
        )

        old_vertical = (
            vertical.value()
        )

        image_x = (
            canvas_position.x()
            - self.canvas.workspace_margin_x
        ) / old_zoom

        image_y = (
            canvas_position.y()
            - self.canvas.workspace_margin_y
        ) / old_zoom

        viewport_x = (
            canvas_position.x()
            - old_horizontal
        )

        viewport_y = (
            canvas_position.y()
            - old_vertical
        )

        self.canvas.set_zoom(
            old_zoom * factor
        )

        horizontal.setValue(
            int(
                self.canvas.workspace_margin_x
                + image_x
                * self.canvas.zoom
                - viewport_x
            )
        )

        vertical.setValue(
            int(
                self.canvas.workspace_margin_y
                + image_y
                * self.canvas.zoom
                - viewport_y
            )
        )

    def fit_image(self):
        if self.canvas.pixmap is None:
            return

        self.update_workspace_margins()

        size = (
            self.scroll.viewport().size()
        )

        self.canvas.fit_zoom(
            max(
                100,
                size.width() - 30,
            ),
            max(
                100,
                size.height() - 30,
            ),
        )

        self.center_image_in_viewport()

    # ========================================================
    # KLASÖR / GÖRÜNTÜ
    # ========================================================

    def open_folder(self):
        selected = (
            QFileDialog.getExistingDirectory(
                self,
                "Görüntü klasörünü seçin",
            )
        )

        if not selected:
            return

        self.save_current_image()

        root = Path(selected)

        image_paths = sorted(
            [
                path
                for path in root.rglob("*")
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in IMAGE_EXTENSIONS
                    and TRAINING_FOLDER
                    not in path.parts
                )
            ],
            key=lambda path: str(
                path.relative_to(root)
            ).lower(),
        )

        if not image_paths:
            QMessageBox.warning(
                self,
                "Görüntü bulunamadı",
                "Seçilen klasörde görüntü bulunamadı.",
            )
            return

        self.dataset_root = root
        self.image_paths = image_paths

        autosave_path = (
            root / AUTOSAVE_FILENAME
        )

        if autosave_path.exists():
            try:
                self.database = json.loads(
                    autosave_path.read_text(
                        encoding="utf-8"
                    )
                )

            except Exception as error:
                QMessageBox.warning(
                    self,
                    "Kayıt okunamadı",
                    str(error),
                )

                self.database = (
                    self.empty_database()
                )
        else:
            self.database = (
                self.empty_database()
            )

        self.database.setdefault(
            "images",
            {},
        )

        self.database.setdefault(
            "last_image",
            None,
        )

        self.database.setdefault(
            "last_model_path",
            None,
        )

        self.database["version"] = 8
        self.database["class_id"] = (
            CLASS_ID
        )
        self.database["class_name"] = (
            CLASS_NAME
        )
        self.database["keypoint_names"] = (
            KEYPOINT_NAMES
        )

        saved_model = self.database.get(
            "last_model_path"
        )

        if (
            saved_model
            and Path(saved_model).exists()
        ):
            self.loaded_model_path = (
                saved_model
            )

        self.current_index = 0

        last_image = self.database.get(
            "last_image"
        )

        if last_image:
            for index, image_path in enumerate(
                self.image_paths
            ):
                if (
                    self.image_key(image_path)
                    == last_image
                ):
                    self.current_index = index
                    break

        self.load_current_image()
        self.fit_image()

    def image_key(self, image_path):
        return image_path.relative_to(
            self.dataset_root
        ).as_posix()

    def load_current_image(self):
        if not self.image_paths:
            return

        image_path = self.image_paths[
            self.current_index
        ]

        key = self.image_key(
            image_path
        )

        objects = (
            self.database
            .get("images", {})
            .get(key, {})
            .get("objects", [])
        )

        try:
            self.canvas.load_image(
                image_path,
                objects,
            )

        except Exception as error:
            QMessageBox.critical(
                self,
                "Görüntü açılamadı",
                str(error),
            )
            return

        self.update_workspace_margins()
        self.center_image_in_viewport()

        self.database["last_image"] = key
        self.save_database()
        self.update_status()

    # ========================================================
    # KAYIT
    # ========================================================

    def clean_object(self, obj):
        obj = normalize_object_metadata(
            obj
        )

        return {
            "bbox": [
                round(float(value), 3)
                for value in obj["bbox"]
            ],
            "grasp": (
                [
                    round(float(value), 3)
                    for value in obj["grasp"]
                ]
                if obj.get("grasp")
                is not None
                else None
            ),
            "angle": (
                [
                    round(float(value), 3)
                    for value in obj["angle"]
                ]
                if obj.get("angle")
                is not None
                else None
            ),
            "source": obj.get(
                "source",
                "human",
            ),
            "reviewed": bool(
                obj.get(
                    "reviewed",
                    True,
                )
            ),
            "confidence": (
                round(
                    float(
                        obj["confidence"]
                    ),
                    6,
                )
                if obj.get("confidence")
                is not None
                else None
            ),
            "keypoint_confidence": (
                round(
                    float(
                        obj[
                            "keypoint_confidence"
                        ]
                    ),
                    6,
                )
                if obj.get(
                    "keypoint_confidence"
                )
                is not None
                else None
            ),
            "model_path": obj.get(
                "model_path"
            ),
        }

    def save_current_image(self):
        if (
            self.dataset_root is None
            or not self.image_paths
            or self.current_index < 0
        ):
            return

        image_path = self.image_paths[
            self.current_index
        ]

        key = self.image_key(
            image_path
        )

        previous_record = (
            self.database
            .get("images", {})
            .get(key, {})
        )

        objects = [
            self.clean_object(obj)
            for obj
            in self.canvas.get_objects()
        ]

        self.database["last_image"] = key

        self.database["images"][key] = {
            "width": (
                self.canvas.image_width
            ),
            "height": (
                self.canvas.image_height
            ),
            "objects": objects,
            "image_reviewed": bool(
                previous_record.get(
                    "image_reviewed",
                    False,
                )
            ),
        }

        self.save_database()

    def save_database(self):
        if self.dataset_root is None:
            return

        output_path = (
            self.dataset_root
            / AUTOSAVE_FILENAME
        )

        temporary_path = (
            self.dataset_root
            / f"{AUTOSAVE_FILENAME}.tmp"
        )

        try:
            temporary_path.write_text(
                json.dumps(
                    self.database,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            temporary_path.replace(
                output_path
            )

        except Exception as error:
            QMessageBox.warning(
                self,
                "Kaydetme hatası",
                str(error),
            )

    def on_annotations_changed(self):
        self.save_current_image()
        self.update_status()

    # ========================================================
    # FOTOĞRAF GEÇİŞİ
    # ========================================================

    def previous_image(self):
        if (
            not self.image_paths
            or self.current_index <= 0
        ):
            return

        self.save_current_image()
        self.current_index -= 1
        self.load_current_image()

    def next_image(self):
        if not self.image_paths:
            return

        if (
            self.current_index
            >= len(self.image_paths) - 1
        ):
            return

        self.save_current_image()
        self.current_index += 1
        self.load_current_image()

    # ========================================================
    # DURUM
    # ========================================================

    def update_mode_label(self, mode):
        names = {
            "box": "yeni kutu çiz",
            "grasp": "sap_ucu koy",
            "angle": "aci_noktasi koy",
            "select": "seçim/düzenleme",
        }

        self.mode_label.setText(
            f"Mod: {names.get(mode, mode)}"
        )

    def update_status(self):
        if not self.image_paths:
            return

        current_objects = (
            self.canvas.get_objects()
        )

        complete = sum(
            1
            for obj in current_objects
            if (
                obj.get("grasp")
                is not None
                and obj.get("angle")
                is not None
            )
        )

        pending = sum(
            1
            for obj in current_objects
            if not obj.get(
                "reviewed",
                True,
            )
        )

        total_reviewed = 0
        reviewed_images = 0

        for record in self.database.get(
            "images",
            {},
        ).values():
            objects = record.get(
                "objects",
                [],
            )

            total_reviewed += sum(
                1
                for obj in objects
                if (
                    obj.get("grasp")
                    is not None
                    and obj.get("angle")
                    is not None
                    and obj.get(
                        "reviewed",
                        True,
                    )
                )
            )

            if record.get(
                "image_reviewed",
                False,
            ):
                reviewed_images += 1

        current_path = self.image_paths[
            self.current_index
        ]

        model_name = (
            Path(
                self.loaded_model_path
            ).name
            if self.loaded_model_path
            else "yok"
        )

        self.status_label.setText(
            f"{self.current_index + 1}/"
            f"{len(self.image_paths)} | "
            f"{self.image_key(current_path)} | "
            f"Tamam: {complete} | "
            f"Onaysız tahmin: {pending} | "
            f"Onaylı görüntü: {reviewed_images} | "
            f"Onaylı fidan: {total_reviewed} | "
            f"Model: {model_name}"
        )

    # ========================================================
    # CUDA / LOG
    # ========================================================

    def append_log(self, message):
        self.training_log.setVisible(
            True
        )

        self.training_log.appendPlainText(
            str(message)
        )

        self.training_log.moveCursor(
            QTextCursor.MoveOperation.End
        )

    def toggle_log(self):
        self.training_log.setVisible(
            not self.training_log.isVisible()
        )

    def check_cuda(self):
        try:
            import torch

            if not torch.cuda.is_available():
                QMessageBox.critical(
                    self,
                    "CUDA bulunamadı",
                    "CUDA bu Python ortamında aktif değil.\n\n"
                    "Uygulamayı .venv içindeki python.exe "
                    "ile çalıştırın.",
                )
                return False

            self.append_log(
                f"CUDA: "
                f"{torch.cuda.get_device_name(0)} | "
                f"PyTorch: {torch.__version__} | "
                f"Runtime: {torch.version.cuda}"
            )

            return True

        except Exception as error:
            QMessageBox.critical(
                self,
                "CUDA kontrol hatası",
                str(error),
            )
            return False

    # ========================================================
    # MODEL YÜKLEME
    # ========================================================

    def load_pose_model(self):
        selected_path, _ = (
            QFileDialog.getOpenFileName(
                self,
                "Fidan YOLO Pose best.pt dosyasını seçin",
                str(
                    self.dataset_root
                    if self.dataset_root
                    else Path.cwd()
                ),
                "PyTorch ağırlığı (*.pt)",
            )
        )

        if not selected_path:
            return

        if not self.check_cuda():
            return

        self.loaded_model_path = str(
            Path(
                selected_path
            ).resolve()
        )

        if self.dataset_root is not None:
            self.database[
                "last_model_path"
            ] = self.loaded_model_path

            self.save_database()

        self.append_log(
            f"Etkin model: "
            f"{self.loaded_model_path}"
        )

        self.update_status()

    def ensure_pose_model(self):
        if not self.loaded_model_path:
            QMessageBox.information(
                self,
                "Model yükleyin",
                "Önce eğitilmiş best.pt dosyasını "
                "'Model Yükle' ile seçin.",
            )
            return False

        if not Path(
            self.loaded_model_path
        ).exists():
            QMessageBox.warning(
                self,
                "Model bulunamadı",
                self.loaded_model_path,
            )
            return False

        return self.check_cuda()

    # ========================================================
    # ÖN ETİKETLEME
    # ========================================================

    def prelabel_current_image(self):
        if not self.image_paths:
            return

        if not self.ensure_pose_model():
            return

        self.save_current_image()

        image_path = self.image_paths[
            self.current_index
        ]

        key = self.image_key(
            image_path
        )

        existing = (
            self.database
            .get("images", {})
            .get(key, {})
            .get("objects", [])
        )

        if existing:
            answer = QMessageBox.question(
                self,
                "Mevcut işaretler var",
                "Bu fotoğraftaki mevcut işaretler model "
                "tahminleriyle değiştirilsin mi?",
                (
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No
                ),
            )

            if (
                answer
                != QMessageBox.StandardButton.Yes
            ):
                return

        self.start_inference([
            (
                key,
                image_path,
            ),
        ])

    def image_is_unlabeled(self, key):
        record = (
            self.database
            .get("images", {})
            .get(key)
        )

        if not record:
            return True

        if record.get(
            "image_reviewed",
            False,
        ):
            return False

        objects = record.get(
            "objects",
            [],
        )

        return len(objects) == 0

    def prelabel_all_unlabeled(self):
        if not self.image_paths:
            return

        if not self.ensure_pose_model():
            return

        self.save_current_image()

        items = []

        for image_path in self.image_paths:
            key = self.image_key(
                image_path
            )

            if self.image_is_unlabeled(
                key
            ):
                items.append(
                    (
                        key,
                        image_path,
                    )
                )

        if not items:
            QMessageBox.information(
                self,
                "Etiketsiz görüntü yok",
                "Ön etiketlenecek boş görüntü bulunamadı.",
            )
            return

        answer = QMessageBox.question(
            self,
            "Toplu ön etiketleme",
            f"{len(items)} görüntü, "
            f"{PREDICT_IMAGE_SIZE}px çözünürlükte "
            "ön etiketlenecek.\n\n"
            "Tahminler mor gösterilecek ve insan onayı "
            "verilene kadar eğitime katılmayacak.\n\n"
            "Başlatılsın mı?",
            (
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
            ),
        )

        if (
            answer
            != QMessageBox.StandardButton.Yes
        ):
            return

        self.start_inference(items)

    def start_inference(self, items):
        if self.inference_running:
            QMessageBox.information(
                self,
                "İşlem sürüyor",
                "Ön etiketleme zaten çalışıyor.",
            )
            return

        self.inference_running = True

        self.progress_bar.setVisible(
            True
        )

        self.progress_bar.setRange(
            0,
            len(items),
        )

        self.progress_bar.setValue(0)

        thread = QThread(self)

        worker = PoseInferenceWorker(
            self.loaded_model_path,
            items,
            self.confidence_spin.value(),
        )

        worker.moveToThread(thread)

        thread.started.connect(
            worker.run
        )

        worker.progress.connect(
            self.on_inference_progress
        )

        worker.prediction_ready.connect(
            self.on_prediction_ready
        )

        worker.completed.connect(
            self.on_inference_completed
        )

        worker.failed.connect(
            self.on_inference_failed
        )

        worker.completed.connect(
            thread.quit
        )

        worker.failed.connect(
            thread.quit
        )

        thread.finished.connect(
            worker.deleteLater
        )

        thread.finished.connect(
            thread.deleteLater
        )

        self.inference_thread = thread
        self.inference_worker = worker

        self.append_log(
            f"Ön etiketleme başladı | "
            f"Görüntü: {len(items)} | "
            f"imgsz: {PREDICT_IMAGE_SIZE} | "
            f"Güven: "
            f"{self.confidence_spin.value():.2f}"
        )

        thread.start()

    def on_inference_progress(
        self,
        current,
        total,
        filename,
    ):
        self.progress_bar.setRange(
            0,
            total,
        )

        self.progress_bar.setValue(
            current
        )

        self.append_log(
            f"[{current}/{total}] "
            f"{filename}"
        )

    def on_prediction_ready(
        self,
        key,
        predictions,
    ):
        image_path = next(
            (
                path
                for path in self.image_paths
                if self.image_key(path)
                == key
            ),
            None,
        )

        if image_path is None:
            return

        reader = QImageReader(
            str(image_path)
        )

        size = reader.size()

        self.database["images"][key] = {
            "width": size.width(),
            "height": size.height(),
            "objects": [
                self.clean_object(obj)
                for obj in predictions
            ],
            "image_reviewed": False,
        }

        self.save_database()

        current_key = self.image_key(
            self.image_paths[
                self.current_index
            ]
        )

        if key == current_key:
            self.canvas.load_image(
                image_path,
                predictions,
            )

            self.update_workspace_margins()
            self.center_image_in_viewport()

        self.update_status()

    def on_inference_completed(self):
        self.inference_running = False
        self.progress_bar.setVisible(False)

        self.append_log(
            "Ön etiketleme tamamlandı. "
            "Mor tahminleri kontrol edin."
        )

        self.inference_worker = None
        self.inference_thread = None

        self.load_current_image()

    def on_inference_failed(self, message):
        self.inference_running = False
        self.progress_bar.setVisible(False)

        self.append_log(
            f"Ön etiketleme hatası: "
            f"{message}"
        )

        QMessageBox.critical(
            self,
            "Ön etiketleme hatası",
            message,
        )

        self.inference_worker = None
        self.inference_thread = None

    # ========================================================
    # ONAY
    # ========================================================

    def approve_current_and_next(self):
        if not self.image_paths:
            return

        incomplete = [
            obj
            for obj in self.canvas.objects
            if (
                obj.get("grasp")
                is None
                or obj.get("angle")
                is None
            )
        ]

        if incomplete:
            QMessageBox.warning(
                self,
                "Eksik işaret var",
                f"{len(incomplete)} nesnede iki "
                "keypoint tamamlanmamış.",
            )
            return

        for obj in self.canvas.objects:
            if obj.get("source") in {
                "model",
                "model_edited",
            }:
                obj["source"] = (
                    "model_reviewed"
                )


            obj["reviewed"] = True

        image_path = self.image_paths[
            self.current_index
        ]

        key = self.image_key(
            image_path
        )

        self.database["images"][key] = {
            "width": (
                self.canvas.image_width
            ),
            "height": (
                self.canvas.image_height
            ),
            "objects": [
                self.clean_object(obj)
                for obj
                in self.canvas.objects
            ],
            "image_reviewed": True,
        }

        self.save_database()
        self.update_status()

        if (
            self.current_index
            < len(self.image_paths) - 1
        ):
            self.current_index += 1
            self.load_current_image()
        else:
            QMessageBox.information(
                self,
                "Tamamlandı",
                "Son görüntü de onaylandı.",
            )

    # ========================================================
    # YOLO RECTANGLE İÇE AKTARMA
    # ========================================================

    def import_yolo_boxes(self):
        if self.dataset_root is None:
            QMessageBox.information(
                self,
                "Önce klasör açın",
                "Önce görüntü klasörünü açın.",
            )
            return

        selected_folder = (
            QFileDialog.getExistingDirectory(
                self,
                "YOLO rectangle labels klasörünü seçin",
            )
        )

        if not selected_folder:
            return

        labels_folder = Path(
            selected_folder
        )

        label_lookup = {}

        for txt_path in labels_folder.rglob(
            "*.txt"
        ):
            label_lookup.setdefault(
                txt_path.stem.lower(),
                [],
            ).append(txt_path)

        answer = QMessageBox.question(
            self,
            "İçe aktarma",
            "Mevcut anotasyonlar rectangle kutularıyla "
            "değiştirilsin mi?\n\n"
            "Evet: üzerine yazar.\n"
            "Hayır: mevcut kutulara ekler.",
            (
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel
            ),
        )

        if (
            answer
            == QMessageBox.StandardButton.Cancel
        ):
            return

        replace_existing = (
            answer
            == QMessageBox.StandardButton.Yes
        )

        imported_images = 0
        imported_boxes = 0

        for image_path in self.image_paths:
            matches = label_lookup.get(
                image_path.stem.lower(),
                [],
            )

            if len(matches) != 1:
                continue

            reader = QImageReader(
                str(image_path)
            )

            size = reader.size()
            width = size.width()
            height = size.height()

            if width <= 0 or height <= 0:
                continue

            try:
                lines = matches[0].read_text(
                    encoding="utf-8"
                ).splitlines()
            except UnicodeDecodeError:
                lines = matches[0].read_text(
                    encoding="latin-1"
                ).splitlines()

            imported_objects = []

            for line in lines:
                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                try:
                    class_id = int(
                        float(parts[0])
                    )

                    x_center = float(
                        parts[1]
                    )

                    y_center = float(
                        parts[2]
                    )

                    box_width = float(
                        parts[3]
                    )

                    box_height = float(
                        parts[4]
                    )

                except ValueError:
                    continue

                if class_id != CLASS_ID:
                    continue

                x1 = (
                    x_center
                    - box_width / 2
                ) * width

                y1 = (
                    y_center
                    - box_height / 2
                ) * height

                x2 = (
                    x_center
                    + box_width / 2
                ) * width

                y2 = (
                    y_center
                    + box_height / 2
                ) * height

                bbox = normalize_bbox([
                    clamp(
                        x1,
                        0,
                        width,
                    ),
                    clamp(
                        y1,
                        0,
                        height,
                    ),
                    clamp(
                        x2,
                        0,
                        width,
                    ),
                    clamp(
                        y2,
                        0,
                        height,
                    ),
                ])

                if (
                    bbox[2] - bbox[0] < 1
                    or bbox[3] - bbox[1] < 1
                ):
                    continue

                imported_objects.append({
                    "bbox": bbox,
                    "grasp": None,
                    "angle": None,
                    "source": (
                        "imported_rectangle"
                    ),
                    "reviewed": False,
                    "confidence": None,
                    "keypoint_confidence": None,
                    "model_path": None,
                })

            if not imported_objects:
                continue

            key = self.image_key(
                image_path
            )

            record = self.database[
                "images"
            ].get(
                key,
                {
                    "width": width,
                    "height": height,
                    "objects": [],
                    "image_reviewed": False,
                },
            )

            if replace_existing:
                record["objects"] = (
                    imported_objects
                )
            else:
                record.setdefault(
                    "objects",
                    [],
                ).extend(
                    imported_objects
                )

            record["width"] = width
            record["height"] = height
            record[
                "image_reviewed"
            ] = False

            self.database[
                "images"
            ][key] = record

            imported_images += 1
            imported_boxes += len(
                imported_objects
            )

        self.save_database()
        self.load_current_image()

        QMessageBox.information(
            self,
            "İçe aktarma tamamlandı",
            f"{imported_images} görüntüden "
            f"{imported_boxes} kutu içe aktarıldı.\n\n"
            "Kutuyu S ile seçip K ile iki noktayı ekleyin.",
        )

    # ========================================================
    # EĞİTİM VERİSİ
    # ========================================================

    def collect_training_records(self):
        records = []

        for image_path in self.image_paths:
            key = self.image_key(
                image_path
            )

            record = (
                self.database
                .get("images", {})
                .get(key)
            )

            if not record:
                continue

            reviewed_objects = [
                copy.deepcopy(obj)
                for obj in record.get(
                    "objects",
                    [],
                )
                if (
                    obj.get("grasp")
                    is not None
                    and obj.get("angle")
                    is not None
                    and bool(
                        obj.get(
                            "reviewed",
                            True,
                        )
                    )
                )
            ]

            image_reviewed = bool(
                record.get(
                    "image_reviewed",
                    False,
                )
            )

            # Pozitif onaylı görüntüler ile bilinçli olarak
            # onaylanmış boş negatif görüntüler dahil edilir.
            if (
                reviewed_objects
                or image_reviewed
            ):
                width = record.get(
                    "width"
                )

                height = record.get(
                    "height"
                )

                if not width or not height:
                    reader = QImageReader(
                        str(image_path)
                    )

                    size = reader.size()
                    width = size.width()
                    height = size.height()

                records.append({
                    "image_path": image_path,
                    "width": width,
                    "height": height,
                    "objects": reviewed_objects,
                })

        return records

    def write_yolo_split(
        self,
        records,
        output_root,
        split_name,
    ):
        image_directory = (
            output_root
            / "images"
            / split_name
        )

        label_directory = (
            output_root
            / "labels"
            / split_name
        )

        image_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        label_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for index, record in enumerate(
            records,
            start=1,
        ):
            source_path = record[
                "image_path"
            ]

            unique_stem = (
                f"{split_name}_"
                f"{index:06d}"
            )

            image_filename = (
                unique_stem
                + source_path.suffix.lower()
            )

            shutil.copy2(
                source_path,
                image_directory
                / image_filename,
            )

            width = float(
                record["width"]
            )

            height = float(
                record["height"]
            )

            lines = []

            for obj in record["objects"]:
                x1, y1, x2, y2 = (
                    obj["bbox"]
                )

                grasp = obj["grasp"]
                angle = obj["angle"]

                values = [
                    clamp(
                        ((x1 + x2) / 2)
                        / width,
                        0,
                        1,
                    ),
                    clamp(
                        ((y1 + y2) / 2)
                        / height,
                        0,
                        1,
                    ),
                    clamp(
                        (x2 - x1)
                        / width,
                        0,
                        1,
                    ),
                    clamp(
                        (y2 - y1)
                        / height,
                        0,
                        1,
                    ),
                    clamp(
                        grasp[0]
                        / width,
                        0,
                        1,
                    ),
                    clamp(
                        grasp[1]
                        / height,
                        0,
                        1,
                    ),
                    2,
                    clamp(
                        angle[0]
                        / width,
                        0,
                        1,
                    ),
                    clamp(
                        angle[1]
                        / height,
                        0,
                        1,
                    ),
                    2,
                ]

                line = (
                    f"{CLASS_ID} "
                    f"{values[0]:.8f} "
                    f"{values[1]:.8f} "
                    f"{values[2]:.8f} "
                    f"{values[3]:.8f} "
                    f"{values[4]:.8f} "
                    f"{values[5]:.8f} "
                    f"{int(values[6])} "
                    f"{values[7]:.8f} "
                    f"{values[8]:.8f} "
                    f"{int(values[9])}"
                )

                lines.append(line)

            label_path = (
                label_directory
                / f"{unique_stem}.txt"
            )

            label_path.write_text(
                (
                    "\n".join(lines) + "\n"
                    if lines
                    else ""
                ),
                encoding="utf-8",
            )

    def prepare_training_dataset(self):
        records = (
            self.collect_training_records()
        )

        positive_images = sum(
            1
            for record in records
            if record["objects"]
        )

        negative_images = (
            len(records)
            - positive_images
        )

        object_count = sum(
            len(record["objects"])
            for record in records
        )

        if len(records) < 2:
            raise RuntimeError(
                "Eğitim için en az iki onaylı "
                "görüntü gerekir."
            )

        if positive_images < 2:
            raise RuntimeError(
                "Eğitim için en az iki pozitif "
                "görüntü gerekir."
            )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        export_root = (
            self.dataset_root
            / TRAINING_FOLDER
            / f"dataset_{timestamp}"
        )

        yolo_root = (
            export_root
            / "yolo_pose"
        )

        shuffled = list(records)

        random.Random(42).shuffle(
            shuffled
        )

        validation_count = max(
            1,
            round(
                len(shuffled)
                * 0.2
            ),
        )

        validation_count = min(
            validation_count,
            len(shuffled) - 1,
        )

        validation_records = (
            shuffled[
                :validation_count
            ]
        )

        train_records = (
            shuffled[
                validation_count:
            ]
        )

        self.write_yolo_split(
            train_records,
            yolo_root,
            "train",
        )

        self.write_yolo_split(
            validation_records,
            yolo_root,
            "val",
        )

        yolo_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        data_path = (
            yolo_root
            / "data.yaml"
        )

        data_path.write_text(
            (
                f'path: "'
                f'{yolo_root.resolve().as_posix()}'
                f'"\n'
                "train: images/train\n"
                "val: images/val\n\n"
                "kpt_shape: [2, 3]\n"
                "flip_idx: [0, 1]\n\n"
                "names:\n"
                "  0: fidan\n"
            ),
            encoding="utf-8",
        )

        return {
            "export_root": export_root,
            "data_path": data_path,
            "images": len(records),
            "positive_images": (
                positive_images
            ),
            "negative_images": (
                negative_images
            ),
            "objects": object_count,
        }

    # ========================================================
    # YOLO11 EĞİTİM
    # ========================================================

    def start_yolo11_training(self):
        if self.dataset_root is None:
            QMessageBox.information(
                self,
                "Önce klasör açın",
                "Önce görüntü klasörünü açın.",
            )
            return

        if (
            self.training_process
            is not None
            and self.training_process.state()
            != QProcess.ProcessState.NotRunning
        ):
            QMessageBox.information(
                self,
                "Eğitim sürüyor",
                "Bir eğitim işlemi zaten çalışıyor.",
            )
            return

        if self.inference_running:
            QMessageBox.information(
                self,
                "Ön etiketleme sürüyor",
                "Önce ön etiketlemenin bitmesini bekleyin.",
            )
            return

        if not self.check_cuda():
            return

        self.save_current_image()

        try:
            info = (
                self.prepare_training_dataset()
            )

        except Exception as error:
            QMessageBox.critical(
                self,
                "Eğitim verisi hazırlanamadı",
                str(error),
            )
            return

        answer = QMessageBox.question(
            self,
            "YOLO11m Pose eğitimi",
            (
                f"Model: {BASE_MODEL}\n"
                f"Onaylı görüntü: "
                f"{info['images']}\n"
                f"Pozitif görüntü: "
                f"{info['positive_images']}\n"
                f"Negatif görüntü: "
                f"{info['negative_images']}\n"
                f"Fidan instance: "
                f"{info['objects']}\n\n"
                f"Ayarlar:\n"
                f"Epoch: {TRAIN_EPOCHS}\n"
                f"Görüntü: "
                f"{TRAIN_IMAGE_SIZE}x"
                f"{TRAIN_IMAGE_SIZE}\n"
                f"Batch: {TRAIN_BATCH}\n"
                f"Patience: "
                f"{TRAIN_PATIENCE}\n"
                f"Workers: 0\n"
                f"Mosaic: kapalı\n"
                f"AMP: kapalı\n"
                f"GPU: CUDA device 0\n\n"
                f"Dokunulmamış diğer görüntüler "
                f"eğitime dahil edilmeyecek.\n\n"
                f"Eğitim başlatılsın mı?"
            ),
            (
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
            ),
        )

        if (
            answer
            != QMessageBox.StandardButton.Yes
        ):
            return

        run_name = datetime.now().strftime(
            "round_%Y%m%d_%H%M%S"
        )

        runs_root = (
            self.dataset_root
            / TRAINING_FOLDER
            / "runs"
        ).resolve()

        self.training_run_directory = (
            runs_root
            / run_name
        )

        script_path = (
            info["export_root"]
            / "run_training.py"
        )

        training_script = f'''from ultralytics import YOLO
import torch


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA aktif degil")

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
        flush=True,
    )

    print(
        "Torch:",
        torch.__version__,
        flush=True,
    )

    print(
        "CUDA:",
        torch.version.cuda,
        flush=True,
    )

    print(
        "Ayarlar: imgsz={TRAIN_IMAGE_SIZE}, "
        "batch={TRAIN_BATCH}, "
        "epochs={TRAIN_EPOCHS}, "
        "workers=0, amp=False, mosaic=0",
        flush=True,
    )

    model = YOLO({BASE_MODEL!r})

    model.train(
        data={str(info["data_path"].resolve())!r},
        epochs={TRAIN_EPOCHS},
        imgsz={TRAIN_IMAGE_SIZE},
        batch={TRAIN_BATCH},
        device=0,
        workers=0,
        cache=False,
        amp=False,
        patience={TRAIN_PATIENCE},
        mosaic=0.0,
        degrees=0.0,
        translate=0.05,
        scale=0.20,
        shear=0.0,
        perspective=0.0,
        fliplr=0.5,
        flipud=0.0,
        plots=True,
        project={str(runs_root)!r},
        name={run_name!r},
        exist_ok=False,
    )

    print(
        "TRAINING_COMPLETE",
        flush=True,
    )


if __name__ == "__main__":
    main()
'''

        script_path.write_text(
            training_script,
            encoding="utf-8",
        )

        self.training_process = (
            QProcess(self)
        )

        environment = (
            QProcessEnvironment
            .systemEnvironment()
        )

        environment.insert(
            "PYTHONUNBUFFERED",
            "1",
        )

        self.training_process.setProcessEnvironment(
            environment
        )

        self.training_process.setProgram(
            sys.executable
        )

        self.training_process.setArguments([
            "-u",
            str(script_path),
        ])

        self.training_process.setWorkingDirectory(
            str(info["export_root"])
        )

        self.training_process.setProcessChannelMode(
            QProcess.ProcessChannelMode.MergedChannels
        )

        self.training_process.readyReadStandardOutput.connect(
            self.read_training_output
        )

        self.training_process.finished.connect(
            self.on_training_finished
        )

        self.training_process.errorOccurred.connect(
            self.on_training_process_error
        )

        self.training_log.clear()
        self.training_log.setVisible(True)

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        self.append_log(
            f"Eğitim başlatılıyor: "
            f"{BASE_MODEL}"
        )

        self.append_log(
            f"Python: {sys.executable}"
        )

        self.append_log(
            f"Veri: "
            f"{info['data_path']}"
        )

        self.append_log(
            f"Ayarlar: "
            f"imgsz={TRAIN_IMAGE_SIZE}, "
            f"batch={TRAIN_BATCH}, "
            f"epochs={TRAIN_EPOCHS}, "
            f"workers=0, amp=False"
        )

        self.training_process.start()

    def read_training_output(self):
        if self.training_process is None:
            return

        raw = (
            self.training_process
            .readAllStandardOutput()
        )

        text = bytes(raw).decode(
            "utf-8",
            errors="replace",
        )

        if text:
            self.training_log.insertPlainText(
                text
            )

            self.training_log.moveCursor(
                QTextCursor.MoveOperation.End
            )

    def on_training_process_error(
        self,
        process_error,
    ):
        self.append_log(
            f"\nQProcess hatası: "
            f"{process_error}"
        )

    def on_training_finished(
        self,
        exit_code,
        exit_status,
    ):
        self.read_training_output()

        self.progress_bar.setVisible(
            False
        )

        best_path = None

        if self.training_run_directory:
            candidate = (
                self.training_run_directory
                / "weights"
                / "best.pt"
            )

            if candidate.exists():
                best_path = (
                    candidate.resolve()
                )

        if (
            exit_code == 0
            and best_path is not None
        ):
            self.append_log(
                f"\nEğitim tamamlandı.\n"
                f"best.pt: {best_path}"
            )

            answer = QMessageBox.question(
                self,
                "Eğitim tamamlandı",
                f"Yeni model:\n"
                f"{best_path}\n\n"
                "Bu modeli ön etiketleme modeli "
                "olarak etkinleştirelim mi?",
                (
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No
                ),
            )

            if (
                answer
                == QMessageBox.StandardButton.Yes
            ):
                self.loaded_model_path = str(
                    best_path
                )

                self.database[
                    "last_model_path"
                ] = self.loaded_model_path

                self.save_database()
                self.update_status()

                self.append_log(
                    f"Etkin model: "
                    f"{best_path}"
                )

        else:
            QMessageBox.critical(
                self,
                "Eğitim tamamlanamadı",
                "Eğitim hata verdi veya best.pt "
                "bulunamadı.\n"
                "Log bölümündeki son satırları kontrol edin.",
            )

        self.training_process = None

    # ========================================================
    # MANUEL DIŞA AKTAR
    # ========================================================

    def export_dataset(self):
        if self.dataset_root is None:
            QMessageBox.information(
                self,
                "Önce klasör açın",
                "Önce görüntü klasörünü açın.",
            )
            return

        self.save_current_image()

        records = (
            self.collect_training_records()
        )

        if not records:
            QMessageBox.warning(
                self,
                "Onaylı anotasyon yok",
                "Eğitime uygun onaylı veri bulunamadı.",
            )
            return

        selected = (
            QFileDialog.getExistingDirectory(
                self,
                "Dışa aktarma klasörünü seçin",
            )
        )

        if not selected:
            return

        export_root = (
            Path(selected)
            / "fidan_pose_export"
        )

        if export_root.exists():
            answer = QMessageBox.question(
                self,
                "Klasör mevcut",
                f"{export_root} silinip yeniden "
                "oluşturulsun mu?",
                (
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No
                ),
            )

            if (
                answer
                != QMessageBox.StandardButton.Yes
            ):
                return

            shutil.rmtree(
                export_root
            )

        yolo_root = (
            export_root
            / "yolo_pose"
        )

        shuffled = list(records)

        random.Random(42).shuffle(
            shuffled
        )

        validation_count = max(
            1,
            round(
                len(shuffled)
                * 0.2
            ),
        )

        if len(shuffled) > 1:
            validation_count = min(
                validation_count,
                len(shuffled) - 1,
            )

            validation = shuffled[
                :validation_count
            ]

            train = shuffled[
                validation_count:
            ]
        else:
            train = shuffled
            validation = shuffled

        self.write_yolo_split(
            train,
            yolo_root,
            "train",
        )

        self.write_yolo_split(
            validation,
            yolo_root,
            "val",
        )

        yolo_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            yolo_root
            / "data.yaml"
        ).write_text(
            (
                f'path: "'
                f'{yolo_root.resolve().as_posix()}'
                f'"\n'
                "train: images/train\n"
                "val: images/val\n\n"
                "kpt_shape: [2, 3]\n"
                "flip_idx: [0, 1]\n\n"
                "names:\n"
                "  0: fidan\n"
            ),
            encoding="utf-8",
        )

        QMessageBox.information(
            self,
            "Dışa aktarma tamamlandı",
            f"YOLO11 Pose veri seti:\n"
            f"{yolo_root}",
        )

    # ========================================================
    # KAPANIŞ
    # ========================================================

    def closeEvent(self, event):
        self.save_current_image()

        if (
            self.inference_running
            and self.inference_worker
            is not None
        ):
            self.inference_worker.cancel()

        if (
            self.training_process
            is not None
            and self.training_process.state()
            != QProcess.ProcessState.NotRunning
        ):
            answer = QMessageBox.question(
                self,
                "Eğitim sürüyor",
                "Eğitim devam ediyor. Uygulama "
                "kapatılırsa eğitim durdurulacak. "
                "Kapatılsın mı?",
                (
                    QMessageBox.StandardButton.Yes
                    | QMessageBox.StandardButton.No
                ),
            )

            if (
                answer
                != QMessageBox.StandardButton.Yes
            ):
                event.ignore()
                return

            self.training_process.kill()

            self.training_process.waitForFinished(
                3000
            )

        event.accept()


# ============================================================
# BAŞLAT
# ============================================================

def main():
    app = QApplication(sys.argv)

    app.setApplicationName(
        "Fidan Pose Etiketleyici"
    )

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
