"""Full-screen calibration and placement overlays."""

import ctypes
from ctypes import wintypes

from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..detection import normalize_custom_rgb_controls
from ..image_processing import (
    clamp_rect_to_canvas,
    default_image_placement,
    normalize_image_placement,
)
from .widgets import pil_rgba_to_qpixmap


class _WinPOINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class _WinMSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", _WinPOINT),
    ]


WM_NCHITTEST = 0x0084
HTTRANSPARENT = -1

class DetectionOverlay(QWidget):
    """Transparent always-on-top overlay for OpenCV detection debugging and manual calibration."""

    detection_changed = Signal(object, object, object, object)
    drag_finished = Signal()
    calibration_saved = Signal(object, object, object, object)
    calibration_cancelled = Signal()
    profile_save_requested = Signal(object, object, object, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.canvas = None
        self.palette = []
        self.brush_positions = []
        self.custom_rgb_controls = None
        self._origin_x = 0
        self._origin_y = 0
        self._last_message = "OpenCV Detection Overlay"
        self.edit_mode = False
        self._drag_target = None
        self._drag_last_global = None

    def set_edit_mode(self, enabled):
        self.edit_mode = bool(enabled)
        # Normal overlay is click-through.  Edit mode receives the mouse so
        # the detected canvas / palette / brush points can be dragged directly.
        self.setAttribute(Qt.WA_TransparentForMouseEvents, not self.edit_mode)
        self.setCursor(Qt.OpenHandCursor if self.edit_mode else Qt.ArrowCursor)
        self.update()

    def _update_virtual_geometry(self):
        app = QApplication.instance()
        screens = app.screens() if app else []
        if not screens:
            return

        left = min(screen.geometry().x() for screen in screens)
        top = min(screen.geometry().y() for screen in screens)
        right = max(screen.geometry().x() + screen.geometry().width() for screen in screens)
        bottom = max(screen.geometry().y() + screen.geometry().height() for screen in screens)

        self._origin_x = int(left)
        self._origin_y = int(top)
        self.setGeometry(int(left), int(top), int(right - left), int(bottom - top))

    def set_detection(self, canvas, palette, brush_positions, custom_rgb_controls=None):
        self.canvas = tuple(canvas) if canvas else None
        self.palette = list(palette or [])
        self.brush_positions = list(brush_positions or [])
        self.custom_rgb_controls = normalize_custom_rgb_controls(custom_rgb_controls)
        rgb_count = 0
        if self.custom_rgb_controls:
            rgb_count = 1 + len(self.custom_rgb_controls.get("inputs") or [])
        self._last_message = (
            f"Canvas: {self.canvas} | Palette: {len(self.palette)} | "
            f"Brush: {len(self.brush_positions)} | RGB: {rgb_count}"
        )
        self._update_virtual_geometry()
        self.update()

    def show_overlay(self):
        self._update_virtual_geometry()
        self.show()
        self.raise_()
        self.update()

    def nativeEvent(self, event_type, message):
        """Windows-only selective click-through while edit mode is on.

        The overlay is full-screen.  In edit mode it must receive clicks near
        draggable handles, but clicks elsewhere should pass through to the main
        Qt UI / browser instead of making buttons feel disabled.
        """
        if not self.edit_mode:
            return False, 0

        try:
            if "windows" not in str(event_type).lower():
                return False, 0

            msg = _WinMSG.from_address(int(message))
            if msg.message != WM_NCHITTEST:
                return False, 0

            gx = ctypes.c_short(int(msg.lParam) & 0xFFFF).value
            gy = ctypes.c_short((int(msg.lParam) >> 16) & 0xFFFF).value

            if self._hit_test(gx, gy) is None:
                return True, HTTRANSPARENT

        except Exception:
            return False, 0

        return False, 0

    def _local_point(self, x, y):
        return int(round(x - self._origin_x)), int(round(y - self._origin_y))

    def _event_global_point(self, event):
        p = event.position()
        return int(round(p.x() + self._origin_x)), int(round(p.y() + self._origin_y))

    def _point_near(self, x1, y1, x2, y2, radius):
        return (x1 - x2) ** 2 + (y1 - y2) ** 2 <= radius ** 2

    def _hud_rect(self):
        hud_w = min(760, max(420, self.width() - 36))
        hud_h = 120 if self.edit_mode else 74
        return QRectF(18, 18, hud_w, hud_h)

    def _action_button_rects(self):
        if not self.edit_mode:
            return {}
        hud = self._hud_rect()
        button_w = 96
        profile_w = 120
        button_h = 30
        gap = 10
        y = hud.bottom() - button_h - 12
        cancel = QRectF(hud.right() - button_w - 14, y, button_w, button_h)
        save = QRectF(cancel.left() - button_w - gap, y, button_w, button_h)
        save_profile = QRectF(save.left() - profile_w - gap, y, profile_w, button_h)
        return {"save_profile_button": save_profile, "save_button": save, "cancel_button": cancel}

    def _hit_action_button(self, gx, gy):
        lx, ly = self._local_point(gx, gy)
        for name, rect in self._action_button_rects().items():
            if rect.contains(float(lx), float(ly)):
                return (name, None)
        return None

    def _hit_test(self, gx, gy):
        action_target = self._hit_action_button(gx, gy)
        if action_target is not None:
            return action_target
        # Custom RGB controls get highest priority because the R/G/B boxes can
        # sit close to the palette area when the custom color panel is open.
        if self.custom_rgb_controls:
            try:
                swatch = self.custom_rgb_controls.get("swatch")
                if swatch and self._point_near(gx, gy, swatch[0], swatch[1], 34):
                    return ("rgb_swatch", None)
                for idx, pos in enumerate(self.custom_rgb_controls.get("inputs") or []):
                    if self._point_near(gx, gy, pos[0], pos[1], 28):
                        return ("rgb_input", idx)
            except Exception:
                pass

        # Palette / brush get priority so they are still draggable even when
        # close to the canvas box.
        for idx, item in enumerate(self.palette):
            try:
                px, py = item.get("pos", (None, None)) if isinstance(item, dict) else item
                if px is not None and py is not None and self._point_near(gx, gy, px, py, 26):
                    return ("palette", idx)
            except Exception:
                pass

        for idx, pos in enumerate(self.brush_positions):
            try:
                bx, by = pos
                if self._point_near(gx, gy, bx, by, 30):
                    return ("brush", idx)
            except Exception:
                pass

        if self.canvas:
            x1, y1, x2, y2 = self.canvas
            left, right = min(x1, x2), max(x1, x2)
            top, bottom = min(y1, y2), max(y1, y2)
            near_border = (
                left - 18 <= gx <= right + 18
                and top - 18 <= gy <= bottom + 18
                and (
                    abs(gx - left) <= 18 or abs(gx - right) <= 18
                    or abs(gy - top) <= 18 or abs(gy - bottom) <= 18
                    or (left <= gx <= right and top <= gy <= bottom)
                )
            )
            if near_border:
                return ("canvas", None)

        return None

    def _apply_drag_delta(self, target, dx, dy):
        kind, index = target
        dx, dy = int(dx), int(dy)
        if dx == 0 and dy == 0:
            return

        if kind == "canvas" and self.canvas:
            x1, y1, x2, y2 = self.canvas
            self.canvas = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

        elif kind == "palette" and index is not None and 0 <= index < len(self.palette):
            item = self.palette[index]
            if isinstance(item, dict):
                px, py = item.get("pos", (0, 0))
                new_item = dict(item)
                new_item["pos"] = (int(px + dx), int(py + dy))
                self.palette[index] = new_item
            else:
                px, py = item
                self.palette[index] = (int(px + dx), int(py + dy))

        elif kind == "brush" and index is not None and 0 <= index < len(self.brush_positions):
            bx, by = self.brush_positions[index]
            self.brush_positions[index] = (int(bx + dx), int(by + dy))

        elif kind == "rgb_swatch" and self.custom_rgb_controls:
            sx, sy = self.custom_rgb_controls.get("swatch", (0, 0))
            new_controls = dict(self.custom_rgb_controls)
            new_controls["swatch"] = (int(sx + dx), int(sy + dy))
            self.custom_rgb_controls = normalize_custom_rgb_controls(new_controls)

        elif kind == "rgb_input" and self.custom_rgb_controls and index is not None:
            inputs = list(self.custom_rgb_controls.get("inputs") or [])
            if 0 <= index < len(inputs):
                ix, iy = inputs[index]
                inputs[index] = (int(ix + dx), int(iy + dy))
                new_controls = dict(self.custom_rgb_controls)
                new_controls["inputs"] = inputs
                new_controls["source"] = "manual-overlay"
                self.custom_rgb_controls = normalize_custom_rgb_controls(new_controls)

        rgb_count = 0
        if self.custom_rgb_controls:
            rgb_count = 1 + len(self.custom_rgb_controls.get("inputs") or [])
        self._last_message = (
            f"Canvas: {self.canvas} | Palette: {len(self.palette)} | "
            f"Brush: {len(self.brush_positions)} | RGB: {rgb_count}"
        )
        self.detection_changed.emit(self.canvas, self.palette, self.brush_positions, self.custom_rgb_controls)
        self.update()

    def mousePressEvent(self, event):
        if not self.edit_mode or event.button() != Qt.LeftButton:
            event.ignore()
            return
        gx, gy = self._event_global_point(event)
        target = self._hit_test(gx, gy)
        if target is None:
            event.ignore()
            return
        if target[0] == "save_profile_button":
            self.profile_save_requested.emit(self.canvas, self.palette, self.brush_positions, self.custom_rgb_controls)
            event.accept()
            return
        if target[0] == "save_button":
            self.calibration_saved.emit(self.canvas, self.palette, self.brush_positions, self.custom_rgb_controls)
            event.accept()
            return
        if target[0] == "cancel_button":
            self.calibration_cancelled.emit()
            event.accept()
            return
        self._drag_target = target
        self._drag_last_global = (gx, gy)
        self.setCursor(Qt.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event):
        if not self.edit_mode or self._drag_target is None or self._drag_last_global is None:
            event.ignore()
            return
        gx, gy = self._event_global_point(event)
        last_x, last_y = self._drag_last_global
        self._apply_drag_delta(self._drag_target, gx - last_x, gy - last_y)
        self._drag_last_global = (gx, gy)
        event.accept()

    def mouseReleaseEvent(self, event):
        if not self.edit_mode or event.button() != Qt.LeftButton:
            event.ignore()
            return
        self._drag_target = None
        self._drag_last_global = None
        self.setCursor(Qt.OpenHandCursor)
        self.drag_finished.emit()
        event.accept()

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Small top-left HUD with Save / Cancel buttons in calibration mode.
        hud_rect = self._hud_rect()
        painter.setPen(QPen(QColor(125, 211, 252, 150), 1.0))
        painter.setBrush(QColor(7, 11, 22, 184))
        painter.drawRoundedRect(hud_rect, 12, 12)
        painter.setPen(QColor("#E0F2FE"))
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        title = "OpenCV Detect Overlay 偵測覆蓋層"
        if self.edit_mode:
            title += "  ·  拖動校正中"
        painter.drawText(hud_rect.adjusted(14, 9, -14, -72 if self.edit_mode else -38), Qt.AlignLeft | Qt.AlignVCenter, title)
        painter.setFont(QFont("Cascadia Mono", 9))
        painter.setPen(QColor("#93C5FD"))
        painter.drawText(hud_rect.adjusted(14, 35, -14, -42 if self.edit_mode else -8), Qt.AlignLeft | Qt.AlignVCenter, self._last_message)

        if self.edit_mode:
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(QColor("#BAE6FD"))
            painter.drawText(
                hud_rect.adjusted(14, 62, -205, -13),
                Qt.AlignLeft | Qt.AlignVCenter,
                "拖動畫布框 / 色盤 / 筆刷 / RGB 標記，完成後可保存本次或存成設定檔。"
            )
            for name, rect in self._action_button_rects().items():
                if name == "save_button":
                    grad = QLinearGradient(rect.topLeft(), rect.topRight())
                    grad.setColorAt(0.0, QColor("#22C55E"))
                    grad.setColorAt(1.0, QColor("#0D9488"))
                    label = "保存"
                    border = QColor("#86EFAC")
                elif name == "save_profile_button":
                    grad = QLinearGradient(rect.topLeft(), rect.topRight())
                    grad.setColorAt(0.0, QColor("#38BDF8"))
                    grad.setColorAt(1.0, QColor("#6366F1"))
                    label = "保存設定檔"
                    border = QColor("#7DD3FC")
                else:
                    grad = QLinearGradient(rect.topLeft(), rect.topRight())
                    grad.setColorAt(0.0, QColor("#475569"))
                    grad.setColorAt(1.0, QColor("#1E293B"))
                    label = "取消"
                    border = QColor("#94A3B8")
                painter.setPen(QPen(border, 1.2))
                painter.setBrush(grad)
                painter.drawRoundedRect(rect, 9, 9)
                painter.setPen(QColor("#FFFFFF"))
                painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
                painter.drawText(rect, Qt.AlignCenter, label)

        if self.canvas:
            x1, y1, x2, y2 = self.canvas
            lx1, ly1 = self._local_point(min(x1, x2), min(y1, y2))
            lx2, ly2 = self._local_point(max(x1, x2), max(y1, y2))
            rect = QRectF(lx1, ly1, lx2 - lx1, ly2 - ly1)

            painter.setPen(QPen(QColor(56, 189, 248, 235), 4.0))
            painter.setBrush(QColor(56, 189, 248, 22))
            painter.drawRoundedRect(rect, 10, 10)

            label_rect = QRectF(lx1, max(0, ly1 - 31), 260, 26)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(2, 132, 199, 210))
            painter.drawRoundedRect(label_rect, 7, 7)
            painter.setPen(QColor("#FFFFFF"))
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            painter.drawText(label_rect, Qt.AlignCenter, f"Canvas 畫布 {lx2-lx1} x {ly2-ly1}")

        # Palette swatches.
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        for idx, item in enumerate(self.palette):
            try:
                px, py = item.get("pos", (None, None)) if isinstance(item, dict) else item
                color = item.get("color", (56, 189, 248)) if isinstance(item, dict) else (56, 189, 248)
                if px is None or py is None:
                    continue
                lx, ly = self._local_point(px, py)
                size = 17
                swatch_rect = QRectF(lx - size / 2, ly - size / 2, size, size)
                painter.setPen(QPen(QColor("#FFFFFF"), 1.4))
                painter.setBrush(QColor(int(color[0]), int(color[1]), int(color[2]), 220))
                painter.drawRoundedRect(swatch_rect, 4, 4)
                painter.setPen(QColor("#E0F2FE"))
                painter.drawText(QRectF(lx + 10, ly - 10, 28, 20), Qt.AlignLeft | Qt.AlignVCenter, str(idx + 1))
            except Exception:
                continue

        # Brush buttons.
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        for idx, pos in enumerate(self.brush_positions):
            try:
                bx, by = pos
                lx, ly = self._local_point(bx, by)
                radius = 14
                painter.setPen(QPen(QColor(167, 139, 250, 235), 2.6))
                painter.setBrush(QColor(167, 139, 250, 42))
                painter.drawEllipse(QRectF(lx - radius, ly - radius, radius * 2, radius * 2))
                painter.setPen(QColor("#FFFFFF"))
                painter.drawText(QRectF(lx - radius, ly - radius, radius * 2, radius * 2), Qt.AlignCenter, str(idx + 1))
            except Exception:
                continue

        # Custom RGB controls: swatch opener + R/G/B input fields.
        if self.custom_rgb_controls:
            try:
                swatch = self.custom_rgb_controls.get("swatch")
                inputs = list(self.custom_rgb_controls.get("inputs") or [])
                if swatch:
                    sx, sy = swatch
                    lx, ly = self._local_point(sx, sy)
                    rect = QRectF(lx - 36, ly - 17, 72, 34)
                    painter.setPen(QPen(QColor(132, 204, 22, 235), 2.4))
                    painter.setBrush(QColor(132, 204, 22, 42))
                    painter.drawRoundedRect(rect, 7, 7)
                    painter.setPen(QColor("#ECFCCB"))
                    painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
                    painter.drawText(rect, Qt.AlignCenter, "RGB")

                labels = ["R", "G", "B"]
                colors = [QColor("#F87171"), QColor("#86EFAC"), QColor("#60A5FA")]
                painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
                for idx, pos in enumerate(inputs[:3]):
                    ix, iy = pos
                    lx, ly = self._local_point(ix, iy)
                    rect = QRectF(lx - 18, ly - 13, 36, 26)
                    painter.setPen(QPen(colors[idx], 2.2))
                    painter.setBrush(QColor(colors[idx].red(), colors[idx].green(), colors[idx].blue(), 42))
                    painter.drawRoundedRect(rect, 6, 6)
                    painter.setPen(QColor("#FFFFFF"))
                    painter.drawText(rect, Qt.AlignCenter, labels[idx])
            except Exception:
                pass

        painter.end()


class ImagePlacementOverlay(QWidget):
    """Drag/resize a translucent copy of the loaded image to choose the actual draw area."""

    placement_changed = Signal(object)
    placement_saved = Signal(object)
    placement_cancelled = Signal()

    HANDLE_MARGIN = 14
    HANDLE_SIZE = 18
    MIN_SIZE = 18

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.canvas = None
        self.placement_rect = None
        self.image_pixmap = QPixmap()
        self._image_aspect = 1.0
        self._origin_x = 0
        self._origin_y = 0
        self._drag_mode = None
        self._drag_start_global = None
        self._drag_start_rect = None
        self._last_message = "Image Position Overlay"
        self.setCursor(Qt.OpenHandCursor)

    def _update_virtual_geometry(self):
        app = QApplication.instance()
        screens = app.screens() if app else []
        if not screens:
            return
        left = min(screen.geometry().x() for screen in screens)
        top = min(screen.geometry().y() for screen in screens)
        right = max(screen.geometry().x() + screen.geometry().width() for screen in screens)
        bottom = max(screen.geometry().y() + screen.geometry().height() for screen in screens)
        self._origin_x = int(left)
        self._origin_y = int(top)
        self.setGeometry(int(left), int(top), int(right - left), int(bottom - top))

    def set_context(self, canvas, image, placement_rect=None, scale_percent=85):
        self.canvas = tuple(canvas) if canvas else None
        rect = normalize_image_placement(placement_rect)
        if rect is None:
            rect = default_image_placement(self.canvas, image, scale_percent=scale_percent)
        self.placement_rect = clamp_rect_to_canvas(rect, self.canvas)
        self.image_pixmap = pil_rgba_to_qpixmap(image, max_side=1200)
        try:
            img_w, img_h = image.size
            if img_w > 0 and img_h > 0:
                self._image_aspect = float(img_w) / float(img_h)
        except Exception:
            self._image_aspect = 1.0
        self._update_message()
        self._update_virtual_geometry()
        self.update()

    def show_overlay(self):
        self._update_virtual_geometry()
        self.show()
        self.raise_()
        self.update()

    def _local_point(self, x, y):
        return int(round(x - self._origin_x)), int(round(y - self._origin_y))

    def _event_global_point(self, event):
        p = event.position()
        return int(round(p.x() + self._origin_x)), int(round(p.y() + self._origin_y))

    def _placement_local_rect(self):
        rect = normalize_image_placement(self.placement_rect)
        if not rect:
            return QRectF()
        x1, y1 = self._local_point(rect[0], rect[1])
        x2, y2 = self._local_point(rect[2], rect[3])
        return QRectF(x1, y1, x2 - x1, y2 - y1)

    def _hud_rect(self):
        return QRectF(18, 18, min(800, max(500, self.width() - 36)), 122)

    def _action_button_rects(self):
        hud = self._hud_rect()
        button_h = 30
        gap = 10
        y = hud.bottom() - button_h - 12
        cancel = QRectF(hud.right() - 96 - 14, y, 96, button_h)
        save = QRectF(cancel.left() - 96 - gap, y, 96, button_h)
        center = QRectF(save.left() - 112 - gap, y, 112, button_h)
        return {"center_button": center, "save_button": save, "cancel_button": cancel}

    def _hit_action_button(self, gx, gy):
        lx, ly = self._local_point(gx, gy)
        for name, rect in self._action_button_rects().items():
            if rect.contains(float(lx), float(ly)):
                return name
        return None

    def _handle_hit_test(self, gx, gy):
        rect = normalize_image_placement(self.placement_rect)
        if not rect:
            return None
        x1, y1, x2, y2 = rect
        m = self.HANDLE_MARGIN
        inside_expanded = (x1 - m <= gx <= x2 + m and y1 - m <= gy <= y2 + m)
        if not inside_expanded:
            return None

        near_left = abs(gx - x1) <= m
        near_right = abs(gx - x2) <= m
        near_top = abs(gy - y1) <= m
        near_bottom = abs(gy - y2) <= m

        if near_left and near_top:
            return "resize_tl"
        if near_right and near_top:
            return "resize_tr"
        if near_left and near_bottom:
            return "resize_bl"
        if near_right and near_bottom:
            return "resize_br"
        if near_left and y1 <= gy <= y2:
            return "resize_left"
        if near_right and y1 <= gy <= y2:
            return "resize_right"
        if near_top and x1 <= gx <= x2:
            return "resize_top"
        if near_bottom and x1 <= gx <= x2:
            return "resize_bottom"
        if x1 <= gx <= x2 and y1 <= gy <= y2:
            return "image"
        return None

    def _hit_test(self, gx, gy):
        action = self._hit_action_button(gx, gy)
        if action:
            return action
        return self._handle_hit_test(gx, gy)

    def nativeEvent(self, event_type, message):
        """Let clicks outside the image/HUD pass through on Windows."""
        try:
            if "windows" not in str(event_type).lower():
                return False, 0
            msg = _WinMSG.from_address(int(message))
            if msg.message != WM_NCHITTEST:
                return False, 0
            gx = ctypes.c_short(int(msg.lParam) & 0xFFFF).value
            gy = ctypes.c_short((int(msg.lParam) >> 16) & 0xFFFF).value
            if self._hit_test(gx, gy) is None:
                return True, HTTRANSPARENT
        except Exception:
            return False, 0
        return False, 0

    def _update_message(self):
        rect = normalize_image_placement(self.placement_rect)
        if rect:
            self._last_message = f"位置: ({rect[0]}, {rect[1]})  大小: {rect[2]-rect[0]} x {rect[3]-rect[1]}"
        else:
            self._last_message = "尚未設定圖片位置"

    def _emit_rect_changed(self):
        self._update_message()
        self.placement_changed.emit(self.placement_rect)
        self.update()

    def _move_by(self, dx, dy):
        rect = normalize_image_placement(self.placement_rect)
        if not rect:
            return
        x1, y1, x2, y2 = rect
        self.placement_rect = clamp_rect_to_canvas((x1 + dx, y1 + dy, x2 + dx, y2 + dy), self.canvas)
        self._emit_rect_changed()

    def _canvas_bounds(self):
        canvas = normalize_image_placement(self.canvas)
        if not canvas:
            return None
        return canvas

    def _limit_size_to_canvas(self, w, h):
        canvas = self._canvas_bounds()
        w = max(float(self.MIN_SIZE), float(w))
        h = max(float(self.MIN_SIZE), float(h))
        if not canvas:
            return w, h
        canvas_w = max(1.0, float(canvas[2] - canvas[0]))
        canvas_h = max(1.0, float(canvas[3] - canvas[1]))
        if w > canvas_w or h > canvas_h:
            scale = min(canvas_w / max(w, 1.0), canvas_h / max(h, 1.0))
            w *= scale
            h *= scale
        return max(float(self.MIN_SIZE), w), max(float(self.MIN_SIZE), h)

    def _rect_from_resize(self, mode, start_rect, dx, dy):
        left, top, right, bottom = [float(v) for v in start_rect]
        old_w = max(1.0, right - left)
        old_h = max(1.0, bottom - top)
        aspect = max(0.05, float(self._image_aspect or old_w / old_h))
        min_w = float(self.MIN_SIZE)
        min_h = max(float(self.MIN_SIZE), min_w / aspect)
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0

        def size_from_width(w):
            w = max(min_w, float(w))
            return w, max(min_h, w / aspect)

        def size_from_height(h):
            h = max(min_h, float(h))
            return max(min_w, h * aspect), h

        def size_from_corner(raw_w, raw_h):
            raw_w = max(min_w, float(raw_w))
            raw_h = max(min_h, float(raw_h))
            # Pick the axis the mouse changed more strongly, then preserve image ratio.
            if abs(raw_w / old_w - 1.0) >= abs(raw_h / old_h - 1.0):
                return size_from_width(raw_w)
            return size_from_height(raw_h)

        if mode == "resize_right":
            w, h = size_from_width(old_w + dx)
            w, h = self._limit_size_to_canvas(w, h)
            return (left, cy - h / 2, left + w, cy + h / 2)
        if mode == "resize_left":
            w, h = size_from_width(old_w - dx)
            w, h = self._limit_size_to_canvas(w, h)
            return (right - w, cy - h / 2, right, cy + h / 2)
        if mode == "resize_bottom":
            w, h = size_from_height(old_h + dy)
            w, h = self._limit_size_to_canvas(w, h)
            return (cx - w / 2, top, cx + w / 2, top + h)
        if mode == "resize_top":
            w, h = size_from_height(old_h - dy)
            w, h = self._limit_size_to_canvas(w, h)
            return (cx - w / 2, bottom - h, cx + w / 2, bottom)
        if mode == "resize_br":
            w, h = size_from_corner(old_w + dx, old_h + dy)
            w, h = self._limit_size_to_canvas(w, h)
            return (left, top, left + w, top + h)
        if mode == "resize_tr":
            w, h = size_from_corner(old_w + dx, old_h - dy)
            w, h = self._limit_size_to_canvas(w, h)
            return (left, bottom - h, left + w, bottom)
        if mode == "resize_bl":
            w, h = size_from_corner(old_w - dx, old_h + dy)
            w, h = self._limit_size_to_canvas(w, h)
            return (right - w, top, right, top + h)
        if mode == "resize_tl":
            w, h = size_from_corner(old_w - dx, old_h - dy)
            w, h = self._limit_size_to_canvas(w, h)
            return (right - w, bottom - h, right, bottom)
        return start_rect

    def _resize_from_drag(self, mode, dx, dy):
        if not self._drag_start_rect:
            return
        new_rect = self._rect_from_resize(mode, self._drag_start_rect, dx, dy)
        self.placement_rect = clamp_rect_to_canvas(new_rect, self.canvas)
        self._emit_rect_changed()

    def _scale_at_center(self, factor):
        rect = normalize_image_placement(self.placement_rect)
        canvas = normalize_image_placement(self.canvas)
        if not rect or not canvas:
            return
        x1, y1, x2, y2 = rect
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = max(self.MIN_SIZE, (x2 - x1) * float(factor))
        h = max(self.MIN_SIZE, w / max(0.05, self._image_aspect))
        w, h = self._limit_size_to_canvas(w, h)
        self.placement_rect = clamp_rect_to_canvas((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), self.canvas)
        self._emit_rect_changed()

    def reset_center(self):
        rect = normalize_image_placement(self.placement_rect)
        canvas = normalize_image_placement(self.canvas)
        if not rect or not canvas:
            return
        w, h = rect[2] - rect[0], rect[3] - rect[1]
        left = canvas[0] + (canvas[2] - canvas[0] - w) / 2
        top = canvas[1] + (canvas[3] - canvas[1] - h) / 2
        self.placement_rect = clamp_rect_to_canvas((left, top, left + w, top + h), self.canvas)
        self._emit_rect_changed()

    def _cursor_for_target(self, target):
        if target == "image":
            return Qt.OpenHandCursor
        if target in ("resize_tl", "resize_br"):
            return Qt.SizeFDiagCursor
        if target in ("resize_tr", "resize_bl"):
            return Qt.SizeBDiagCursor
        if target in ("resize_left", "resize_right"):
            return Qt.SizeHorCursor
        if target in ("resize_top", "resize_bottom"):
            return Qt.SizeVerCursor
        if target in ("save_button", "cancel_button", "center_button"):
            return Qt.PointingHandCursor
        return Qt.ArrowCursor

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            event.ignore()
            return
        gx, gy = self._event_global_point(event)
        target = self._hit_test(gx, gy)
        if target == "save_button":
            self.placement_saved.emit(self.placement_rect)
            event.accept()
            return
        if target == "cancel_button":
            self.placement_cancelled.emit()
            event.accept()
            return
        if target == "center_button":
            self.reset_center()
            event.accept()
            return
        if target == "image" or str(target).startswith("resize_"):
            self._drag_mode = target
            self._drag_start_global = (gx, gy)
            self._drag_start_rect = normalize_image_placement(self.placement_rect)
            self.setCursor(Qt.ClosedHandCursor if target == "image" else self._cursor_for_target(target))
            event.accept()
            return
        event.ignore()

    def mouseMoveEvent(self, event):
        gx, gy = self._event_global_point(event)
        if not self._drag_mode or self._drag_start_global is None:
            self.setCursor(self._cursor_for_target(self._hit_test(gx, gy)))
            event.ignore()
            return
        start_x, start_y = self._drag_start_global
        dx = gx - start_x
        dy = gy - start_y
        if self._drag_mode == "image":
            # Moving is incremental so the rectangle keeps clamping naturally at canvas edges.
            last_x, last_y = self._drag_start_global
            self._move_by(gx - last_x, gy - last_y)
            self._drag_start_global = (gx, gy)
            self._drag_start_rect = normalize_image_placement(self.placement_rect)
        else:
            self._resize_from_drag(self._drag_mode, dx, dy)
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drag_mode:
            self._drag_mode = None
            self._drag_start_global = None
            self._drag_start_rect = None
            gx, gy = self._event_global_point(event)
            self.setCursor(self._cursor_for_target(self._hit_test(gx, gy)))
            event.accept()
            return
        event.ignore()

    def leaveEvent(self, event):
        if not self._drag_mode:
            self.setCursor(Qt.ArrowCursor)
        super().leaveEvent(event)

    def wheelEvent(self, event):
        gx, gy = self._event_global_point(event)
        target = self._hit_test(gx, gy)
        if target not in ("image", "resize_tl", "resize_tr", "resize_bl", "resize_br", "resize_left", "resize_right", "resize_top", "resize_bottom"):
            event.ignore()
            return
        delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        self._scale_at_center(1.06 if delta > 0 else 0.94)
        event.accept()

    def _draw_resize_handles(self, painter, rect):
        """Draw polished resize handles around the placement frame."""
        s = self.HANDLE_SIZE
        half = s / 2.0
        corner_len = max(28.0, s * 1.65)
        edge_w = max(28.0, s * 1.60)
        edge_h = 8.0

        # Soft outer glow around all resize points.
        glow_pen = QPen(QColor(56, 189, 248, 70), 8.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(glow_pen)
        painter.setBrush(Qt.NoBrush)

        corners = [
            (rect.left(), rect.top(), 1, 1),
            (rect.right(), rect.top(), -1, 1),
            (rect.left(), rect.bottom(), 1, -1),
            (rect.right(), rect.bottom(), -1, -1),
        ]
        for x, y, sx, sy in corners:
            painter.drawLine(QRectF(x, y, sx * corner_len, 0).topLeft(), QRectF(x + sx * corner_len, y, 0, 0).topLeft())
            painter.drawLine(QRectF(x, y, 0, sy * corner_len).topLeft(), QRectF(x, y + sy * corner_len, 0, 0).topLeft())

        # Sharp L-shaped corner brackets, like a clean tool/crop frame.
        painter.setPen(QPen(QColor(255, 255, 255, 245), 5.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        for x, y, sx, sy in corners:
            painter.drawLine(int(round(x)), int(round(y)), int(round(x + sx * corner_len)), int(round(y)))
            painter.drawLine(int(round(x)), int(round(y)), int(round(x)), int(round(y + sy * corner_len)))

        painter.setPen(QPen(QColor(56, 189, 248, 255), 3.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        for x, y, sx, sy in corners:
            painter.drawLine(int(round(x)), int(round(y)), int(round(x + sx * corner_len)), int(round(y)))
            painter.drawLine(int(round(x)), int(round(y)), int(round(x)), int(round(y + sy * corner_len)))

        # Small edge grips make it obvious that edges resize too.
        edge_points = [
            QRectF(rect.center().x() - edge_w / 2, rect.top() - edge_h / 2, edge_w, edge_h),
            QRectF(rect.center().x() - edge_w / 2, rect.bottom() - edge_h / 2, edge_w, edge_h),
            QRectF(rect.left() - edge_h / 2, rect.center().y() - edge_w / 2, edge_h, edge_w),
            QRectF(rect.right() - edge_h / 2, rect.center().y() - edge_w / 2, edge_h, edge_w),
        ]
        painter.setPen(QPen(QColor(15, 23, 42, 210), 2.0))
        painter.setBrush(QColor(255, 255, 255, 235))
        for handle in edge_points:
            painter.drawRoundedRect(handle, 4, 4)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(56, 189, 248, 245))
        inset = 2.0
        for handle in edge_points:
            painter.drawRoundedRect(handle.adjusted(inset, inset, -inset, -inset), 3, 3)

        # Corner grab dots on top of the L brackets.
        painter.setPen(QPen(QColor(15, 23, 42, 230), 2.0))
        painter.setBrush(QColor(56, 189, 248, 250))
        for x, y, _sx, _sy in corners:
            painter.drawEllipse(QRectF(x - half, y - half, s, s))
        painter.setPen(QPen(QColor(255, 255, 255, 230), 1.4))
        painter.setBrush(Qt.NoBrush)
        for x, y, _sx, _sy in corners:
            painter.drawEllipse(QRectF(x - half + 3, y - half + 3, s - 6, s - 6))

    def _draw_button(self, painter, rect, label, colors, border_color):
        grad = QLinearGradient(rect.topLeft(), rect.bottomRight())
        grad.setColorAt(0.0, QColor(colors[0]))
        grad.setColorAt(1.0, QColor(colors[1]))
        painter.setPen(QPen(QColor(border_color), 1.2))
        painter.setBrush(grad)
        painter.drawRoundedRect(rect, 10, 10)
        painter.setPen(QColor("#FFFFFF"))
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.drawText(rect, Qt.AlignCenter, label)

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        hud_rect = self._hud_rect()
        # Compact tool-style control card.
        for i, alpha in enumerate((30, 22, 14)):
            painter.setPen(QPen(QColor(56, 189, 248, alpha), 8 - i * 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(hud_rect.adjusted(-i * 2, -i * 2, i * 2, i * 2), 14 + i, 14 + i)

        hud_grad = QLinearGradient(hud_rect.topLeft(), hud_rect.bottomRight())
        hud_grad.setColorAt(0.0, QColor(15, 23, 42, 218))
        hud_grad.setColorAt(1.0, QColor(30, 41, 59, 205))
        painter.setPen(QPen(QColor(56, 189, 248, 150), 1.1))
        painter.setBrush(hud_grad)
        painter.drawRoundedRect(hud_rect, 8, 8)

        painter.setPen(QColor("#E0F2FE"))
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.drawText(hud_rect.adjusted(16, 9, -14, -76), Qt.AlignLeft | Qt.AlignVCenter, "Image Position Overlay 圖片定位覆蓋層")
        painter.setFont(QFont("Cascadia Mono", 9))
        painter.setPen(QColor("#7DD3FC"))
        painter.drawText(hud_rect.adjusted(16, 35, -14, -47), Qt.AlignLeft | Qt.AlignVCenter, self._last_message)
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor("#BAE6FD"))
        painter.drawText(hud_rect.adjusted(16, 63, -220, -12), Qt.AlignLeft | Qt.AlignVCenter, "拖中間移動；拖藍色邊框 / 四角像視窗一樣調整大小；滾輪也可縮放。")

        for name, rect in self._action_button_rects().items():
            if name == "save_button":
                self._draw_button(painter, rect, "保存", ("#22C55E", "#0D9488"), "#86EFAC")
            elif name == "center_button":
                self._draw_button(painter, rect, "置中", ("#F59E0B", "#D97706"), "#FDE68A")
            else:
                self._draw_button(painter, rect, "取消", ("#64748B", "#1E293B"), "#CBD5E1")

        canvas = normalize_image_placement(self.canvas)
        if canvas:
            lx1, ly1 = self._local_point(canvas[0], canvas[1])
            lx2, ly2 = self._local_point(canvas[2], canvas[3])
            canvas_rect = QRectF(lx1, ly1, lx2 - lx1, ly2 - ly1)
            painter.setPen(QPen(QColor(56, 189, 248, 160), 2.0, Qt.DashLine))
            painter.setBrush(QColor(56, 189, 248, 16))
            painter.drawRoundedRect(canvas_rect, 12, 12)

        rect = self._placement_local_rect()
        if rect.isValid() and not self.image_pixmap.isNull():
            # Draw a subtle shadow first so the selected image separates from the page.
            shadow_rect = rect.adjusted(8, 10, 8, 10)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, 75))
            painter.drawRoundedRect(shadow_rect, 16, 16)

            painter.setOpacity(0.64)
            painter.drawPixmap(rect, self.image_pixmap, QRectF(self.image_pixmap.rect()))
            painter.setOpacity(1.0)

            # Soft cyan glow around the image frame.
            for spread, alpha, width in ((9, 28, 7.0), (5, 45, 4.8), (2, 85, 3.0)):
                painter.setPen(QPen(QColor(56, 189, 248, alpha), width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                painter.setBrush(Qt.NoBrush)
                painter.drawRoundedRect(rect.adjusted(-spread, -spread, spread, spread), 14 + spread, 14 + spread)

            # Dark translucent fill + clean tool border.
            painter.setPen(QPen(QColor(15, 23, 42, 225), 5.4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.setBrush(QColor(56, 189, 248, 22))
            painter.drawRoundedRect(rect, 6, 6)

            border_grad = QLinearGradient(rect.topLeft(), rect.bottomRight())
            border_grad.setColorAt(0.0, QColor("#FFFFFF"))
            border_grad.setColorAt(0.45, QColor("#38BDF8"))
            border_grad.setColorAt(1.0, QColor("#2563EB"))
            painter.setPen(QPen(QBrush(border_grad), 3.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(rect, 6, 6)

            painter.setPen(QPen(QColor(255, 255, 255, 170), 1.2, Qt.DashLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawRoundedRect(rect.adjusted(6, 6, -6, -6), 4, 4)

            # Top tool strip: no mac-style traffic buttons, just a clean resize HUD.
            title_h = 30 if rect.height() >= 78 else 22
            title_rect = QRectF(rect.left() + 8, rect.top() + 8, max(96.0, rect.width() - 16), title_h)
            title_grad = QLinearGradient(title_rect.topLeft(), title_rect.bottomRight())
            title_grad.setColorAt(0.0, QColor(2, 6, 23, 226))
            title_grad.setColorAt(1.0, QColor(12, 74, 110, 216))
            painter.setPen(QPen(QColor(125, 211, 252, 120), 1.2))
            painter.setBrush(title_grad)
            painter.drawRoundedRect(title_rect, 5, 5)

            # Left status bar + small grip lines, more like a tool overlay than a desktop window.
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(56, 189, 248, 245))
            painter.drawRoundedRect(QRectF(title_rect.left() + 8, title_rect.top() + 6, 4, max(8.0, title_rect.height() - 12)), 2, 2)
            painter.setPen(QPen(QColor(186, 230, 253, 155), 1.2, Qt.SolidLine, Qt.RoundCap))
            grip_x = title_rect.right() - 42
            grip_y = title_rect.center().y() - 5
            for i in range(3):
                painter.drawLine(int(grip_x), int(grip_y + i * 5), int(grip_x + 26), int(grip_y + i * 5))

            painter.setPen(QColor("#FFFFFF"))
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            title_text = f"圖片定位  {int(rect.width())} x {int(rect.height())}"
            painter.drawText(title_rect.adjusted(20, 0, -52, 0), Qt.AlignVCenter | Qt.AlignLeft, title_text)

            self._draw_resize_handles(painter, rect)

            # Bottom helper pill.
            hint_w = min(390.0, max(220.0, rect.width() - 28))
            hint_rect = QRectF(rect.left() + 14, rect.bottom() - 38, hint_w, 26)
            if rect.height() >= 92:
                painter.setPen(QPen(QColor(255, 255, 255, 70), 1.0))
                painter.setBrush(QColor(15, 23, 42, 205))
                painter.drawRoundedRect(hint_rect, 10, 10)
                painter.setPen(QColor("#BAE6FD"))
                painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
                painter.drawText(hint_rect, Qt.AlignCenter, "拖曳藍色邊框或四角調整大小")

        painter.end()
