"""Gartic OpenCV Drawer.

Single-file version kept for easy GitHub downloads.  Sections are grouped so
contributors can still navigate the code without chasing imports across files.
"""

import ctypes
import json
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

import cv2
import mss
import numpy as np
import pyautogui
from PIL import Image, ImageDraw
from PySide6.QtCore import QObject, Qt, Signal, QTimer, QSize, QRectF
from PySide6.QtGui import QColor, QFont, QImage, QPixmap, QPainter, QPen, QLinearGradient, QBrush, QConicalGradient
from PySide6.QtWidgets import (
    QApplication,
    QAbstractButton,
    QAbstractSpinBox,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListView,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from pynput import keyboard as pynput_keyboard
except Exception:
    pynput_keyboard = None


# Give the Qt UI thread more chances to repaint while CPU-heavy prep runs.
try:
    sys.setswitchinterval(0.001)
except Exception:
    pass


# PyAutoGUI global tuning. Keep failsafe enabled so moving the mouse to a
# screen corner still stops runaway drawing.
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.001
pyautogui.MINIMUM_DURATION = 0
if hasattr(pyautogui, "MINIMUM_SLEEP"):
    pyautogui.MINIMUM_SLEEP = 0.001



# ============================================================================
# Configuration
# ============================================================================

DETECT_BUTTON_TEXT = "Auto Detect 自動偵測畫布與色盤"
DRAW_BUTTON_TEXT = "Draw Fast 快速繪製"
COUNTDOWN_SECONDS = 0
MODE_LINE = "line"
MODE_CLEAN_LINE = "clean_line"
MODE_DARK_OUTLINE = "dark_outline"
MODE_SMART_LINE = "smart_line"
MODE_PALETTE = "palette"
MODE_CUSTOM_RGB = "custom_rgb"
MODE_SBR = "sbr"
PROTECTED_WHITE = -2
DEFAULT_LINE_MOVE_MS = 10
DEFAULT_LINE_GAP_MS = 0
DEFAULT_LINE_SCALE = 85
DEFAULT_STROKE_STEP = 1
DEFAULT_CUSTOM_COLORS = 48
MAX_CUSTOM_COLORS = 512
DEFAULT_SBR_STROKES = 300
PREVIEW_MAX_SIZE = 620
PALETTE_SELECT_DELAY = 0.07
PALETTE_DETAIL_SELECT_DELAY = 0.05
CUSTOM_RGB_PANEL_DELAY = 0.30
DEFAULT_RGB_PANEL_DELAY_MS = int(CUSTOM_RGB_PANEL_DELAY * 1000)
HOTKEY_FOCUS_DELAY = 0.08
CUSTOM_RGB_SWATCH_CLICK_DELAY = 0.020
CUSTOM_RGB_INPUT_CLICK_DELAY = 0.085
CUSTOM_RGB_INPUT_WRITE_DELAY = 0.075
CUSTOM_RGB_TYPE_INTERVAL = 0.004
PROJECT_DIR = Path(__file__).resolve().parent
PROFILE_DIR = PROJECT_DIR / "profiles"
PROFILE_FILE = PROFILE_DIR / "gartic_profiles.json"
GARTIC_BRUSH_PIXELS = {
    # Extra-fine brush shortcut. Gartic exposes it via the backtick key (`),
    # not as one of the visible 1..5 brush buttons.
    0: 1,
    # Estimated brush diameters for Gartic hotkeys 1..5.
    1: 3,
    2: 6,
    3: 9,
    4: 13,
    5: 18,
}
FIXED_GARTIC_COLORS = [
    # Fixed Gartic palette order, row-major, from the 3 x 6 swatch grid.
    (0, 0, 0),        # black
    (102, 102, 102),  # dark gray
    (0, 85, 205),     # blue
    (255, 255, 255),  # white
    (170, 170, 170),  # light gray
    (45, 190, 230),   # cyan
    (0, 130, 35),     # green
    (175, 0, 0),      # dark red
    (155, 75, 20),    # brown
    (20, 185, 70),    # bright green
    (255, 15, 30),    # red
    (255, 115, 45),   # orange
    (190, 125, 25),   # ocher
    (175, 0, 90),     # magenta
    (200, 90, 90),    # muted red
    (255, 220, 120),  # yellow / blond
    (245, 0, 135),    # hot pink
    (245, 165, 165),  # light pink
]


# ============================================================================
# Shared Helpers
# ============================================================================

class StopDrawingException(Exception):
    pass


class ResponsiveYield:
    """Small cooperative yield helper for CPU-heavy Python loops.

    The drawing app already runs heavy preparation in a background thread,
    but pure-Python loops can still hold the GIL long enough to make Qt feel
    frozen.  Calling maybe() periodically gives the UI thread a chance to
    repaint timers, buttons, logs, and the computing animation.
    """

    def __init__(self, interval=0.015):
        self.interval = float(interval)
        self.last = time.perf_counter()

    def maybe(self):
        now = time.perf_counter()
        if now - self.last >= self.interval:
            time.sleep(0)
            self.last = time.perf_counter()


def ui_yield():
    time.sleep(0)


def raise_if_stopped(stop_event):
    if stop_event is not None and stop_event.is_set():
        raise StopDrawingException()


# ============================================================================
# Screen Detection
# ============================================================================

def capture_screen_rgb():
    with mss.mss() as sct:
        monitor = sct.monitors[0]
        shot = np.array(sct.grab(monitor))
        img_rgb = cv2.cvtColor(shot, cv2.COLOR_BGRA2RGB)
        return img_rgb, monitor["left"], monitor["top"]


def detect_canvas(img_rgb):
    """
    Browser/adaptive Gartic canvas detector.
    Works across different browsers, zoom levels, window sizes, and slightly
    tinted Gartic canvases by looking for large low-saturation bright regions
    instead of only pure white pixels.
    """
    h, w, _ = img_rgb.shape
    screen_area = max(1, h * w)

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    rgb_min = np.min(img_rgb, axis=2)
    rgb_max = np.max(img_rgb, axis=2)

    masks = []
    # White / pale blue / pale gray Gartic paper.
    masks.append(((val >= 202) & (sat <= 80)).astype(np.uint8))
    # Strict white fallback for browsers that render the canvas very cleanly.
    masks.append(((rgb_min >= 235) & ((rgb_max - rgb_min) <= 45)).astype(np.uint8))
    # Some browsers/subpixel scaling make the paper slightly darker.
    masks.append(((val >= 185) & (sat <= 55)).astype(np.uint8))

    candidates = []
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, ww, hh = cv2.boundingRect(contour)
            if ww <= 0 or hh <= 0:
                continue

            area = ww * hh
            ratio = ww / max(hh, 1)
            if area < screen_area * 0.035:
                continue
            if ww < w * 0.22 or hh < h * 0.16:
                continue
            if not (1.05 <= ratio <= 2.85):
                continue

            fill = float(np.mean(mask[y:y + hh, x:x + ww] > 0))
            if fill < 0.30:
                continue

            cx = x + ww / 2
            cy = y + hh / 2
            center_dx = abs(cx - w / 2) / max(w / 2, 1)
            center_dy = abs(cy - h / 2) / max(h / 2, 1)
            # Gartic canvas usually sits around the center and is much larger
            # than ads / browser UI cards.  Keep this as a soft preference.
            center_bonus = 1.0 - min(0.70, center_dx * 0.34 + center_dy * 0.20)
            score = area * (0.55 + fill) * center_bonus
            candidates.append((score, x, y, x + ww, y + hh, fill, ratio))

    if not candidates:
        return None

    candidates.sort(reverse=True, key=lambda item: item[0])
    _score, x1, y1, x2, y2, _fill, _ratio = candidates[0]

    # Refine edges inside the selected rectangle with a stricter paper mask.
    crop = img_rgb[y1:y2, x1:x2]
    if crop.size:
        crop_hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        c_sat = crop_hsv[:, :, 1]
        c_val = crop_hsv[:, :, 2]
        paper = ((c_val >= 205) & (c_sat <= 85)).astype(np.uint8)
        paper = cv2.morphologyEx(
            paper,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)),
            iterations=1,
        )
        contours, _ = cv2.findContours(paper, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Select the largest inner paper-like component that still looks like a canvas.
            best_inner = None
            best_area = 0
            for contour in contours:
                x, y, ww, hh = cv2.boundingRect(contour)
                area = ww * hh
                ratio = ww / max(hh, 1)
                if area > best_area and area > (x2 - x1) * (y2 - y1) * 0.35 and 1.05 <= ratio <= 2.85:
                    best_area = area
                    best_inner = (x, y, x + ww, y + hh)
            if best_inner:
                ix1, iy1, ix2, iy2 = best_inner
                # Only accept refinement when it does not shrink suspiciously hard.
                if (ix2 - ix1) >= (x2 - x1) * 0.70 and (iy2 - iy1) >= (y2 - y1) * 0.70:
                    x1, y1, x2, y2 = x1 + ix1, y1 + iy1, x1 + ix2, y1 + iy2

    return (int(x1), int(y1), int(x2), int(y2))


def sort_centers_grid(centers):
    centers = sorted(centers, key=lambda p: (p[1], p[0]))

    rows = []
    for p in centers:
        added = False
        for row in rows:
            if abs(row[0][1] - p[1]) < 18:
                row.append(p)
                added = True
                break
        if not added:
            rows.append([p])

    rows = [sorted(row, key=lambda p: p[0]) for row in rows]
    rows = sorted(rows, key=lambda row: row[0][1])

    result = []
    for row in rows:
        result.extend(row)

    return result


def _dedupe_centers(centers, min_dist=14):
    result = []
    for cx, cy in sorted(centers, key=lambda p: (p[1], p[0])):
        duplicate = False
        for px, py in result:
            if (px - cx) ** 2 + (py - cy) ** 2 < min_dist ** 2:
                duplicate = True
                break
        if not duplicate:
            result.append((int(cx), int(cy)))
    return result


def _cluster_axis(values, tolerance):
    values = sorted([float(v) for v in values])
    clusters = []
    for value in values:
        if not clusters or abs(np.mean(clusters[-1]) - value) > tolerance:
            clusters.append([value])
        else:
            clusters[-1].append(value)
    return [float(np.median(cluster)) for cluster in clusters]


def _infer_palette_grid_from_candidates(centers, expected_cols=3, expected_rows=6):
    if len(centers) < 10:
        return []

    xs = [p[0] for p in centers]
    ys = [p[1] for p in centers]
    span_x = max(xs) - min(xs) if xs else 1
    span_y = max(ys) - min(ys) if ys else 1
    x_tol = max(10, span_x / max(expected_cols * 2.2, 1))
    y_tol = max(10, span_y / max(expected_rows * 2.2, 1))
    col_centers = _cluster_axis(xs, x_tol)
    row_centers = _cluster_axis(ys, y_tol)

    # If contours detected only edges or missed white/black squares, infer the
    # whole regular grid from median gaps.
    if len(col_centers) >= expected_cols:
        col_centers = sorted(col_centers)[:expected_cols]
    elif len(col_centers) >= 2:
        gap = float(np.median(np.diff(sorted(col_centers))))
        start = min(col_centers)
        col_centers = [start + i * gap for i in range(expected_cols)]
    else:
        return []

    if len(row_centers) >= expected_rows:
        # Pick the densest six top-to-bottom rows.
        row_centers = sorted(row_centers)[:expected_rows]
    elif len(row_centers) >= 2:
        gap = float(np.median(np.diff(sorted(row_centers))))
        start = min(row_centers)
        row_centers = [start + i * gap for i in range(expected_rows)]
    else:
        return []

    return [(int(round(x)), int(round(y))) for y in row_centers for x in col_centers]


def _palette_fallback_centers(canvas, img_shape, side="left"):
    x1, y1, x2, y2 = canvas
    h, w, _ = img_shape
    canvas_w = max(1, x2 - x1)
    canvas_h = max(1, y2 - y1)

    swatch_gap_x = max(30, int(round(canvas_w * 0.050)))
    row_gap = max(34, int(round(canvas_h * 0.087)))
    start_y = int(round(y1 + canvas_h * 0.125))

    if side == "right":
        base_x = int(round(x2 + canvas_w * 0.055))
        xs = [base_x + i * swatch_gap_x for i in range(3)]
    else:
        xs = [
            int(round(x1 - canvas_w * 0.155)),
            int(round(x1 - canvas_w * 0.105)),
            int(round(x1 - canvas_w * 0.055)),
        ]

    ys = [start_y + i * row_gap for i in range(6)]
    centers = []
    for y in ys:
        for x in xs:
            centers.append((int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))))
    return centers


def sample_palette_color(img_rgb, cx, cy, radius=7):
    h, w, _ = img_rgb.shape
    x1 = max(0, cx - radius)
    x2 = min(w, cx + radius + 1)
    y1 = max(0, cy - radius)
    y2 = min(h, cy + radius + 1)
    patch = img_rgb[y1:y2, x1:x2].reshape(-1, 3)

    if len(patch) == 0:
        return tuple(int(v) for v in img_rgb[cy, cx][:3])

    return tuple(int(v) for v in np.median(patch, axis=0))


def palette_colors_for_mapping(detected_colors):
    return FIXED_GARTIC_COLORS


def clamp_brush_key(value):
    return int(np.clip(int(value), 0, 5))


def gartic_brush_pixels(brush_key):
    return GARTIC_BRUSH_PIXELS.get(clamp_brush_key(brush_key), GARTIC_BRUSH_PIXELS[3])


def white_palette_indices(palette_colors, threshold=235):
    """
    找出接近白色的色盤 index。
    Gartic 畫布本來就是白色，所以全彩 / 簡化上色時不需要再畫白色。
    """
    if not palette_colors:
        return set()

    colors = np.asarray(palette_colors, dtype=np.int32)

    white_mask = (
        np.min(colors, axis=1) >= threshold
    ) & (
        np.max(colors, axis=1) - np.min(colors, axis=1) <= 35
    )

    return set(np.where(white_mask)[0].tolist())


def canvas_white_pixel_mask(rgb, alpha, threshold=248, chroma_limit=22):
    rgb = np.asarray(rgb, dtype=np.uint8)
    alpha = np.asarray(alpha, dtype=np.uint8)
    return (
        (alpha >= 40)
        & (np.min(rgb, axis=2) >= threshold)
        & ((np.max(rgb, axis=2) - np.min(rgb, axis=2)) <= chroma_limit)
    )


def remap_palette_white_cells(idx, rgb, alpha, palette_colors, skip_mask, protected_mask):
    """
    Palette mode has only one true white swatch. Pale skin can be nearest to
    that white swatch, but skipping it leaves face holes. Keep real canvas-white
    pixels blank, and remap warm/off-white pixels to the nearest non-white color.
    """
    white_indices = white_palette_indices(palette_colors)
    if not white_indices:
        return idx

    idx = np.asarray(idx, dtype=np.int16).copy()
    white_idx_mask = np.isin(idx, list(white_indices))
    if not np.any(white_idx_mask):
        return idx

    real_canvas_white = canvas_white_pixel_mask(rgb, alpha)
    skip_mask = np.asarray(skip_mask, dtype=bool)
    protected_mask = np.asarray(protected_mask, dtype=bool)
    remap_mask = white_idx_mask & (~real_canvas_white) & (~skip_mask) & (~protected_mask)

    non_white_indices = [i for i in range(len(palette_colors)) if i not in white_indices]
    if remap_mask.any() and non_white_indices:
        non_white_colors = [palette_colors[i] for i in non_white_indices]
        local_idx = nearest_color_index_map(rgb, non_white_colors)
        index_lookup = np.asarray(non_white_indices, dtype=np.int16)
        idx[remap_mask] = index_lookup[local_idx[remap_mask]]

    idx[white_idx_mask & real_canvas_white] = -1
    return idx


def nearest_color_index_map(rgb, palette_colors):
    """Memory-friendly nearest-color quantization for RGB images."""
    colors = np.asarray(palette_colors, dtype=np.int32)
    h, w = rgb.shape[:2]

    if len(colors) == 0:
        return np.full((h, w), -1, dtype=np.int16)

    rgb_i = np.asarray(rgb, dtype=np.int32)
    best_idx = np.zeros((h, w), dtype=np.int16)
    best_dist = np.full((h, w), np.iinfo(np.int32).max, dtype=np.int32)
    responsive = ResponsiveYield()

    for color_idx, color in enumerate(colors):
        responsive.maybe()
        diff = rgb_i - color.reshape(1, 1, 3)
        dist = np.sum(diff * diff, axis=2, dtype=np.int32)
        better = dist < best_dist
        if np.any(better):
            best_dist[better] = dist[better]
            best_idx[better] = color_idx

    return best_idx


def detect_palette(img_rgb, canvas):
    x1, y1, x2, y2 = canvas
    h, w, _ = img_rgb.shape
    canvas_w = max(1, x2 - x1)
    canvas_h = max(1, y2 - y1)

    search_regions = []
    # Gartic normally places the color palette on the left of the canvas.
    search_regions.append((
        max(0, int(x1 - canvas_w * 0.24)),
        max(0, int(y1 + canvas_h * 0.04)),
        min(w, int(x1 - canvas_w * 0.01)),
        min(h, int(y1 + canvas_h * 0.82)),
        "left",
    ))
    # Future / mirrored layouts or different browser scaling can place usable
    # controls on the right; keep this as a secondary path.
    search_regions.append((
        max(0, int(x2 + canvas_w * 0.01)),
        max(0, int(y1 + canvas_h * 0.04)),
        min(w, int(x2 + canvas_w * 0.24)),
        min(h, int(y1 + canvas_h * 0.82)),
        "right",
    ))

    best_centers = []
    best_side = "left"
    best_score = -1

    for rx1, ry1, rx2, ry2, side in search_regions:
        if rx2 <= rx1 + 20 or ry2 <= ry1 + 20:
            continue
        crop = img_rgb[ry1:ry2, rx1:rx2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        color_mask = (((sat > 45) & (val > 45)) | (gray < 70) | (gray > 238)).astype(np.uint8) * 255
        edges = cv2.Canny(gray, 35, 140)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        mixed = cv2.bitwise_or(edges, color_mask)
        mixed = cv2.morphologyEx(mixed, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(mixed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        min_square = max(16, int(canvas_h * 0.028))
        max_square = max(36, int(canvas_h * 0.095))

        for contour in contours:
            x, y, ww, hh = cv2.boundingRect(contour)
            if ww < min_square or hh < min_square or ww > max_square or hh > max_square:
                continue
            if abs(ww - hh) > max(10, max(ww, hh) * 0.45):
                continue
            area = ww * hh
            contour_area = abs(cv2.contourArea(contour))
            if contour_area < area * 0.18:
                continue
            cx = rx1 + x + ww // 2
            cy = ry1 + y + hh // 2
            centers.append((cx, cy))

        centers = _dedupe_centers(centers, min_dist=max(12, int(canvas_h * 0.030)))
        inferred = _infer_palette_grid_from_candidates(centers)
        score = len(inferred) * 10 + len(centers)
        if score > best_score:
            best_score = score
            best_centers = inferred if len(inferred) >= 18 else centers
            best_side = side

    if len(best_centers) >= 18:
        centers = sort_centers_grid(best_centers)[:18]
    else:
        centers = _palette_fallback_centers(canvas, img_rgb.shape, side=best_side)

    palette = []
    for idx, (cx, cy) in enumerate(centers[:len(FIXED_GARTIC_COLORS)]):
        cx = int(np.clip(cx, 0, w - 1))
        cy = int(np.clip(cy, 0, h - 1))
        palette.append({
            "pos": (cx, cy),
            "color": FIXED_GARTIC_COLORS[idx]
        })

    return palette


def detect_brush_buttons(img_rgb, canvas):
    x1, y1, x2, y2 = canvas
    h, w, _ = img_rgb.shape
    canvas_w = max(1, x2 - x1)
    canvas_h = max(1, y2 - y1)

    rx1 = max(0, int(x1 - canvas_w * 0.05))
    rx2 = min(w, int(x1 + canvas_w * 0.78))
    ry1 = min(h, max(0, int(y2 + canvas_h * 0.03)))
    ry2 = min(h, int(y2 + canvas_h * 0.28))

    centers = []

    if ry2 > ry1 + 36 and rx2 > rx1 + 120:
        crop = img_rgb[ry1:ry2, rx1:rx2]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(24, int(canvas_w * 0.045)),
            param1=80,
            param2=15,
            minRadius=max(8, int(canvas_h * 0.017)),
            maxRadius=max(20, int(canvas_h * 0.055))
        )

        if circles is not None:
            for cx, cy, radius in np.round(circles[0]).astype(int):
                gx = rx1 + cx
                gy = ry1 + cy
                if x1 - canvas_w * 0.08 <= gx <= x1 + canvas_w * 0.55 and y2 <= gy <= y2 + canvas_h * 0.25:
                    centers.append((gx, gy))

        if len(centers) < 5:
            edges = cv2.Canny(gray, 35, 130)
            edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, ww, hh = cv2.boundingRect(contour)
                if not (max(16, canvas_h * 0.025) <= ww <= max(50, canvas_h * 0.09)):
                    continue
                if not (max(16, canvas_h * 0.025) <= hh <= max(50, canvas_h * 0.09)):
                    continue
                if abs(ww - hh) > max(10, max(ww, hh) * 0.45):
                    continue
                centers.append((rx1 + x + ww // 2, ry1 + y + hh // 2))

    centers = _dedupe_centers(centers, min_dist=max(18, int(canvas_w * 0.035)))

    if len(centers) >= 5:
        # Pick the densest row of five brush circles.
        rows = []
        for p in sorted(centers, key=lambda p: p[1]):
            for row in rows:
                if abs(np.mean([q[1] for q in row]) - p[1]) < max(20, canvas_h * 0.04):
                    row.append(p)
                    break
            else:
                rows.append([p])
        rows.sort(key=lambda row: (-len(row), np.mean([q[0] for q in row])))
        centers = sorted(rows[0], key=lambda p: p[0])[:5]

    if len(centers) < 5:
        # Ratio fallback for the current Gartic layout.
        spacing = int(np.clip(canvas_w * 0.068, 32, 95))
        start_x = int(x1 + canvas_w * 0.043)
        brush_y = int(min(h - 1, y2 + canvas_h * 0.145))
        centers = [(start_x + i * spacing, brush_y) for i in range(5)]

    return [(int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))) for x, y in centers[:5]]


def _palette_xy_arrays(palette):
    xs = []
    ys = []
    for item in palette or []:
        try:
            if isinstance(item, dict):
                x, y = item.get("pos", (None, None))
            else:
                x, y = item
            if x is not None and y is not None:
                xs.append(float(x))
                ys.append(float(y))
        except Exception:
            continue
    return xs, ys


def estimate_custom_rgb_controls(palette, canvas=None):
    """
    Estimate Gartic custom RGB panel controls from detected palette.

    swatch = the custom color rectangle below the 18-color palette.
    inputs = R/G/B text fields after clicking the swatch.
    These are also shown in Overlay and can be dragged for calibration.
    """
    if not palette or len(palette) < 6:
        return None

    xs, ys = _palette_xy_arrays(palette)
    if len(xs) < 6 or len(ys) < 6:
        return None

    # The 18 swatches are normally 3 x 6.  Work with all detected points so
    # different browser zoom / UI scaling still has a reasonable fallback.
    x_clusters = []
    for x in sorted(xs):
        for group in x_clusters:
            if abs(np.mean(group) - x) < 18:
                group.append(x)
                break
        else:
            x_clusters.append([x])
    col_centers = [float(np.mean(group)) for group in x_clusters]
    col_gap = max(36.0, float(np.median(np.diff(sorted(col_centers))))) if len(col_centers) >= 2 else 42.0

    # Robust row gap: palette has duplicate x values, so use row-like y clusters.
    y_clusters = []
    for y in sorted(ys):
        for group in y_clusters:
            if abs(np.mean(group) - y) < 18:
                group.append(y)
                break
        else:
            y_clusters.append([y])
    row_centers = [float(np.mean(group)) for group in y_clusters]
    row_gap = max(1.0, float(np.median(np.diff(sorted(row_centers))))) if len(row_centers) >= 2 else col_gap

    center_x = float(np.mean(col_centers)) if col_centers else float(np.mean(xs))
    last_y = float(max(row_centers) if row_centers else max(ys))

    swatch = (
        int(round(center_x)),
        int(round(last_y + row_gap * 1.35))
    )

    # When the RGB panel opens, the R/G/B inputs are horizontally aligned.
    # This estimate is good enough to show draggable overlay handles; if the
    # panel is visible during Auto Detect, detect_custom_rgb_controls() refines it.
    input_y = int(round(swatch[1] + row_gap * 3.40))
    input_gap = max(36.0, col_gap * 0.95)
    inputs = [
        (int(round(center_x - input_gap)), input_y),
        (int(round(center_x)), input_y),
        (int(round(center_x + input_gap)), input_y),
    ]

    return {
        "swatch": swatch,
        "inputs": inputs,
        "source": "palette-estimate",
    }


def normalize_custom_rgb_controls(controls):
    if not controls:
        return None
    try:
        swatch = controls.get("swatch")
        inputs = list(controls.get("inputs") or [])
        if not swatch or len(inputs) < 3:
            return None
        return {
            "swatch": (int(swatch[0]), int(swatch[1])),
            "inputs": [(int(x), int(y)) for x, y in inputs[:3]],
            "source": controls.get("source", "manual"),
        }
    except Exception:
        return None


def offset_custom_rgb_controls(controls, dx, dy):
    controls = normalize_custom_rgb_controls(controls)
    if not controls:
        return None
    dx, dy = int(dx), int(dy)
    sx, sy = controls["swatch"]
    return {
        "swatch": (sx + dx, sy + dy),
        "inputs": [(x + dx, y + dy) for x, y in controls["inputs"]],
        "source": controls.get("source", "offset"),
    }


def _find_rgb_input_row(img_rgb, search_rect, expected_y=None):
    """Detect the three white RGB input boxes if the custom panel is open."""
    if img_rgb is None or search_rect is None:
        return None
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in search_rect]
    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, x1 + 1, w))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, y1 + 1, h))
    crop = img_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # RGB text boxes are very light rectangles.  Use low saturation/high value
    # instead of pure white so Windows/browser font smoothing does not break it.
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = ((val > 218) & (sat < 55)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        if 34 <= ww <= 95 and 18 <= hh <= 45 and 1.25 <= ww / max(hh, 1) <= 4.5:
            gx = x1 + x + ww // 2
            gy = y1 + y + hh // 2
            candidates.append((gx, gy, ww, hh))

    if len(candidates) < 3:
        return None

    rows = []
    for cand in sorted(candidates, key=lambda p: p[1]):
        added = False
        for row in rows:
            if abs(np.mean([q[1] for q in row]) - cand[1]) <= 18:
                row.append(cand)
                added = True
                break
        if not added:
            rows.append([cand])

    best = None
    best_score = -1e18
    for row in rows:
        row = sorted(row, key=lambda p: p[0])
        if len(row) < 3:
            continue
        # Choose the most evenly spaced three in this row.
        for i in range(0, len(row) - 2):
            trio = row[i:i + 3]
            gaps = [trio[1][0] - trio[0][0], trio[2][0] - trio[1][0]]
            if min(gaps) < 22 or max(gaps) > 115:
                continue
            yavg = float(np.mean([p[1] for p in trio]))
            evenness = -abs(gaps[0] - gaps[1])
            y_score = -abs(yavg - expected_y) if expected_y is not None else yavg * 0.05
            score = evenness + y_score
            if score > best_score:
                best_score = score
                best = [(int(p[0]), int(p[1])) for p in trio]

    return best


def detect_custom_rgb_controls(img_rgb, canvas=None, palette=None):
    """
    Detect/estimate Custom RGB controls.
    If the RGB panel is visible, refine the R/G/B input positions by OpenCV.
    Otherwise return a draggable palette-based estimate.
    """
    fallback = estimate_custom_rgb_controls(palette, canvas)
    if img_rgb is None or fallback is None:
        return fallback

    h, w = img_rgb.shape[:2]
    xs, ys = _palette_xy_arrays(palette)
    if not xs or not ys:
        return fallback

    center_x = float(np.mean(xs))
    last_y = float(max(ys))
    row_gap = max(36.0, float(np.median(np.diff(sorted(set(int(round(y)) for y in ys)))))) if len(set(int(round(y)) for y in ys)) >= 2 else 46.0

    sx, sy = fallback["swatch"]
    expected_input_y = fallback["inputs"][0][1]
    search_rect = (
        int(center_x - row_gap * 3.2),
        int(last_y + row_gap * 0.65),
        int(center_x + row_gap * 3.6),
        int(min(h, last_y + row_gap * 6.8)),
    )
    inputs = _find_rgb_input_row(img_rgb, search_rect, expected_y=expected_input_y)

    if inputs:
        return {
            "swatch": (int(sx), int(sy)),
            "inputs": inputs,
            "source": "opencv-rgb-inputs",
        }

    return fallback


# ============================================================================
# Mouse Automation And Paths
# ============================================================================

def screen_point_px(left, top, offset_x, offset_y, point):
    x, y = point
    return (
        int(round(left + offset_x + x)),
        int(round(top + offset_y + y))
    )


def color_drag_duration(distance_px, brush_px, requested_drag):
    """Duration for color fill drags.  Too-fast drags cause dotted/empty slits."""
    distance_px = abs(float(distance_px))
    brush_px = max(1, int(brush_px))
    requested_drag = max(0.001, float(requested_drag))
    if distance_px <= 1:
        return 0
    # Bigger brush can move faster, but still not as fast as the old 0.008s cap.
    speed = 4200.0 if brush_px <= 3 else 5200.0 if brush_px <= 6 else 6500.0
    return max(requested_drag, min(0.085, max(0.004, distance_px / speed)))


def draw_color_run_solid(sx, ex, sy, brush_px, requested_drag, stop_event):
    """Draw a horizontal color run in overlapped chunks for fewer gaps."""
    sx, ex, sy = int(round(sx)), int(round(ex)), int(round(sy))
    if abs(ex - sx) <= 1:
        pyautogui.click(sx, sy)
        return

    direction = 1 if ex >= sx else -1
    distance = abs(ex - sx)
    # Segment long runs so browser/Gartic does not drop mouse samples.
    segment_px = 120 if brush_px <= 3 else 170 if brush_px <= 6 else 230
    overlap_px = max(2, min(10, int(max(brush_px, 2) * 0.55)))

    if distance <= segment_px:
        pyautogui.moveTo(sx, sy, duration=0)
        pyautogui.dragTo(ex, sy, duration=color_drag_duration(distance, brush_px, requested_drag), button="left")
        return

    current = sx
    while True:
        raise_if_stopped(stop_event)
        remaining = abs(ex - current)
        if remaining <= segment_px:
            target = ex
        else:
            target = current + direction * segment_px

        pyautogui.moveTo(current, sy, duration=0)
        pyautogui.dragTo(
            target, sy,
            duration=color_drag_duration(abs(target - current), brush_px, requested_drag),
            button="left"
        )

        if target == ex:
            break

        # Start the next chunk slightly inside the previous one.
        current = target - direction * overlap_px


def select_gartic_brush(brush_key, brush_positions=None, stop_event=None, focus_point=None):
    raise_if_stopped(stop_event)
    key = clamp_brush_key(brush_key)

    if key == 0:
        if brush_positions:
            x, y = brush_positions[0]
            pyautogui.click(x, y)
            time.sleep(HOTKEY_FOCUS_DELAY)
        elif focus_point:
            pyautogui.click(*focus_point)
            time.sleep(HOTKEY_FOCUS_DELAY)
        pyautogui.press("`")
        time.sleep(0.30)
    elif brush_positions and len(brush_positions) >= key:
        x, y = brush_positions[key - 1]
        pyautogui.click(x, y)
        time.sleep(0.20)
    else:
        if focus_point:
            pyautogui.click(*focus_point)
            time.sleep(HOTKEY_FOCUS_DELAY)
        pyautogui.press(str(key))
        time.sleep(0.30)

    raise_if_stopped(stop_event)


def set_custom_rgb_color(rgb, controls, stop_event=None, panel_delay=None):
    raise_if_stopped(stop_event)

    controls = normalize_custom_rgb_controls(controls)
    if not controls:
        raise RuntimeError("尚未取得 RGB 面板位置，請先 Auto Detect，或用 Overlay 拖動校正 RGB / R / G / B 座標。")

    inputs = controls.get("inputs") or []
    if len(inputs) < 3:
        raise RuntimeError("RGB 輸入框座標不足，請用 Overlay 拖動校正 R/G/B 三個輸入框。")

    time.sleep(CUSTOM_RGB_SWATCH_CLICK_DELAY)
    pyautogui.click(*controls["swatch"])
    time.sleep(CUSTOM_RGB_PANEL_DELAY if panel_delay is None else panel_delay)

    for pos, value in zip(inputs[:3], rgb):
        raise_if_stopped(stop_event)
        pyautogui.click(*pos)
        time.sleep(CUSTOM_RGB_INPUT_CLICK_DELAY)
        pyautogui.hotkey("ctrl", "a")
        pyautogui.write(str(int(np.clip(value, 0, 255))), interval=CUSTOM_RGB_TYPE_INTERVAL)
        time.sleep(CUSTOM_RGB_INPUT_WRITE_DELAY)

    raise_if_stopped(stop_event)


def draw_stroke_path(points, move_seconds, stop_event=None):
    raise_if_stopped(stop_event)

    if len(points) < 2:
        if len(points) == 1:
            pyautogui.click(points[0][0], points[0][1])
        return

    pyautogui.moveTo(points[0][0], points[0][1], duration=0)
    time.sleep(0.02)
    pyautogui.mouseDown()
    time.sleep(0.02)

    try:
        for point in points[1:]:
            raise_if_stopped(stop_event)
            pyautogui.moveTo(point[0], point[1], duration=move_seconds)
    finally:
        time.sleep(0.02)
        pyautogui.mouseUp()


def build_color_runs(color_map, palette_size, bridge_gap=0):
    responsive = ResponsiveYield()
    runs_by_color = [[] for _ in range(palette_size)]
    pixel_counts = [0 for _ in range(palette_size)]
    draw_h, draw_w = color_map.shape
    bridge_gap = max(0, int(bridge_gap))

    for y in range(draw_h):
        if y % 12 == 0:
            responsive.maybe()
        row = color_map[y]
        x = 0

        while x < draw_w:
            color_idx = int(row[x])

            if color_idx < 0 or color_idx >= palette_size:
                x += 1
                continue

            start = x
            x += 1

            while x < draw_w and int(row[x]) == color_idx:
                x += 1

            if bridge_gap > 0:
                while True:
                    gap_start = x
                    gap_end = gap_start

                    while (
                        gap_end < draw_w
                        and int(row[gap_end]) < 0
                        and gap_end - gap_start < bridge_gap
                    ):
                        gap_end += 1

                    gap_len = gap_end - gap_start

                    if (
                        0 < gap_len <= bridge_gap
                        and gap_end < draw_w
                        and int(row[gap_end]) == color_idx
                        and not np.any(row[gap_start:gap_end] == PROTECTED_WHITE)
                    ):
                        x = gap_end + 1

                        while x < draw_w and int(row[x]) == color_idx:
                            x += 1

                        continue

                    break

            runs_by_color[color_idx].append((y, start, x))
            pixel_counts[color_idx] += x - start

    return runs_by_color, pixel_counts


def extract_runs_from_binary_mask(mask, bridge_gap=0, min_run_len=1):
    """Convert a binary mask into horizontal Gartic fill runs."""
    responsive = ResponsiveYield()
    mask = (mask > 0).astype(np.uint8)
    runs = []
    h, w = mask.shape
    bridge_gap = max(0, int(bridge_gap))
    min_run_len = max(1, int(min_run_len))

    for y in range(h):
        if y % 12 == 0:
            responsive.maybe()
        row = mask[y]
        x = 0

        while x < w:
            if row[x] == 0:
                x += 1
                continue

            start = x
            x += 1

            while x < w and row[x] > 0:
                x += 1

            if bridge_gap > 0:
                while True:
                    gap_start = x
                    gap_end = gap_start

                    while (
                        gap_end < w
                        and row[gap_end] == 0
                        and gap_end - gap_start < bridge_gap
                    ):
                        gap_end += 1

                    gap_len = gap_end - gap_start

                    if 0 < gap_len <= bridge_gap and gap_end < w and row[gap_end] > 0:
                        x = gap_end + 1

                        while x < w and row[x] > 0:
                            x += 1

                        continue

                    break

            end = x

            if end - start >= min_run_len:
                runs.append((y, start, end))

    return runs


def contour_simplify_mask(mask, min_area=8, epsilon_factor=0.006):
    """
    Simplify binary regions with cv2.findContours + RDP approxPolyDP.
    This keeps the original color-map idea, but removes stair-step noise on
    region edges before generating fill strokes.
    """
    mask = (mask > 0).astype(np.uint8)

    if not np.any(mask):
        return mask

    simplified = np.zeros_like(mask, dtype=np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None or not contours:
        return mask

    hierarchy = hierarchy[0]
    min_area = max(1, int(min_area))
    epsilon_factor = max(0.0, float(epsilon_factor))

    for i, contour in enumerate(contours):
        area = abs(cv2.contourArea(contour))

        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, True)
        epsilon = max(0.35, perimeter * epsilon_factor)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Parent contour = filled area; child contour = hole.
        fill_value = 1 if hierarchy[i][3] == -1 else 0
        cv2.drawContours(simplified, [approx], -1, fill_value, thickness=cv2.FILLED)

    # Avoid accidentally expanding outside the original region too much.
    # Keep a one-pixel tolerance so diagonal RDP edges can still become smooth.
    dilated_original = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    simplified = np.where(dilated_original > 0, simplified, 0).astype(np.uint8)

    if not np.any(simplified):
        return mask

    return simplified


def build_color_runs_contour(color_map, palette_size, brush_px=1, bridge_gap=0, optimize=True):
    """
    Contour-aware fill:
    color_map -> same-color masks -> contour/RDP smoothing -> horizontal fill runs
    -> nearest-neighbor run order with reverse drawing support.
    """
    runs_by_color = [[] for _ in range(palette_size)]
    pixel_counts = [0 for _ in range(palette_size)]
    brush_px = max(1, int(brush_px))
    bridge_gap = max(0, int(bridge_gap))
    protected_mask = color_map == PROTECTED_WHITE
    has_protected_white = bool(np.any(protected_mask))

    # Larger brushes can tolerate stronger simplification.
    epsilon_factor = 0.0025 if brush_px <= 3 else 0.0045 if brush_px <= 9 else 0.0065
    min_area = max(4, int(brush_px * brush_px * 0.75))
    min_run_len = 1

    for color_idx in range(palette_size):
        mask = (color_map == color_idx).astype(np.uint8)

        if not np.any(mask):
            continue

        if bridge_gap > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            if has_protected_white:
                mask[protected_mask] = 0

        pixel_counts[color_idx] = int(np.sum(mask))
        simplified = contour_simplify_mask(mask, min_area=min_area, epsilon_factor=epsilon_factor)
        if has_protected_white:
            simplified[protected_mask] = 0
        runs = extract_runs_from_binary_mask(
            simplified,
            bridge_gap=0 if has_protected_white else bridge_gap,
            min_run_len=min_run_len
        )
        runs_by_color[color_idx] = optimize_run_order(runs) if optimize else runs

    return runs_by_color, pixel_counts


def rotate_points_to_nearest(points, target_point=None):
    """Rotate a closed/open contour point list so it starts near target_point."""
    if not points:
        return points
    pts = list(points)
    closed = len(pts) > 2 and pts[0] == pts[-1]
    body = pts[:-1] if closed else pts
    if not body:
        return pts
    if target_point is None:
        start_idx = min(range(len(body)), key=lambda i: (body[i][1], body[i][0]))
    else:
        tx, ty = target_point
        start_idx = min(range(len(body)), key=lambda i: (body[i][0] - tx) ** 2 + (body[i][1] - ty) ** 2)
    rotated = body[start_idx:] + body[:start_idx]
    if closed:
        rotated.append(rotated[0])
    return rotated


def contour_to_points(contour, epsilon_factor=0.004, min_points=6):
    """Convert a contour to a simplified closed point path."""
    if contour is None or len(contour) < 3:
        return []
    perimeter = cv2.arcLength(contour, True)
    epsilon = max(0.35, perimeter * float(epsilon_factor))
    approx = cv2.approxPolyDP(contour, epsilon, True)
    pts = [(int(p[0][0]), int(p[0][1])) for p in approx]
    if len(pts) < min_points:
        raw = contour.reshape(-1, 2)
        if len(raw) == 0:
            return []
        step = max(1, len(raw) // max(min_points, 18))
        pts = [(int(x), int(y)) for x, y in raw[::step]]
    if len(pts) >= 2 and pts[0] != pts[-1]:
        pts.append(pts[0])
    return pts


def component_to_spiral_path(component_mask, brush_px=3, cell_step_px=1, max_loops=160):
    """
    Convert one same-color component into a mosquito-coil / spiral-like path.
    It draws the outer contour, repeatedly erodes inward, and connects each
    inner loop to the nearest point so the mouse can stay down most of the time.
    """
    component_mask = (component_mask > 0).astype(np.uint8)
    if not np.any(component_mask):
        return []

    cell_step_px = max(1, int(cell_step_px))
    brush_px = max(1, int(brush_px))
    erode_iter = max(1, int(round((brush_px / cell_step_px) * 0.42)))
    kernel_size = 2 * erode_iter + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    cur = component_mask.copy()
    path = []
    current_end = None
    responsive = ResponsiveYield()

    for depth in range(int(max_loops)):
        responsive.maybe()
        contours, _hierarchy = cv2.findContours(cur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [c for c in contours if cv2.contourArea(c) >= 2.0]
        if not contours:
            break
        contours.sort(key=cv2.contourArea, reverse=True)
        for contour in contours[:4]:
            pts = contour_to_points(contour, epsilon_factor=0.0035 if depth < 4 else 0.006)
            if len(pts) < 3:
                continue
            pts = rotate_points_to_nearest(pts, current_end)
            if path and pts:
                path.append(pts[0])
            path.extend(pts)
            current_end = path[-1]

        nxt = cv2.erode(cur, kernel, iterations=1)
        if np.array_equal(nxt, cur):
            break
        cur = nxt

    compact = []
    for pt in path:
        if not compact or compact[-1] != pt:
            compact.append(pt)
    return compact


def build_spiral_fill_paths(color_map, palette_size, brush_px=3, cell_step_px=1, min_area=None, remove_from_fallback=True):
    """
    Build spiral fill paths for large simple same-color components.

    When remove_from_fallback is False, the spiral pass becomes an extra texture
    / reinforcement pass and the normal scanline fill still paints the full
    region.  This is safer for Palette mode, where replacing a large flat color
    with only a spiral path can look patchy on Gartic.
    """
    brush_px = max(1, int(brush_px))
    cell_step_px = max(1, int(cell_step_px))
    if min_area is None:
        min_area = max(90, int((brush_px / cell_step_px) ** 2 * 18))

    spiral_paths_by_color = [[] for _ in range(palette_size)]
    fallback = color_map.copy()
    accepted_components = 0
    rejected_components = 0
    covered_pixels = 0
    responsive = ResponsiveYield()

    for color_idx in range(palette_size):
        responsive.maybe()
        mask = (color_map == color_idx).astype(np.uint8)
        if not np.any(mask):
            continue
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
        for label in range(1, num_labels):
            responsive.maybe()
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                rejected_components += 1
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            ww = int(stats[label, cv2.CC_STAT_WIDTH])
            hh = int(stats[label, cv2.CC_STAT_HEIGHT])
            if ww < 4 or hh < 4:
                rejected_components += 1
                continue
            thinness = area / max(1, ww * hh)
            if thinness < 0.08:
                rejected_components += 1
                continue

            component = (labels[y:y + hh, x:x + ww] == label).astype(np.uint8)
            _contours, hierarchy = cv2.findContours(component, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            child_count = 0
            if hierarchy is not None:
                child_count = int(np.sum(hierarchy[0][:, 3] != -1))
            if child_count > 2:
                rejected_components += 1
                continue

            local_path = component_to_spiral_path(component, brush_px=brush_px, cell_step_px=cell_step_px)
            if len(local_path) < 8:
                rejected_components += 1
                continue
            global_path = [(px + x, py + y) for px, py in local_path]
            spiral_paths_by_color[color_idx].append(global_path)
            if remove_from_fallback:
                fallback[labels == label] = -1
            accepted_components += 1
            covered_pixels += area

    stats_out = {
        "spiral_components": accepted_components,
        "fallback_components": rejected_components,
        "spiral_paths": sum(len(paths) for paths in spiral_paths_by_color),
        "covered_pixels": covered_pixels,
    }
    return spiral_paths_by_color, fallback, stats_out


def optimize_spiral_path_order(paths):
    responsive = ResponsiveYield()
    paths = [p for p in paths if len(p) >= 2]
    if len(paths) <= 2:
        return paths
    remaining = list(range(len(paths)))
    start_pos = max(range(len(remaining)), key=lambda pos: len(paths[remaining[pos]]))
    first_idx = remaining.pop(start_pos)
    ordered = [paths[first_idx]]
    current_end = ordered[-1][-1]
    while remaining:
        responsive.maybe()
        best_pos = 0
        best_reverse = False
        best_dist = float("inf")
        for pos, idx in enumerate(remaining):
            path = paths[idx]
            sx, sy = path[0]
            ex, ey = path[-1]
            d_start = (current_end[0] - sx) ** 2 + (current_end[1] - sy) ** 2
            d_end = (current_end[0] - ex) ** 2 + (current_end[1] - ey) ** 2
            if d_start < best_dist:
                best_dist = d_start
                best_pos = pos
                best_reverse = False
            if d_end < best_dist:
                best_dist = d_end
                best_pos = pos
                best_reverse = True
        path = paths[remaining.pop(best_pos)]
        if best_reverse:
            path = list(reversed(path))
        ordered.append(path)
        current_end = path[-1]
    return ordered


def draw_spiral_screen_path(points, stop_event=None):
    """Draw one long same-color spiral path with minimal mouse lifts."""
    raise_if_stopped(stop_event)
    if len(points) < 2:
        if len(points) == 1:
            pyautogui.click(points[0][0], points[0][1])
        return
    pyautogui.moveTo(points[0][0], points[0][1], duration=0)
    time.sleep(0.006)
    pyautogui.mouseDown()
    try:
        last = points[0]
        for point in points[1:]:
            raise_if_stopped(stop_event)
            dx = point[0] - last[0]
            dy = point[1] - last[1]
            dist = (dx * dx + dy * dy) ** 0.5
            duration = max(0.0008, min(0.0045, dist / 18000.0))
            pyautogui.moveTo(point[0], point[1], duration=duration)
            last = point
    finally:
        time.sleep(0.004)
        pyautogui.mouseUp()


def normalize_run(run):
    if len(run) == 4:
        return run

    y, start, end = run
    return y, start, end, False


def run_endpoints(run):
    y, start, end, reverse = normalize_run(run)

    if reverse:
        return (end - 1, y), (start, y)

    return (start, y), (end - 1, y)


def run_air_distance(runs):
    if len(runs) <= 1:
        return 0.0

    total = 0.0

    for prev, cur in zip(runs, runs[1:]):
        _prev_start, prev_end = run_endpoints(prev)
        cur_start, _cur_end = run_endpoints(cur)
        total += ((prev_end[0] - cur_start[0]) ** 2 + (prev_end[1] - cur_start[1]) ** 2) ** 0.5

    return total


def optimize_run_order(runs):
    """
    Nearest-neighbor ordering for same-color horizontal runs.
    Runs may be drawn left-to-right or right-to-left to reduce air travel.
    """
    if len(runs) <= 2:
        return [normalize_run(run) for run in runs]

    remaining = list(range(len(runs)))
    start_pos = max(
        range(len(remaining)),
        key=lambda pos: normalize_run(runs[remaining[pos]])[2] - normalize_run(runs[remaining[pos]])[1]
    )
    start_idx = remaining.pop(start_pos)
    y, start, end, _reverse = normalize_run(runs[start_idx])
    ordered = [(y, start, end, False)]
    _current_start, current_end = run_endpoints(ordered[-1])
    responsive = ResponsiveYield()

    while remaining:
        responsive.maybe()
        best_pos = 0
        best_reverse = False
        best_dist = float("inf")

        for pos, idx in enumerate(remaining):
            run_y, run_start, run_end, _run_reverse = normalize_run(runs[idx])
            forward_start = (run_start, run_y)
            reverse_start = (run_end - 1, run_y)
            d_forward = (current_end[0] - forward_start[0]) ** 2 + (current_end[1] - forward_start[1]) ** 2
            d_reverse = (current_end[0] - reverse_start[0]) ** 2 + (current_end[1] - reverse_start[1]) ** 2

            if d_forward < best_dist:
                best_dist = d_forward
                best_pos = pos
                best_reverse = False

            if d_reverse < best_dist:
                best_dist = d_reverse
                best_pos = pos
                best_reverse = True

        run_y, run_start, run_end, _run_reverse = normalize_run(runs[remaining.pop(best_pos)])
        run = (run_y, run_start, run_end, best_reverse)
        ordered.append(run)
        _current_start, current_end = run_endpoints(run)

    return ordered


# ============================================================================
# Image Processing And Preview
# ============================================================================

def resize_keep_aspect(img_rgba, max_w, max_h):
    w, h = img_rgba.size
    scale = min(max_w / w, max_h / h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    return img_rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)


def pil_rgb_alpha_arrays(img, dtype=np.uint8):
    """Return RGB and alpha arrays for RGB/RGBA/P-mode images."""
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba, dtype=dtype)
    return arr[:, :, :3], arr[:, :, 3]


def protected_white_shape_mask(rgb, alpha, white_threshold=248, chroma_limit=22, min_area=None, dilate_px=0):
    """
    Preserve large white logo/text shapes as untouched canvas.

    For logos like osu!, white letters are the foreground, not background.
    They should remain blank canvas, and later fill/hole repair must not paint
    over them. Small highlights are intentionally ignored.
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    alpha = np.asarray(alpha, dtype=np.uint8)
    h, w = alpha.shape

    white = (
        (alpha >= 40)
        & (np.min(rgb, axis=2) >= white_threshold)
        & ((np.max(rgb, axis=2) - np.min(rgb, axis=2)) <= chroma_limit)
    )

    if not np.any(white):
        return np.zeros(alpha.shape, dtype=bool)

    if min_area is None:
        min_area = max(24, int(h * w * 0.00050))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white.astype(np.uint8), 8)
    protected = np.zeros(alpha.shape, dtype=bool)

    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= int(min_area):
            protected[labels == label] = True

    dilate_px = max(0, int(dilate_px))
    if dilate_px > 0 and np.any(protected):
        kernel_size = dilate_px * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        protected = cv2.dilate(protected.astype(np.uint8), kernel, iterations=1).astype(bool)

    return protected


def image_to_palette_map(img_rgba, palette_colors, max_w, max_h, skip_white=True, white_protect_radius=0):
    img = resize_keep_aspect(img_rgba, max_w, max_h)
    rgb, alpha = pil_rgb_alpha_arrays(img, dtype=np.int32)

    idx = nearest_color_index_map(rgb, palette_colors)

    skip = alpha < 40
    protected_white = np.zeros(alpha.shape, dtype=bool)

    if skip_white:
        # 只跳過「連到圖片邊界的白背景」與大型白色前景形狀。
        # 舊版會把圖片內部接近白色的高光也全部 skip，
        # 全彩模式就容易出現一條一條的白色空隙。
        protected_white = protected_white_shape_mask(
            rgb.astype(np.uint8),
            alpha.astype(np.uint8),
            dilate_px=white_protect_radius
        )
        skip = skip | background_white_mask(rgb.astype(np.uint8), alpha.astype(np.uint8))

        idx = remap_palette_white_cells(
            idx,
            rgb.astype(np.uint8),
            alpha.astype(np.uint8),
            palette_colors,
            skip,
            protected_white,
        )

    idx[skip] = -1
    idx[protected_white] = PROTECTED_WHITE

    return idx, img.size


def detect_eye_detail_mask(rgb, alpha):
    """
    Detect compact anime face details: eyes, eyebrows, mouth, blush, nose and
    tiny dark/red strokes.  Custom RGB with very high color counts can otherwise
    merge these small regions back into skin/hair colors or paint them too early.

    The detector is conservative: it only keeps compact components in the upper
    character area, so large red clothing/hood regions are rejected.
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    alpha = np.asarray(alpha, dtype=np.uint8)
    h, w = alpha.shape

    if h <= 2 or w <= 2:
        return np.zeros((h, w), dtype=bool)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    yy, xx = np.mgrid[0:h, 0:w]
    alpha_mask = alpha >= 40

    # Most anime face details sit around the upper/center part of the image.
    # Keep mouth handling narrower than eye handling so cheek/face shadows do
    # not get promoted into hard final-pass strokes.
    eye_roi = (
        alpha_mask
        & (yy >= int(h * 0.10))
        & (yy <= int(h * 0.58))
        & (xx >= int(w * 0.06))
        & (xx <= int(w * 0.94))
    )
    mouth_roi = (
        alpha_mask
        & (yy >= int(h * 0.36))
        & (yy <= int(h * 0.66))
        & (xx >= int(w * 0.24))
        & (xx <= int(w * 0.76))
    )

    red_channel = rgb[:, :, 0].astype(np.int16)
    green_channel = rgb[:, :, 1].astype(np.int16)
    blue_channel = rgb[:, :, 2].astype(np.int16)

    # Mouth / blush / warm eye colors.  The lower thresholds are intentional:
    # after resizing and bilateral filtering, mouth pixels often become soft
    # pink and would otherwise be treated as normal skin.
    pink_or_red = (
        (red_channel > green_channel + 10)
        & (red_channel > blue_channel + 7)
        & (sat > 18)
        & (val > 85)
        & (val < 252)
    )
    hue_warm = (
        ((hue <= 24) | (hue >= 165) | ((hue >= 5) & (hue <= 34)))
        & (sat > 24)
        & (val > 60)
        & (val < 252)
    )

    # Eyes, eyelashes, eyebrows, nostril/mouth outline.
    dark_feature = (gray < 132) & alpha_mask
    soft_line = (gray < 158) & (sat > 24) & alpha_mask

    candidate = eye_roi & (pink_or_red | hue_warm | dark_feature | soft_line)
    candidate |= mouth_roi & (pink_or_red | hue_warm | (gray < 135))
    candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, 8)
    keep = np.zeros((h, w), dtype=np.uint8)
    max_area = max(8, int(h * w * 0.012))
    responsive = ResponsiveYield()

    for label in range(1, num_labels):
        responsive.maybe()
        x, y, ww, hh, area = stats[label]
        cx, cy = centroids[label]

        if area < 2:
            continue
        if area > max_area:
            continue
        # Reject large pieces of hood/hair/clothes.  Mouth/eye/blush pieces are
        # compact, even when the image is resized large.
        if ww > w * 0.30 or hh > h * 0.20:
            continue
        if cy < h * 0.11 or cy > h * 0.72:
            continue
        # Lower face details should be tiny. This prevents cheek/chin shading on
        # the lower-right face from being repainted as a hard facial feature.
        if cy > h * 0.52 and (ww > w * 0.16 or hh > h * 0.08 or area > max(10, int(h * w * 0.0035))):
            continue

        keep[labels == label] = 1

    if not np.any(keep):
        return keep.astype(bool)

    near_detail = cv2.dilate(keep, np.ones((5, 5), np.uint8), iterations=1) > 0
    small_highlight = (
        eye_roi
        & near_detail
        & (np.min(rgb, axis=2) > 205)
        & ((np.max(rgb, axis=2) - np.min(rgb, axis=2)) < 70)
    )
    nearby_colored = near_detail & ((eye_roi & (sat > 24)) | (mouth_roi & (pink_or_red | hue_warm))) & (val < 252)
    nearby_dark = near_detail & (eye_roi | mouth_roi) & (gray < 150)

    detail = (keep > 0) | small_highlight | nearby_colored | nearby_dark
    detail = cv2.morphologyEx(detail.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, 1), np.uint8))
    return detail.astype(bool)

def image_to_eye_detail_map(img_rgba, palette_colors, max_w, max_h):
    """
    High-resolution final pass for eyes / mouth / face details.
    Returns a color_map where non-detail pixels are -1.  Unlike global Skip White,
    white eye highlights are allowed so they can be restored after red/dark fills.
    """
    img = resize_keep_aspect(img_rgba, max_w, max_h)
    rgb, alpha = pil_rgb_alpha_arrays(img, dtype=np.uint8)

    if not palette_colors:
        return np.full(rgb.shape[:2], -1, dtype=np.int16), img.size

    mask = detect_eye_detail_mask(rgb, alpha)
    if not np.any(mask):
        return np.full(rgb.shape[:2], -1, dtype=np.int16), img.size

    idx = nearest_color_index_map(rgb, palette_colors)
    idx[~mask] = -1
    idx[alpha < 40] = -1
    return idx, img.size


def resize_label_map_to_shape(label_map, target_shape, fill_value=-1):
    """Resize/crop a label map to exactly match another map shape."""
    target_h, target_w = int(target_shape[0]), int(target_shape[1])

    if label_map.shape == (target_h, target_w):
        return label_map

    if target_h <= 0 or target_w <= 0:
        return np.full((max(1, target_h), max(1, target_w)), fill_value, dtype=label_map.dtype)

    resized = cv2.resize(
        label_map.astype(np.float32),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST
    )
    return np.rint(resized).astype(label_map.dtype)


def background_white_mask(rgb, alpha, very_white_threshold=248, chroma_limit=18):
    """
    只跳過「連到圖片邊界的純白背景」。
    不再把角色臉、衣服反光、淡色頭髮一起當背景刪掉。
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    alpha = np.asarray(alpha, dtype=np.uint8)

    very_white = (
        (alpha >= 40)
        & (np.min(rgb, axis=2) >= very_white_threshold)
        & ((np.max(rgb, axis=2) - np.min(rgb, axis=2)) <= chroma_limit)
    )

    if not np.any(very_white):
        return np.zeros(alpha.shape, dtype=bool)

    num_labels, labels = cv2.connectedComponents(very_white.astype(np.uint8), 8)
    if num_labels <= 1:
        return np.zeros(alpha.shape, dtype=bool)

    border_labels = set(labels[0, :].tolist())
    border_labels.update(labels[-1, :].tolist())
    border_labels.update(labels[:, 0].tolist())
    border_labels.update(labels[:, -1].tolist())
    border_labels.discard(0)

    if not border_labels:
        return np.zeros(alpha.shape, dtype=bool)

    return np.isin(labels, list(border_labels))


def image_to_custom_rgb_map(img_rgba, max_w, max_h, color_count=24, skip_white=True, white_protect_radius=0):
    """
    高還原 Custom RGB：
    - 只移除邊界純白背景，不吃掉臉/衣服高光。
    - 高色數時不做 label medianBlur，避免眼睛、嘴巴、細線被抹掉。
    - 小色塊清理改很保守，保留動漫圖細節。
    """
    img = resize_keep_aspect(img_rgba, max_w, max_h)
    rgb, alpha = pil_rgb_alpha_arrays(img, dtype=np.uint8)

    # 輕微降噪即可；原本太強會把眼睛/臉部小色塊糊掉。
    rgb_smooth = cv2.bilateralFilter(rgb, 5, 28, 28)

    skip = alpha < 40
    protected_white = np.zeros(alpha.shape, dtype=bool)

    if skip_white:
        protected_white = protected_white_shape_mask(
            rgb_smooth,
            alpha,
            dilate_px=white_protect_radius
        )
        skip = skip | background_white_mask(rgb_smooth, alpha) | protected_white

    # 深色線條與五官不准被 skip_white 影響。
    gray = cv2.cvtColor(rgb_smooth, cv2.COLOR_RGB2GRAY)
    dark_detail = (gray < 170) & (alpha >= 40)
    skip[dark_detail] = False

    # 高色數時，小嘴巴 / 臉紅 / 眼睛只佔很少像素，隨機 k-means sample
    # 很容易沒抽到，最後就會被附近膚色吃掉。先抓出臉部細節，後面
    # 讓它們在取色樣本中有更高權重，並在清小碎塊後強制補回。
    face_detail_mask = detect_eye_detail_mask(rgb_smooth, alpha) & (~skip)
    face_detail_pixels = rgb[face_detail_mask]

    pixels = rgb_smooth[~skip]

    if len(pixels) == 0:
        return np.full(rgb.shape[:2], -1, dtype=np.int16), [], img.size

    color_count = int(np.clip(color_count, 2, MAX_CUSTOM_COLORS))
    k = min(color_count, len(pixels))

    # 用隨機但固定 seed 的 sample，比 linspace 更不容易偏向圖片某一側。
    rng = np.random.default_rng(12345)
    if len(pixels) > 50000:
        sample_idx = rng.choice(len(pixels), 50000, replace=False)
        samples = pixels[sample_idx]
    else:
        samples = pixels

    if len(face_detail_pixels) > 0:
        detail_limit = 5000 if color_count >= 128 else 2500
        if len(face_detail_pixels) > detail_limit:
            detail_idx = rng.choice(len(face_detail_pixels), detail_limit, replace=False)
            detail_samples = face_detail_pixels[detail_idx]
        else:
            detail_samples = face_detail_pixels

        if color_count >= 256:
            detail_weight = 6
        elif color_count >= 128:
            detail_weight = 4
        elif color_count >= 64:
            detail_weight = 3
        else:
            detail_weight = 2

        samples = np.concatenate([
            samples,
            np.repeat(detail_samples, detail_weight, axis=0)
        ], axis=0)

    samples = np.float32(samples)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 35, 0.6)
    _compactness, _labels, centers = cv2.kmeans(
        samples,
        k,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS
    )

    centers_i = np.clip(np.round(centers), 0, 255).astype(np.int32)
    idx = nearest_color_index_map(rgb_smooth, centers_i)
    idx[skip] = -1

    # 只有低色數才做 medianBlur；48 色高還原時不抹細節。
    valid = idx >= 0
    if color_count <= 24:
        shifted = np.where(valid, idx + 1, 0).astype(np.uint8)
        shifted = cv2.medianBlur(shifted, 3)
        idx = shifted.astype(np.int16) - 1
        idx[~valid] = -1
        idx[skip] = -1

    # 小區塊清理變保守，避免眼睛/嘴巴/髮絲被刪掉。
    if color_count >= 128:
        min_area = 1
    elif color_count >= 40:
        min_area = 2
    elif color_count >= 32:
        min_area = 3
    elif color_count >= 24:
        min_area = 4
    else:
        min_area = max(4, int(max_w * max_h * 0.00018))

    idx = remove_small_color_regions(idx, k, min_area)

    if np.any(face_detail_mask):
        # 清小區塊可能會把嘴巴、腮紅、眼神光這種很小的顏色刪掉；
        # 這裡用原始 RGB 再量化一次，把臉部細節補回最後的 map。
        face_detail_idx = nearest_color_index_map(rgb, centers_i)
        idx[face_detail_mask] = face_detail_idx[face_detail_mask]

    idx[protected_white] = PROTECTED_WHITE

    colors = [tuple(int(v) for v in color) for color in centers_i]
    return idx, colors, img.size


def remove_small_color_regions(color_map, palette_size, min_area):
    cleaned = np.full(color_map.shape, -1, dtype=np.int16)

    responsive = ResponsiveYield()

    for color_idx in range(palette_size):
        responsive.maybe()
        mask = (color_map == color_idx).astype(np.uint8)
        if not np.any(mask):
            continue

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == label] = color_idx

    return cleaned


def solidify_color_map(color_map, palette_size, max_hole_area=12):
    """
    Fill tiny skipped holes with the dominant neighboring color.
    Large white/background areas are kept untouched, so faces and blank canvas
    do not get flooded accidentally.
    """
    result = color_map.copy()
    h, w = result.shape
    invalid = (result < 0).astype(np.uint8)

    if not np.any(invalid):
        return result

    responsive = ResponsiveYield()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(invalid, 8)
    max_hole_area = max(1, int(max_hole_area))
    kernel = np.ones((3, 3), np.uint8)

    for label in range(1, num_labels):
        x, y, ww, hh, area = stats[label]
        component_values = result[labels == label]

        if np.any(component_values == PROTECTED_WHITE):
            continue

        if area > max_hole_area:
            continue

        if x == 0 or y == 0 or x + ww >= w or y + hh >= h:
            continue

        component = (labels == label).astype(np.uint8)
        ring = (cv2.dilate(component, kernel, iterations=1) > 0) & (component == 0)
        neighbor_values = result[ring]
        neighbor_values = neighbor_values[
            (neighbor_values >= 0) & (neighbor_values < palette_size)
        ]

        if len(neighbor_values) == 0:
            continue

        counts = np.bincount(neighbor_values.astype(np.int32), minlength=palette_size)
        fill_color = int(np.argmax(counts))

        if counts[fill_color] >= max(2, len(neighbor_values) * 0.45):
            result[labels == label] = fill_color

    return result


def image_to_cartoon_color_map(img_rgba, palette_colors, max_w, max_h, skip_white=True, white_protect_radius=0):
    img = resize_keep_aspect(img_rgba, max_w, max_h)
    rgb, alpha = pil_rgb_alpha_arrays(img, dtype=np.uint8)
    rgb = cv2.bilateralFilter(rgb, 7, 45, 45)
    idx = nearest_color_index_map(rgb, palette_colors)

    skip = alpha < 40
    protected_white = np.zeros(alpha.shape, dtype=bool)

    if skip_white:
        protected_white = protected_white_shape_mask(
            rgb,
            alpha,
            dilate_px=white_protect_radius
        )
        skip = skip | background_white_mask(rgb, alpha)
        idx = remap_palette_white_cells(idx, rgb, alpha, palette_colors, skip, protected_white)

    idx[skip] = -1

    min_area = max(4, int(max_w * max_h * 0.00045))
    idx = remove_small_color_regions(idx, len(palette_colors), min_area)
    idx[protected_white] = PROTECTED_WHITE

    return idx, img.size


def image_to_line_strokes(img_rgba, max_w, max_h, detail=3):
    img = resize_keep_aspect(img_rgba, max_w, max_h)
    rgb, alpha = pil_rgb_alpha_arrays(img, dtype=np.float32)
    alpha = alpha[:, :, None] / 255.0
    rgb_on_white = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    gray = cv2.cvtColor(rgb_on_white, cv2.COLOR_RGB2GRAY)

    detail = int(np.clip(detail, 1, 5))
    threshold = 135 + detail * 20
    mask = (gray < threshold).astype(np.uint8)

    if detail <= 2:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    elif detail >= 4:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    skeleton = zhang_suen_thinning(mask)
    min_points = max(3, 8 - detail)
    epsilon = max(0.15, 0.85 - detail * 0.1)
    strokes = trace_skeleton_strokes(skeleton, min_points=min_points)
    strokes = [simplify_stroke(stroke, epsilon) for stroke in strokes]
    strokes = [densify_stroke(stroke, max_gap=1.25) for stroke in strokes]
    strokes = [stroke for stroke in strokes if len(stroke) >= 2]
    strokes.sort(key=len, reverse=True)

    return strokes, img.size


def image_to_clean_line_strokes(img_rgba, max_w, max_h, detail=3):
    detail = int(np.clip(detail, 1, 5))
    strokes, size = image_to_line_strokes(img_rgba, max_w, max_h, detail=5)
    w, h = size

    def stroke_score(stroke):
        xs = [point[0] for point in stroke]
        ys = [point[1] for point in stroke]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        score = len(stroke)

        # Preserve face / hair / upper-body detail better than plain length sorting.
        if 0.22 * w <= cx <= 0.78 * w and 0.12 * h <= cy <= 0.62 * h:
            score += 220
        if 0.32 * w <= cx <= 0.68 * w and 0.18 * h <= cy <= 0.48 * h:
            score += 260

        return score

    strokes = sorted(strokes, key=stroke_score, reverse=True)
    max_strokes = {1: 500, 2: 900, 3: 1300, 4: 1700, 5: 2200}[detail]

    return strokes[:max_strokes], size


def image_to_dark_outline_strokes(img_rgba, max_w, max_h, detail=3):
    img = resize_keep_aspect(img_rgba, max_w, max_h)
    rgb, alpha = pil_rgb_alpha_arrays(img, dtype=np.float32)
    alpha = alpha[:, :, None] / 255.0
    rgb_on_white = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    gray = cv2.cvtColor(rgb_on_white, cv2.COLOR_RGB2GRAY)

    detail = int(np.clip(detail, 1, 5))
    threshold = 55 + detail * 12
    mask = (gray < threshold).astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = max(8, 80 - detail * 10)
    cleaned = np.zeros_like(mask)

    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 1

    skeleton = zhang_suen_thinning(cleaned)
    min_points = max(3, 9 - detail)
    epsilon = max(0.2, 1.0 - detail * 0.12)
    strokes = trace_skeleton_strokes(skeleton, min_points=min_points)
    strokes = [simplify_stroke(stroke, epsilon) for stroke in strokes]
    strokes = [densify_stroke(stroke, max_gap=1.25) for stroke in strokes]
    strokes = [stroke for stroke in strokes if len(stroke) >= 2]
    strokes.sort(key=len, reverse=True)

    return strokes, img.size


def strokes_to_binary_mask(strokes, size, width=1):
    """Render stroke lists into a binary mask for merging line modes."""
    w, h = size
    mask = np.zeros((h, w), dtype=np.uint8)
    width = max(1, int(width))

    for stroke in strokes:
        if len(stroke) < 2:
            continue

        pts = np.asarray(stroke, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(mask, [pts], False, 1, thickness=width, lineType=cv2.LINE_8)

    return mask


def image_to_smart_line_strokes(img_rgba, max_w, max_h, detail=3):
    """
    Smart Line Art 整合線稿：
    把 Black Line Art、Clean Line、Dark Outline 三種線稿結果融合成一個模式。

    - Black Line Art：補細節線條
    - Clean Line：保留主要線條、減少雜線
    - Dark Outline：強化深色外輪廓
    - 最後重新骨架化、RDP 簡化、TSP 排序前的 stroke 產生
    """
    detail = int(np.clip(detail, 1, 5))

    line_strokes, size = image_to_line_strokes(
        img_rgba,
        max_w,
        max_h,
        detail=detail
    )
    clean_strokes, _ = image_to_clean_line_strokes(
        img_rgba,
        max_w,
        max_h,
        detail=detail
    )
    dark_strokes, _ = image_to_dark_outline_strokes(
        img_rgba,
        max_w,
        max_h,
        detail=detail
    )

    mask = np.zeros((size[1], size[0]), dtype=np.uint8)

    # 一般線條提供細節；精簡線稿和深色描邊用略粗一點併入，避免重要輪廓斷掉。
    mask |= strokes_to_binary_mask(line_strokes, size, width=1)
    mask |= strokes_to_binary_mask(clean_strokes, size, width=1)
    mask |= strokes_to_binary_mask(dark_strokes, size, width=1 if detail <= 2 else 2)

    if detail <= 2:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    elif detail >= 4:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    # 移除極小碎線，保留主線與五官細節。
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    cleaned = np.zeros_like(mask)
    min_area = max(2, 8 - detail)

    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 1

    if not np.any(cleaned):
        cleaned = (mask > 0).astype(np.uint8)

    skeleton = zhang_suen_thinning(cleaned)
    min_points = max(3, 8 - detail)
    epsilon = max(0.12, 0.75 - detail * 0.09)
    strokes = trace_skeleton_strokes(skeleton, min_points=min_points)
    strokes = [simplify_stroke(stroke, epsilon) for stroke in strokes]
    strokes = [densify_stroke(stroke, max_gap=1.25) for stroke in strokes]
    strokes = [stroke for stroke in strokes if len(stroke) >= 2]
    strokes.sort(key=len, reverse=True)

    return strokes, size


def nearest_palette_index(color, palette_colors):
    colors = np.asarray(palette_colors, dtype=np.int32)
    target = np.asarray(color, dtype=np.int32)
    dist = np.sum((colors - target) ** 2, axis=1)
    return int(np.argmin(dist))


def sbr_strokes_to_image(size, strokes, palette_colors):
    preview = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(preview)

    for color_idx, start, end, width in strokes:
        color = tuple(int(v) for v in palette_colors[color_idx])
        draw.line((start, end), fill=color, width=max(1, int(width)))

    return preview


def image_to_sbr_strokes(img_rgba, max_w, max_h, palette_colors, stroke_count=300, brush_px=3, skip_white=True):
    img = resize_keep_aspect(img_rgba, max_w, max_h)
    rgb, alpha = pil_rgb_alpha_arrays(img, dtype=np.float32)
    alpha = alpha[:, :, None] / 255.0
    target = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    h, w = target.shape[:2]

    skip = alpha[:, :, 0] < 0.15

    if skip_white:
        near_white = (
            np.min(target, axis=2) > 238
        ) & (
            np.max(target, axis=2) - np.min(target, axis=2) < 35
        )
        skip = skip | near_white

    palette_arr = np.asarray(palette_colors, dtype=np.uint8)
    white_indices = white_palette_indices(palette_colors) if skip_white else set()
    canvas = np.full_like(target, 255, dtype=np.uint8)
    strokes = []
    rng = np.random.default_rng(12345)
    stroke_count = int(np.clip(stroke_count, 25, 1500))
    brush_px = max(1, int(brush_px))

    gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_strength = cv2.magnitude(grad_x, grad_y)
    edge_max = float(np.max(edge_strength))
    if edge_max > 0:
        edge_norm = edge_strength / edge_max
    else:
        edge_norm = np.zeros((h, w), dtype=np.float32)
    edge_norm = cv2.GaussianBlur(edge_norm.astype(np.float32), (5, 5), 0)

    # Structure tensor gives a stable local line direction.  The stroke should
    # follow the tangent of the edge, not cut across it like random scratches.
    jxx = cv2.GaussianBlur(grad_x * grad_x, (7, 7), 0)
    jxy = cv2.GaussianBlur(grad_x * grad_y, (7, 7), 0)
    jyy = cv2.GaussianBlur(grad_y * grad_y, (7, 7), 0)
    tangent_angles = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy) + np.pi / 2.0
    orientation_strength = np.sqrt((jxx - jyy) ** 2 + 4.0 * (jxy ** 2))

    angle_offsets = np.deg2rad([-30, -15, 0, 15, 30])
    fallback_angles = np.deg2rad([0, 45, 90, 135])
    lengths = [
        max(2, brush_px),
        max(4, brush_px * 2),
        max(7, brush_px * 4),
        max(12, brush_px * 7),
    ]
    stop_error = 140 if brush_px <= 3 else 260

    responsive = ResponsiveYield()

    for _iteration in range(stroke_count):
        responsive.maybe()
        diff = target.astype(np.int16) - canvas.astype(np.int16)
        err = np.sum(diff * diff, axis=2).astype(np.float32)
        err[skip] = 0
        max_err = float(np.max(err))

        if max_err < stop_error:
            break

        importance = err * (1.0 + edge_norm * 2.0)
        importance[skip] = 0
        flat = importance.ravel()
        top_n = min(240, flat.size)
        top_indices = np.argpartition(flat, -top_n)[-top_n:]
        top_values = flat[top_indices]
        pool_size = min(18, top_n)
        pool = top_indices[np.argpartition(top_values, -pool_size)[-pool_size:]]
        chosen = int(rng.choice(pool))
        y, x = divmod(chosen, w)
        color_idx = nearest_palette_index(target[y, x], palette_colors)

        if color_idx in white_indices:
            skip[y, x] = True
            continue

        color = palette_arr[color_idx]
        best_score = 0.0
        best_mask = None
        best_line = None

        if orientation_strength[y, x] > 1.0:
            base_angle = float(tangent_angles[y, x])
        else:
            base_angle = float(fallback_angles[((x // 9) + (y // 13)) % len(fallback_angles)])

        for angle in (base_angle + angle_offsets):
            dx = float(np.cos(angle))
            dy = float(np.sin(angle))

            for length in lengths:
                half = length / 2.0
                x1 = int(round(np.clip(x - dx * half, 0, w - 1)))
                y1 = int(round(np.clip(y - dy * half, 0, h - 1)))
                x2 = int(round(np.clip(x + dx * half, 0, w - 1)))
                y2 = int(round(np.clip(y + dy * half, 0, h - 1)))

                if x1 == x2 and y1 == y2:
                    continue

                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.line(mask, (x1, y1), (x2, y2), 255, brush_px, lineType=cv2.LINE_8)
                mask_bool = (mask > 0) & (~skip)

                if not np.any(mask_bool):
                    continue

                before = np.sum((target[mask_bool].astype(np.int16) - canvas[mask_bool].astype(np.int16)) ** 2)
                after = np.sum((target[mask_bool].astype(np.int16) - color.astype(np.int16)) ** 2)
                score = float(before - after)

                if score > best_score:
                    best_score = score
                    best_mask = mask_bool
                    best_line = ((x1, y1), (x2, y2))

        if best_mask is None or best_score <= 0:
            skip[y, x] = True
            continue

        canvas[best_mask] = color
        strokes.append((color_idx, best_line[0], best_line[1], brush_px))

    preview = Image.fromarray(canvas, "RGB")
    return strokes, img.size, preview


def zhang_suen_thinning(mask):
    responsive = ResponsiveYield()
    img = (mask > 0).astype(np.uint8)
    changed = True

    while changed:
        changed = False

        for step in (0, 1):
            remove = []
            h, w = img.shape

            for y in range(1, h - 1):
                if y % 12 == 0:
                    responsive.maybe()
                for x in range(1, w - 1):
                    if img[y, x] == 0:
                        continue

                    p2 = img[y - 1, x]
                    p3 = img[y - 1, x + 1]
                    p4 = img[y, x + 1]
                    p5 = img[y + 1, x + 1]
                    p6 = img[y + 1, x]
                    p7 = img[y + 1, x - 1]
                    p8 = img[y, x - 1]
                    p9 = img[y - 1, x - 1]

                    neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                    count = sum(neighbors)

                    if count < 2 or count > 6:
                        continue

                    transitions = 0
                    loop = neighbors + [p2]
                    for i in range(8):
                        if loop[i] == 0 and loop[i + 1] == 1:
                            transitions += 1

                    if transitions != 1:
                        continue

                    if step == 0:
                        if p2 * p4 * p6 != 0 or p4 * p6 * p8 != 0:
                            continue
                    else:
                        if p2 * p4 * p8 != 0 or p2 * p6 * p8 != 0:
                            continue

                    remove.append((y, x))

            if remove:
                changed = True
                for y, x in remove:
                    img[y, x] = 0

    return img


def trace_skeleton_strokes(skeleton, min_points=4):
    responsive = ResponsiveYield()
    points = set(map(tuple, np.argwhere(skeleton > 0)))
    neighbor_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    def neighbors(point):
        y, x = point
        result = []
        for dy, dx in neighbor_offsets:
            candidate = (y + dy, x + dx)
            if candidate in points:
                result.append(candidate)
        return result

    degree = {point: len(neighbors(point)) for point in points}
    nodes = [point for point, count in degree.items() if count != 2]
    visited_edges = set()
    strokes = []

    def edge_key(a, b):
        return tuple(sorted((a, b)))

    def follow_path(start, first_next):
        path = [start]
        prev = start
        cur = first_next
        visited_edges.add(edge_key(prev, cur))

        while True:
            path.append(cur)

            if cur != start and degree.get(cur, 0) != 2:
                break

            next_candidates = [
                point for point in neighbors(cur)
                if point != prev and edge_key(cur, point) not in visited_edges
            ]

            if not next_candidates:
                break

            nxt = next_candidates[0]
            visited_edges.add(edge_key(cur, nxt))
            prev, cur = cur, nxt

        return path

    for node in nodes:
        responsive.maybe()
        for nxt in neighbors(node):
            if edge_key(node, nxt) in visited_edges:
                continue
            path = follow_path(node, nxt)
            if len(path) >= min_points:
                strokes.append([(x, y) for y, x in path])

    for point in points:
        responsive.maybe()
        unvisited = [
            nxt for nxt in neighbors(point)
            if edge_key(point, nxt) not in visited_edges
        ]

        for nxt in unvisited:
            path = follow_path(point, nxt)
            if len(path) >= min_points:
                strokes.append([(x, y) for y, x in path])

    return strokes


def simplify_stroke(stroke, epsilon):
    if len(stroke) <= 2:
        return stroke

    contour = np.asarray(stroke, dtype=np.float32).reshape((-1, 1, 2))
    simplified = cv2.approxPolyDP(contour, epsilon, False)
    return [(int(p[0][0]), int(p[0][1])) for p in simplified]


def densify_stroke(stroke, max_gap=1.5):
    if len(stroke) <= 1:
        return stroke

    result = [stroke[0]]

    for start, end in zip(stroke, stroke[1:]):
        x1, y1 = start
        x2, y2 = end
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        steps = max(1, int(np.ceil(distance / max_gap)))

        for step in range(1, steps + 1):
            t = step / steps
            x = int(round(x1 + (x2 - x1) * t))
            y = int(round(y1 + (y2 - y1) * t))

            if (x, y) != result[-1]:
                result.append((x, y))

    return result


def decimate_stroke(stroke, step=1):
    step = max(1, int(step))

    if step <= 1 or len(stroke) <= 2:
        return stroke

    result = stroke[::step]

    if result[-1] != stroke[-1]:
        result.append(stroke[-1])

    return result


def stroke_air_distance(strokes):
    if len(strokes) <= 1:
        return 0.0

    total = 0.0

    for prev, cur in zip(strokes, strokes[1:]):
        x1, y1 = prev[-1]
        x2, y2 = cur[0]
        total += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    return total


def optimize_stroke_order(strokes):
    """
    Nearest-neighbor TSP approximation over whole strokes.
    Each stroke may be reversed, which reduces non-drawing mouse travel.
    """
    if len(strokes) <= 2:
        return strokes

    remaining = list(range(len(strokes)))
    start_pos = max(range(len(remaining)), key=lambda pos: len(strokes[remaining[pos]]))
    start_idx = remaining.pop(start_pos)
    ordered = [strokes[start_idx]]
    current_end = ordered[-1][-1]
    responsive = ResponsiveYield()

    while remaining:
        responsive.maybe()
        best_pos = 0
        best_reverse = False
        best_dist = float("inf")

        for pos, idx in enumerate(remaining):
            stroke = strokes[idx]
            sx, sy = stroke[0]
            ex, ey = stroke[-1]
            d_start = (current_end[0] - sx) ** 2 + (current_end[1] - sy) ** 2
            d_end = (current_end[0] - ex) ** 2 + (current_end[1] - ey) ** 2

            if d_start < best_dist:
                best_dist = d_start
                best_pos = pos
                best_reverse = False

            if d_end < best_dist:
                best_dist = d_end
                best_pos = pos
                best_reverse = True

        stroke = strokes[remaining.pop(best_pos)]

        if best_reverse:
            stroke = list(reversed(stroke))

        ordered.append(stroke)
        current_end = stroke[-1]

    return ordered


def line_strokes_to_image(strokes, size, line_width=1):
    preview = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(preview)

    for stroke in strokes:
        if len(stroke) >= 2:
            draw.line(stroke, fill="black", width=line_width)

    return preview


def palette_map_to_image(color_map, palette_colors, scale=1):
    h, w = color_map.shape
    preview = np.full((h, w, 3), 255, dtype=np.uint8)
    palette_arr = np.asarray(palette_colors, dtype=np.uint8)

    for color_idx, color in enumerate(palette_arr):
        preview[color_map == color_idx] = color

    img = Image.fromarray(preview, "RGB")

    if scale > 1:
        img = img.resize((w * scale, h * scale), Image.Resampling.NEAREST)

    return img


def color_run_bridge_gap(mode, brush_px):
    # 保留 bridge_gap：把接近的同色色塊硬接起來。
    # 這樣可以減少斷線與小碎段，速度也會比較快。
    if mode in (MODE_PALETTE, MODE_CUSTOM_RGB):
        return 2 if brush_px <= 3 else 1

    return 0


def color_draw_order(palette_colors):
    """
    先畫淺色，最後畫深色。
    使用感知亮度排序，比單純 RGB 總和更穩。
    Gartic 沒有圖層，深色 / 線條最後畫比較不會被蓋掉。
    """
    def _luma(rgb):
        r, g, b = rgb
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    return sorted(
        range(len(palette_colors)),
        key=lambda i: (
            _luma(palette_colors[i]),
            sum(palette_colors[i]),
            max(palette_colors[i])
        ),
        reverse=True
    )


def color_fill_step(mode, brush_px):
    """
    Fill-line spacing.

    Palette / Custom RGB need dense overlap because Gartic's real brush size can
    be smaller than the hotkey estimate, especially after browser zoom or device
    scaling.  A sparse step makes color areas look like broken scratches instead
    of solid fills, so color modes intentionally use a tighter grid.
    """
    brush_px = max(1, int(brush_px))

    if mode in (MODE_PALETTE, MODE_CUSTOM_RGB):
        # 比原本再密一點，避免瀏覽器縮放 / Gartic 實際筆刷比估算小時
        # 產生水平白縫。代價是全彩會稍慢，但填色穩定很多。
        if brush_px <= 6:
            return 1
        if brush_px <= 9:
            return 2
        if brush_px <= 13:
            return 3
        return 4

    return brush_px


def white_protect_radius_for_brush(brush_px, cell_step_px):
    """
    Reserve a small no-paint margin around protected white logo/text shapes.
    Gartic's real brush bleeds farther than the center-point color map, so a
    one-cell guard keeps letters like the osu! "o" from being tinted.
    """
    brush_px = max(1, int(brush_px))
    cell_step_px = max(1, int(cell_step_px))
    return max(1, int(round((brush_px / cell_step_px) * 0.45)))


def color_hole_area(mode, brush_px):
    if mode in (MODE_PALETTE, MODE_CUSTOM_RGB):
        return max(4, int((max(1, brush_px) ** 2) * 1.5))

    return 0


def seal_thin_color_gaps(color_map, palette_size, max_thickness=2, max_area=None, dominant_ratio=0.58):
    """Fill small/thin skipped gaps with the dominant surrounding color.

    Palette mode can create white slits when pale pixels quantize to Gartic white
    and Skip White is enabled, or when very fast horizontal strokes miss tiny
    strips.  This only touches normal skipped cells (-1).  PROTECTED_WHITE stays
    untouched, so large white logos/text such as the osu! letters remain blank.
    """
    if color_map is None or palette_size <= 0:
        return color_map, 0

    result = color_map.copy()
    invalid = (result == -1).astype(np.uint8)
    if not np.any(invalid):
        return result, 0

    h, w = result.shape
    max_thickness = max(1, int(max_thickness))
    if max_area is None:
        max_area = max(10, int(max_thickness * max(h, w) * 1.20))
    max_area = max(1, int(max_area))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(invalid, 8)
    kernel = np.ones((3, 3), np.uint8)
    filled = 0
    responsive = ResponsiveYield()

    for label in range(1, num_labels):
        responsive.maybe()
        x, y, ww, hh, area = stats[label]
        # Keep large white objects/background.  Only seal tiny spots or very thin
        # horizontal/vertical cracks.
        if area > max_area and min(ww, hh) > max_thickness:
            continue
        if min(ww, hh) > max_thickness and area > max_area * 0.45:
            continue

        component = labels == label
        ring = (cv2.dilate(component.astype(np.uint8), kernel, iterations=1) > 0) & (~component)
        neighbor_values = result[ring]
        neighbor_values = neighbor_values[(neighbor_values >= 0) & (neighbor_values < palette_size)]
        if len(neighbor_values) < 2:
            continue

        counts = np.bincount(neighbor_values.astype(np.int32), minlength=palette_size)
        fill_color = int(np.argmax(counts))
        if counts[fill_color] >= max(2, int(len(neighbor_values) * float(dominant_ratio))):
            result[component] = fill_color
            filled += int(area)

    return result, filled


def color_map_to_gartic_preview(
    color_map,
    palette_colors,
    brush_px,
    reverse_order=False,
    bridge_gap=0,
    use_contour=True,
    cell_step_px=None,
    spiral_paths_by_color=None
):
    brush_px = max(1, int(brush_px))
    cell_step_px = max(1, int(cell_step_px or brush_px))
    h, w = color_map.shape
    preview_w = (w - 1) * cell_step_px + brush_px if w > 0 else brush_px
    preview_h = (h - 1) * cell_step_px + brush_px if h > 0 else brush_px
    preview = np.full((preview_h, preview_w, 3), 255, dtype=np.uint8)

    if not palette_colors:
        return Image.fromarray(preview, "RGB")

    display_map = color_map
    bridge_gap = max(0, int(bridge_gap))

    if use_contour or bridge_gap > 0:
        display_map = np.full(color_map.shape, -1, dtype=np.int16)
        epsilon_factor = 0.0025 if brush_px <= 3 else 0.0045 if brush_px <= 9 else 0.0065
        min_area = max(4, int(brush_px * brush_px * 0.75))
        protected_mask = color_map == PROTECTED_WHITE
        has_protected_white = bool(np.any(protected_mask))
        responsive = ResponsiveYield()

        for color_idx in range(len(palette_colors)):
            responsive.maybe()
            mask = (color_map == color_idx).astype(np.uint8)

            if not np.any(mask):
                continue

            if bridge_gap > 0:
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                if has_protected_white:
                    mask[protected_mask] = 0

            if use_contour:
                mask = contour_simplify_mask(mask, min_area=min_area, epsilon_factor=epsilon_factor)
                if has_protected_white:
                    mask[protected_mask] = 0

            display_map[mask > 0] = color_idx

    color_order = color_draw_order(palette_colors)
    center_w = (w - 1) * cell_step_px + 1 if w > 0 else 1
    center_h = (h - 1) * cell_step_px + 1 if h > 0 else 1
    offset = int(round(brush_px / 2.0))
    kernel_size = max(1, brush_px)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    responsive = ResponsiveYield()

    def paint_cell_mask(cell_mask, color):
        if not np.any(cell_mask):
            return

        if cell_step_px == 1:
            centers = (cell_mask.astype(np.uint8) * 255)
        else:
            centers = cv2.resize(
                (cell_mask.astype(np.uint8) * 255),
                (center_w, center_h),
                interpolation=cv2.INTER_NEAREST
            )

        layer = np.zeros((preview_h, preview_w), dtype=np.uint8)
        y2 = min(preview_h, offset + centers.shape[0])
        x2 = min(preview_w, offset + centers.shape[1])
        layer[offset:y2, offset:x2] = centers[:y2 - offset, :x2 - offset]

        if brush_px > 1:
            layer = cv2.dilate(layer, kernel, iterations=1)

        preview[layer > 0] = color

    def paint_spiral_path(path, color):
        if not path:
            return

        layer = np.zeros((preview_h, preview_w), dtype=np.uint8)
        points = np.asarray(
            [
                (
                    int(round(px * cell_step_px + offset)),
                    int(round(py * cell_step_px + offset))
                )
                for px, py in path
            ],
            dtype=np.int32
        )

        if len(points) == 1:
            cv2.circle(layer, tuple(points[0]), max(1, brush_px // 2), 255, thickness=-1)
        else:
            cv2.polylines(layer, [points.reshape((-1, 1, 2))], False, 255, thickness=brush_px, lineType=cv2.LINE_AA)

        preview[layer > 0] = color

    for color_idx in color_order:
        responsive.maybe()
        color = tuple(int(v) for v in palette_colors[color_idx])

        if spiral_paths_by_color is not None and color_idx < len(spiral_paths_by_color):
            for path in spiral_paths_by_color[color_idx]:
                paint_spiral_path(path, color)

        paint_cell_mask(display_map == color_idx, color)

    return Image.fromarray(preview, "RGB")


def format_estimated_seconds(seconds):
    seconds = max(0.0, float(seconds))

    if seconds < 1:
        return "<1 秒"

    total = int(round(seconds))

    if total < 60:
        return f"約 {total} 秒"

    minutes, secs = divmod(total, 60)

    if minutes < 60:
        if secs:
            return f"約 {minutes} 分 {secs:02d} 秒"
        return f"約 {minutes} 分"

    hours, minutes = divmod(minutes, 60)
    if minutes:
        return f"約 {hours} 小時 {minutes:02d} 分"
    return f"約 {hours} 小時"


def preview_stats_text(estimated_seconds, stroke_count, color_changes):
    return (
        f"估時: {format_estimated_seconds(estimated_seconds)} | "
        f"筆畫: {int(stroke_count)} | 換色: {int(color_changes)}"
    )


def estimate_brush_select_seconds(brush_key, brush_positions=None):
    brush_key = clamp_brush_key(brush_key)
    has_visible_button = bool(brush_positions and len(brush_positions) >= max(1, brush_key))

    if brush_key == 0:
        return HOTKEY_FOCUS_DELAY + 0.30
    if has_visible_button:
        return 0.20
    return HOTKEY_FOCUS_DELAY + 0.30


def estimate_custom_rgb_select_seconds(rgb, panel_delay):
    seconds = CUSTOM_RGB_SWATCH_CLICK_DELAY + max(0.0, float(panel_delay))

    for value in rgb:
        text = str(int(np.clip(value, 0, 255)))
        seconds += CUSTOM_RGB_INPUT_CLICK_DELAY
        seconds += len(text) * CUSTOM_RGB_TYPE_INTERVAL
        seconds += CUSTOM_RGB_INPUT_WRITE_DELAY

    return seconds


def count_color_transitions(color_indices):
    current_color = None
    changes = 0

    for color_idx in color_indices:
        color_idx = int(color_idx)
        if color_idx != current_color:
            changes += 1
            current_color = color_idx

    return changes


def estimate_line_preview_seconds(strokes, line_move_ms, line_gap_ms, color_changes=0, brush_seconds=0.0):
    move_seconds = max(0.0, float(line_move_ms) / 1000.0)
    gap_seconds = max(0.0, float(line_gap_ms) / 1000.0)
    seconds = float(brush_seconds) + max(0, int(color_changes)) * 0.10

    for stroke in strokes:
        if len(stroke) < 2:
            seconds += 0.005
            continue
        seconds += 0.06
        seconds += max(0, len(stroke) - 1) * move_seconds
        seconds += gap_seconds

    return seconds


def estimate_sbr_preview_seconds(strokes, cps, line_move_ms, color_changes, brush_seconds=0.0):
    move_seconds = max(0.001, float(line_move_ms) / 1000.0)
    batch_wait = max(1.0 / max(1.0, float(cps)), 0.001)
    stroke_count = len(strokes)
    seconds = float(brush_seconds)
    seconds += max(0, int(color_changes)) * PALETTE_SELECT_DELAY
    seconds += stroke_count * (0.06 + move_seconds)
    seconds += (stroke_count // 10) * batch_wait
    return seconds


def estimate_color_run_seconds(run, cell_step_px, brush_px, requested_drag):
    y, start, end, reverse = normalize_run(run)
    del y, reverse
    distance = max(0, end - start - 1) * max(1, int(cell_step_px))

    if distance <= 1:
        return 0.003

    brush_px = max(1, int(brush_px))
    segment_px = 120 if brush_px <= 3 else 170 if brush_px <= 6 else 230
    overlap_px = max(2, min(10, int(max(brush_px, 2) * 0.55)))

    if distance <= segment_px:
        return color_drag_duration(distance, brush_px, requested_drag)

    seconds = 0.0
    current = 0

    while True:
        remaining = distance - current
        target = distance if remaining <= segment_px else current + segment_px
        seconds += color_drag_duration(target - current, brush_px, requested_drag)

        if target >= distance:
            break

        current = max(0, target - overlap_px)

    return seconds


def estimate_spiral_path_seconds(path, cell_step_px):
    if len(path) < 2:
        return 0.003 if path else 0.0

    cell_step_px = max(1, int(cell_step_px))
    seconds = 0.010
    last_x, last_y = path[0]

    for x, y in path[1:]:
        dx = (x - last_x) * cell_step_px
        dy = (y - last_y) * cell_step_px
        distance = (dx * dx + dy * dy) ** 0.5
        seconds += max(0.0008, min(0.0045, distance / 18000.0))
        last_x, last_y = x, y

    return seconds


def color_plan_preview_metrics(
    mode,
    mapping_colors,
    color_order,
    runs_by_color,
    spiral_paths_by_color,
    white_indices,
    eye_order,
    eye_runs_by_color,
    cps,
    line_move_ms,
    brush_px,
    cell_step_px,
    rgb_panel_delay_ms,
    eye_brush_px=1,
    eye_cell_step_px=1,
    brush_seconds=0.0,
):
    requested_drag = max(0.001, float(line_move_ms) / 1000.0)
    batch_wait = max(1.0 / max(1.0, float(cps)), 0.001)
    rgb_panel_delay = max(0.0, min(0.5, float(rgb_panel_delay_ms) / 1000.0))
    total_ops = 0
    color_changes = 0
    seconds = float(brush_seconds)

    def selection_seconds(color_idx, detail=False):
        select_delay = PALETTE_DETAIL_SELECT_DELAY if detail else PALETTE_SELECT_DELAY
        if mode == MODE_CUSTOM_RGB:
            return estimate_custom_rgb_select_seconds(mapping_colors[color_idx], rgb_panel_delay) + select_delay
        return select_delay

    for color_idx in color_order:
        if color_idx in white_indices:
            continue

        runs = runs_by_color[color_idx] if color_idx < len(runs_by_color) else []
        spiral_paths = spiral_paths_by_color[color_idx] if color_idx < len(spiral_paths_by_color) else []
        if not runs and not spiral_paths:
            continue

        color_changes += 1
        seconds += selection_seconds(color_idx, detail=False)

        for path in spiral_paths:
            seconds += estimate_spiral_path_seconds(path, cell_step_px)
            total_ops += 1
            if total_ops % 8 == 0:
                seconds += batch_wait

        for run in runs:
            seconds += estimate_color_run_seconds(run, cell_step_px, brush_px, requested_drag)
            total_ops += 1
            if total_ops % 10 == 0:
                seconds += batch_wait

    if eye_runs_by_color is not None and eye_order:
        seconds += estimate_brush_select_seconds(0 if brush_px <= 1 else 1)

        for color_idx in eye_order:
            eye_runs = eye_runs_by_color[color_idx] if color_idx < len(eye_runs_by_color) else []
            if not eye_runs:
                continue

            color_changes += 1
            seconds += selection_seconds(color_idx, detail=True)

            for run in eye_runs:
                seconds += estimate_color_run_seconds(
                    run,
                    eye_cell_step_px,
                    eye_brush_px,
                    max(0.001, min(0.008, requested_drag)),
                )
                total_ops += 1
                if total_ops % 25 == 0:
                    seconds += batch_wait

    return total_ops, color_changes, seconds


def planned_color_runs_to_canvas_preview(
    canvas_w,
    canvas_h,
    mapping_colors,
    color_order,
    runs_by_color,
    spiral_paths_by_color,
    offset_x,
    offset_y,
    brush_px,
    cell_step_px,
    white_indices=None,
    eye_order=None,
    eye_runs_by_color=None,
    eye_offset_x=0,
    eye_offset_y=0,
    eye_brush_px=1,
    eye_cell_step_px=1,
):
    canvas_w = max(1, int(canvas_w))
    canvas_h = max(1, int(canvas_h))
    brush_px = max(1, int(brush_px))
    cell_step_px = max(1, int(cell_step_px))
    white_indices = set(white_indices or [])
    preview = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    def color_tuple(color_idx):
        return tuple(int(v) for v in mapping_colors[color_idx])

    def paint_run(run, color, run_offset_x, run_offset_y, run_brush_px, run_cell_step_px):
        y, start, end, reverse = normalize_run(run)
        if end <= start:
            return

        draw_start = end - 1 if reverse else start
        draw_end = start if reverse else end - 1
        sx = int(round(run_offset_x + draw_start * run_cell_step_px + run_brush_px / 2))
        ex = int(round(run_offset_x + draw_end * run_cell_step_px + run_brush_px / 2))
        sy = int(round(run_offset_y + y * run_cell_step_px + run_brush_px / 2))

        if abs(ex - sx) <= 1:
            cv2.circle(preview, (sx, sy), max(1, run_brush_px // 2), color, thickness=-1, lineType=cv2.LINE_AA)
        else:
            cv2.line(preview, (sx, sy), (ex, sy), color, thickness=run_brush_px, lineType=cv2.LINE_AA)

    def paint_spiral(path, color):
        if not path:
            return

        points = np.asarray(
            [
                (
                    int(round(offset_x + px * cell_step_px + brush_px / 2)),
                    int(round(offset_y + py * cell_step_px + brush_px / 2)),
                )
                for px, py in path
            ],
            dtype=np.int32,
        )

        if len(points) == 1:
            cv2.circle(preview, tuple(points[0]), max(1, brush_px // 2), color, thickness=-1, lineType=cv2.LINE_AA)
        else:
            cv2.polylines(preview, [points.reshape((-1, 1, 2))], False, color, thickness=brush_px, lineType=cv2.LINE_AA)

    for color_idx in color_order:
        if color_idx in white_indices:
            continue

        color = color_tuple(color_idx)

        if color_idx < len(spiral_paths_by_color):
            for path in spiral_paths_by_color[color_idx]:
                paint_spiral(path, color)

        if color_idx < len(runs_by_color):
            for run in runs_by_color[color_idx]:
                paint_run(run, color, offset_x, offset_y, brush_px, cell_step_px)

    if eye_runs_by_color is not None and eye_order:
        eye_brush_px = max(1, int(eye_brush_px))
        eye_cell_step_px = max(1, int(eye_cell_step_px))

        for color_idx in eye_order:
            if color_idx >= len(eye_runs_by_color):
                continue

            color = color_tuple(color_idx)
            for run in eye_runs_by_color[color_idx]:
                paint_run(run, color, eye_offset_x, eye_offset_y, eye_brush_px, eye_cell_step_px)

    return Image.fromarray(preview, "RGB")


def compose_canvas_preview(content, canvas_w, canvas_h):
    canvas_w = max(1, int(canvas_w))
    canvas_h = max(1, int(canvas_h))
    preview = Image.new("RGB", (canvas_w, canvas_h), "white")
    x = (canvas_w - content.size[0]) // 2
    y = (canvas_h - content.size[1]) // 2
    preview.paste(content, (x, y))
    return preview


def compose_canvas_preview_at(content, canvas_w, canvas_h, offset_x, offset_y):
    """Paste preview content at a canvas-relative position instead of centering it."""
    canvas_w = max(1, int(canvas_w))
    canvas_h = max(1, int(canvas_h))
    preview = Image.new("RGB", (canvas_w, canvas_h), "white")
    preview.paste(content, (int(round(offset_x)), int(round(offset_y))))
    return preview


def normalize_image_placement(rect):
    """Normalize a screen-space image placement rectangle to (left, top, right, bottom)."""
    if not rect:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in list(rect)[:4]]
    except Exception:
        return None
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    if right - left < 6 or bottom - top < 6:
        return None
    return (left, top, right, bottom)


def default_image_placement(canvas, image, scale_percent=85):
    """Create a centered screen-space placement rectangle inside the detected canvas."""
    if not canvas or image is None:
        return None
    try:
        x1, y1, x2, y2 = [int(v) for v in canvas]
        left, top = min(x1, x2), min(y1, y2)
        canvas_w, canvas_h = abs(x2 - x1), abs(y2 - y1)
        if canvas_w <= 0 or canvas_h <= 0:
            return None
        img_w, img_h = image.size
        if img_w <= 0 or img_h <= 0:
            return None
        scale_percent = int(np.clip(int(scale_percent), 10, 100))
        max_w = max(8, int(canvas_w * scale_percent / 100))
        max_h = max(8, int(canvas_h * scale_percent / 100))
        scale = min(max_w / img_w, max_h / img_h)
        draw_w = max(8, int(round(img_w * scale)))
        draw_h = max(8, int(round(img_h * scale)))
        px1 = int(round(left + (canvas_w - draw_w) / 2))
        py1 = int(round(top + (canvas_h - draw_h) / 2))
        return (px1, py1, px1 + draw_w, py1 + draw_h)
    except Exception:
        return None


def clamp_rect_to_canvas(rect, canvas):
    rect = normalize_image_placement(rect)
    if rect is None or not canvas:
        return rect
    try:
        x1, y1, x2, y2 = [int(v) for v in canvas]
        c_left, c_top = min(x1, x2), min(y1, y2)
        c_right, c_bottom = max(x1, x2), max(y1, y2)
        left, top, right, bottom = rect
        w, h = right - left, bottom - top
        canvas_w, canvas_h = max(1, c_right - c_left), max(1, c_bottom - c_top)
        if w > canvas_w:
            left, right = c_left, c_right
        else:
            if left < c_left:
                right += c_left - left
                left = c_left
            if right > c_right:
                left -= right - c_right
                right = c_right
        if h > canvas_h:
            top, bottom = c_top, c_bottom
        else:
            if top < c_top:
                bottom += c_top - top
                top = c_top
            if bottom > c_bottom:
                top -= bottom - c_bottom
                bottom = c_bottom
        return normalize_image_placement((left, top, right, bottom))
    except Exception:
        return rect


def target_area_from_placement(canvas, placement_rect, line_scale):
    """Return canvas-relative target box: x, y, w, h, using_manual_placement."""
    x1, y1, x2, y2 = canvas
    canvas_left, canvas_top = min(x1, x2), min(y1, y2)
    canvas_right, canvas_bottom = max(x1, x2), max(y1, y2)
    canvas_w, canvas_h = max(1, canvas_right - canvas_left), max(1, canvas_bottom - canvas_top)

    rect = normalize_image_placement(placement_rect)
    if rect:
        left = max(canvas_left, rect[0])
        top = max(canvas_top, rect[1])
        right = min(canvas_right, rect[2])
        bottom = min(canvas_bottom, rect[3])
        if right - left >= 6 and bottom - top >= 6:
            return (left - canvas_left, top - canvas_top, right - left, bottom - top, True)

    line_scale = min(100, max(10, int(line_scale)))
    target_w = max(1, int(canvas_w * line_scale / 100))
    target_h = max(1, int(canvas_h * line_scale / 100))
    return ((canvas_w - target_w) / 2, (canvas_h - target_h) / 2, target_w, target_h, False)


def fit_preview_image(img, max_size=PREVIEW_MAX_SIZE):
    w, h = img.size
    scale = min(max_size / max(w, 1), max_size / max(h, 1), 1.0)

    if scale >= 1.0:
        return img.copy()

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def darkest_palette_index(palette_colors):
    if not palette_colors:
        return 0
    colors = np.asarray(palette_colors, dtype=np.int32)
    scores = np.sum(colors, axis=1) + np.max(colors, axis=1) * 2
    return int(np.argmin(scores))


# ============================================================================
# Qt Widgets
# ============================================================================

class ToggleSwitch(QAbstractButton):
    """Small modern toggle with a real knob, instead of the default checkbox box."""

    def __init__(self, checked=False, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(bool(checked))
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(46, 26)

    def sizeHint(self):
        return QSize(46, 26)

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        rect = QRectF(1.0, 2.0, 44.0, 22.0)
        checked = self.isChecked()
        enabled = self.isEnabled()

        if not enabled:
            # Disabled by the current mode: keep the on/off position visible,
            # but remove the bright active blue so it does not look clickable.
            track = QColor("#172033") if checked else QColor("#0F172A")
            border = QColor("#2A3548")
            knob = QColor("#64748B") if checked else QColor("#475569")
            knob_x = 23.0 if checked else 3.0
        elif checked:
            track = QColor("#38BDF8")
            border = QColor("#60A5FA")
            knob = QColor("#EFF6FF")
            knob_x = 23.0
        else:
            track = QColor("#111827")
            border = QColor("#334155")
            knob = QColor("#64748B")
            knob_x = 3.0

        if enabled and self.underMouse():
            border = QColor("#7DD3FC")
            if not checked:
                track = QColor("#162033")

        painter.setPen(QPen(border, 1.2))
        painter.setBrush(track)
        painter.drawRoundedRect(rect, 11.0, 11.0)

        knob_rect = QRectF(knob_x, 4.0, 18.0, 18.0)
        painter.setPen(Qt.NoPen)
        painter.setBrush(knob)
        painter.drawEllipse(knob_rect)
        painter.end()


class OptionToggle(QFrame):
    """Clickable option row/card that exposes isChecked()/setChecked() like QCheckBox."""

    def __init__(self, title, subtitle="", checked=False, parent=None):
        super().__init__(parent)
        self.setObjectName("optionToggleCard")
        self.setCursor(Qt.PointingHandCursor)
        self.setProperty("checked", bool(checked))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 11, 12, 11)
        layout.setSpacing(12)

        label_box = QVBoxLayout()
        label_box.setContentsMargins(0, 0, 0, 0)
        label_box.setSpacing(2)

        title_label = QLabel(title)
        title_label.setObjectName("optionTitle")
        label_box.addWidget(title_label)

        if subtitle:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setObjectName("optionSubtitle")
            subtitle_label.setWordWrap(True)
            label_box.addWidget(subtitle_label)

        layout.addLayout(label_box, 1)
        self.switch = ToggleSwitch(checked=checked)
        self.switch.toggled.connect(self._sync_property)
        layout.addWidget(self.switch, 0, Qt.AlignRight | Qt.AlignVCenter)

    def _sync_property(self, checked):
        self.setProperty("checked", bool(checked))
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def setEnabled(self, enabled):
        enabled = bool(enabled)
        super().setEnabled(enabled)
        self.switch.setEnabled(enabled)
        self.setCursor(Qt.PointingHandCursor if enabled else Qt.ArrowCursor)
        self.switch.setCursor(Qt.PointingHandCursor if enabled else Qt.ArrowCursor)
        self.style().unpolish(self)
        self.style().polish(self)
        self.switch.update()
        self.update()

    def mousePressEvent(self, event):
        if not self.isEnabled():
            event.ignore()
            return
        if event.button() == Qt.LeftButton:
            self.switch.toggle()
            event.accept()
            return
        super().mousePressEvent(event)

    def isChecked(self):
        return self.switch.isChecked()

    def setChecked(self, checked):
        self.switch.setChecked(bool(checked))
        self._sync_property(bool(checked))


def pil_to_qpixmap(img):
    """Convert PIL image to QPixmap safely."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = fit_preview_image(img)
    w, h = img.size
    data = img.tobytes("raw", "RGB")
    qimg = QImage(data, w, h, w * 3, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


def pil_rgba_to_qpixmap(img, max_side=1200):
    """Convert a PIL image with alpha to a reasonably sized QPixmap for overlay painting."""
    if img is None:
        return QPixmap()
    rgba = img.convert("RGBA")
    w, h = rgba.size
    max_side = max(64, int(max_side))
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        rgba = rgba.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
        w, h = rgba.size
    data = rgba.tobytes("raw", "RGBA")
    qimg = QImage(data, w, h, w * 4, QImage.Format_RGBA8888).copy()
    return QPixmap.fromImage(qimg)


class AnimatedActionButton(QPushButton):
    """Three-state action button with a soft animated border for the computing stage."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._phase = "idle"
        self._pulse = 0
        self._progress = None
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(44)
        self.setFont(QFont("Segoe UI", 11, QFont.Bold))

    def setPhase(self, phase):
        self._phase = str(phase or "idle")
        self.update()

    def setPulse(self, pulse):
        self._pulse = int(pulse) % 360
        self.update()

    def setProgress(self, progress):
        if progress is None or int(progress) < 0:
            self._progress = None
        else:
            self._progress = max(0, min(100, int(progress)))
        self.update()

    def paintEvent(self, event):
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        rect = QRectF(1.5, 1.5, self.width() - 3.0, self.height() - 3.0)
        radius = 14.0
        hover = self.underMouse()

        if self._phase == "computing":
            # Clean computing state: keep the soft orbiting border, but remove
            # the old moving dots that appeared inside the button.
            text_color = QColor("#F8FAFC")

            fill = QLinearGradient(rect.topLeft(), rect.bottomRight())
            fill.setColorAt(0.0, QColor("#0B1222"))
            fill.setColorAt(1.0, QColor("#111A2E"))
            painter.setPen(QPen(QColor("#1E2B44"), 1.0))
            painter.setBrush(fill)
            painter.drawRoundedRect(rect, radius, radius)

            # Subtle inner color wash, clipped to the button body so it does not
            # create random dots or blobs.
            wash = QLinearGradient(rect.topLeft(), rect.topRight())
            wash.setColorAt(0.00, QColor(56, 189, 248, 22))
            wash.setColorAt(0.45, QColor(167, 139, 250, 16))
            wash.setColorAt(1.00, QColor(45, 212, 191, 20))
            painter.setPen(Qt.NoPen)
            painter.setBrush(wash)
            painter.drawRoundedRect(rect.adjusted(4, 4, -4, -4), radius - 4, radius - 4)

            # Switch-like rotating pastel border.  The gradient itself rotates;
            # no separate dot is drawn, so the border stays clean.
            orbit_rect = rect.adjusted(2.0, 2.0, -2.0, -2.0)
            orbit = QConicalGradient(orbit_rect.center(), -self._pulse)
            orbit.setColorAt(0.00, QColor("#7DD3FC"))
            orbit.setColorAt(0.18, QColor("#A78BFA"))
            orbit.setColorAt(0.36, QColor("#F0ABFC"))
            orbit.setColorAt(0.56, QColor("#67E8F9"))
            orbit.setColorAt(0.76, QColor("#86EFAC"))
            orbit.setColorAt(1.00, QColor("#7DD3FC"))
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QBrush(orbit), 2.4))
            painter.drawRoundedRect(orbit_rect, radius - 1, radius - 1)

            # A faint outer glow keeps the orbit visible on dark backgrounds.
            painter.setPen(QPen(QColor(125, 211, 252, 60), 1.0))
            painter.drawRoundedRect(rect.adjusted(0.8, 0.8, -0.8, -0.8), radius, radius)

        elif self._phase == "drawing":
            progress = self._progress
            base = QLinearGradient(rect.topLeft(), rect.topRight())
            base.setColorAt(0.0, QColor("#0B1222"))
            base.setColorAt(1.0, QColor("#111827"))
            text_color = QColor("#FFFFFF")
            painter.setPen(QPen(QColor("#1F2937"), 1.0))
            painter.setBrush(base)
            painter.drawRoundedRect(rect, radius, radius)

            if progress is None:
                fill = QLinearGradient(rect.topLeft(), rect.topRight())
                fill.setColorAt(0.0, QColor("#22C55E"))
                fill.setColorAt(0.55, QColor("#16A34A"))
                fill.setColorAt(1.0, QColor("#0D9488"))
                painter.setPen(QPen(QColor("#86EFAC"), 1.4))
                painter.setBrush(fill)
                painter.drawRoundedRect(rect, radius, radius)
            else:
                progress = max(0, min(100, int(progress)))
                progress_rect = QRectF(rect)
                progress_rect.setWidth(max(3.0, rect.width() * progress / 100.0) if progress > 0 else 0.0)
                if progress_rect.width() > 0:
                    fill = QLinearGradient(rect.topLeft(), rect.topRight())
                    fill.setColorAt(0.0, QColor("#22C55E"))
                    fill.setColorAt(0.55, QColor("#16A34A"))
                    fill.setColorAt(1.0, QColor("#0D9488"))
                    painter.save()
                    painter.setClipRect(progress_rect)
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(fill)
                    painter.drawRoundedRect(rect, radius, radius)
                    painter.restore()

                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(QColor("#86EFAC"), 1.4))
                painter.drawRoundedRect(rect, radius, radius)

        else:
            fill = QLinearGradient(rect.topLeft(), rect.topRight())
            if hover and self.isEnabled():
                fill.setColorAt(0.0, QColor("#34D399"))
                fill.setColorAt(0.55, QColor("#22C55E"))
                fill.setColorAt(1.0, QColor("#14B8A6"))
            else:
                fill.setColorAt(0.0, QColor("#22C55E"))
                fill.setColorAt(0.55, QColor("#16A34A"))
                fill.setColorAt(1.0, QColor("#0D9488"))
            text_color = QColor("#FFFFFF") if self.isEnabled() else QColor("#94A3B8")
            painter.setPen(Qt.NoPen)
            painter.setBrush(fill if self.isEnabled() else QColor("#1F2937"))
            painter.drawRoundedRect(rect, radius, radius)

        painter.setPen(text_color)
        painter.setFont(QFont("Segoe UI", 11, QFont.Bold))
        painter.drawText(rect, Qt.AlignCenter, self.text())
        painter.end()


class BorderlessProfileCombo(QComboBox):
    """QComboBox with the native Windows popup/focus frame stripped out.

    Qt creates the drop-down inside a private QFrame container.  Styling only
    the QListView is not enough on Windows because that private container can
    still draw a gray outline when the popup opens.  Polish it right after
    showPopup() creates the container.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrame(False)
        self.setFocusPolicy(Qt.NoFocus)
        self.setAttribute(Qt.WA_MacShowFocusRect, False)

    def showPopup(self):
        super().showPopup()
        QTimer.singleShot(0, self._remove_native_popup_frame)

    def _remove_native_popup_frame(self):
        view = self.view()
        if view is None:
            return

        view.setObjectName("profileComboPopup")
        view.setFrameShape(QFrame.NoFrame)
        view.setLineWidth(0)
        view.setMidLineWidth(0)
        view.setContentsMargins(0, 0, 0, 0)
        view.setViewportMargins(0, 0, 0, 0)
        view.setFocusPolicy(Qt.NoFocus)
        view.setAttribute(Qt.WA_MacShowFocusRect, False)
        view.viewport().setContentsMargins(0, 0, 0, 0)
        view.viewport().setAutoFillBackground(False)

        popup = view.window()
        if popup is not None:
            popup.setObjectName("profileComboPopupContainer")
            popup.setContentsMargins(0, 0, 0, 0)
            popup.setAttribute(Qt.WA_MacShowFocusRect, False)
            popup.setAttribute(Qt.WA_StyledBackground, True)
            popup.setWindowFlag(Qt.FramelessWindowHint, True)
            if hasattr(Qt, "NoDropShadowWindowHint"):
                popup.setWindowFlag(Qt.NoDropShadowWindowHint, True)
            if isinstance(popup, QFrame):
                popup.setFrameShape(QFrame.NoFrame)
                popup.setLineWidth(0)
                popup.setMidLineWidth(0)
            if popup.layout() is not None:
                popup.layout().setContentsMargins(0, 0, 0, 0)
                popup.layout().setSpacing(0)
            popup.setStyleSheet("""
                QFrame#profileComboPopupContainer {
                    background: #071020;
                    border: 0px;
                    padding: 0px;
                    margin: 0px;
                    outline: 0px;
                }
                QListView#profileComboPopup {
                    background: #071020;
                    border: 0px;
                    padding: 4px;
                    margin: 0px;
                    outline: 0px;
                }
                QListView#profileComboPopup::viewport {
                    background: #071020;
                    border: 0px;
                    outline: 0px;
                }
                QListView#profileComboPopup::item {
                    min-height: 22px;
                    padding: 4px 8px;
                    margin: 0px 1px;
                    border: 0px;
                    border-radius: 7px;
                    color: #CBD5E1;
                    background: transparent;
                }
                QListView#profileComboPopup::item:hover {
                    color: #F8FAFC;
                    background: rgba(56, 189, 248, 0.12);
                    border: 0px;
                }
                QListView#profileComboPopup::item:selected {
                    color: #FFFFFF;
                    background: rgba(37, 99, 235, 0.35);
                    border: 0px;
                    outline: 0px;
                }
            """)


# ============================================================================
# Calibration Overlays
# ============================================================================

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


# ============================================================================
# Main Application
# ============================================================================

try:
    from pynput import keyboard as pynput_keyboard
except Exception:
    pynput_keyboard = None


class UiSignals(QObject):
    log_message = Signal(str)
    detect_status = Signal(str, str)
    detecting_changed = Signal(bool)
    drawing_changed = Signal(bool)
    draw_phase_changed = Signal(str)
    draw_progress_changed = Signal(int)
    previewing_changed = Signal(bool)
    preview_ready = Signal(object, str)
    overlay_ready = Signal(object, object, object, object)
    stop_requested = Signal()


class GarticQtDrawer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gartic OpenCV Drawer - Qt")
        self.resize(1120, 700)

        self.image = None
        self.image_path = None
        self.canvas = None
        self.palette = []
        self.brush_positions = []
        self.custom_rgb_controls = None
        self.is_detecting = False
        self.is_drawing = False
        self.is_previewing = False
        self.stop_event = threading.Event()
        self.keyboard_listener = None
        self.preview_windows = []
        self.detection_overlay = DetectionOverlay()
        self.detection_overlay.detection_changed.connect(self.apply_overlay_adjustment)
        self.detection_overlay.drag_finished.connect(self.finish_overlay_adjustment)
        self.detection_overlay.calibration_saved.connect(self.save_overlay_calibration)
        self.detection_overlay.calibration_cancelled.connect(self.cancel_overlay_calibration)
        self.detection_overlay.profile_save_requested.connect(self.save_overlay_profile)
        self.overlay_calibration_backup = None
        self.image_placement = None
        self.image_placement_backup = None
        self.image_placement_overlay = ImagePlacementOverlay()
        self.image_placement_overlay.placement_changed.connect(self.apply_image_placement)
        self.image_placement_overlay.placement_saved.connect(self.save_image_placement)
        self.image_placement_overlay.placement_cancelled.connect(self.cancel_image_placement)
        self.profiles = {}
        self.mode_value = MODE_SMART_LINE
        self.draw_start_time = None
        self.last_draw_elapsed = 0.0
        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.setInterval(200)
        self.elapsed_timer.timeout.connect(self.update_elapsed_label)
        self.draw_phase = "idle"
        self.action_pulse = 0
        self.action_anim_timer = QTimer(self)
        self.action_anim_timer.setInterval(38)
        self.action_anim_timer.timeout.connect(self.update_action_animation)

        self.signals = UiSignals()
        self.signals.log_message.connect(self._append_log)
        self.signals.detect_status.connect(self._set_detect_status)
        self.signals.detecting_changed.connect(self.set_detecting)
        self.signals.drawing_changed.connect(self.set_drawing)
        self.signals.draw_phase_changed.connect(self.set_draw_phase)
        self.signals.draw_progress_changed.connect(self.set_draw_progress)
        self.signals.previewing_changed.connect(self.set_previewing)
        self.signals.preview_ready.connect(self.show_preview)
        self.signals.overlay_ready.connect(self.update_detection_overlay)
        self.signals.stop_requested.connect(self.request_stop)

        self.build_ui()
        self.apply_modern_theme()
        self.load_profiles()
        self.start_global_hotkey()
        self.log("=== Qt OpenCV 自適應偵測 + Overlay + Custom RGB 校正版就緒 ===")
        self.log("緊急停止：按 STOP、Esc，或迅速將滑鼠移到螢幕角落")

    def configure_number_inputs(self):
        """Hide spinbox arrow buttons; keep typing and mouse-wheel editing."""
        for spin in (
            self.cps_spin,
            self.brush_spin,
            self.custom_colors_spin,
            self.sbr_strokes_spin,
            self.detail_spin,
            self.line_move_spin,
            self.line_gap_spin,
            self.line_scale_spin,
            self.stroke_step_spin,
            self.rgb_panel_delay_spin,
        ):
            spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
            spin.setAccelerated(True)
            spin.setCursor(Qt.IBeamCursor)

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(18, 14, 18, 14)
        main_layout.setSpacing(12)

        left_panel = QWidget()
        root = QVBoxLayout(left_panel)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)
        main_layout.addWidget(left_panel, 2)

        def make_card(object_name, spacing=10):
            card = QFrame()
            card.setObjectName(object_name)
            effect = QGraphicsDropShadowEffect(card)
            effect.setBlurRadius(18)
            effect.setXOffset(0)
            effect.setYOffset(6)
            effect.setColor(QColor(0, 0, 0, 80))
            card.setGraphicsEffect(effect)
            layout = QVBoxLayout(card)
            layout.setContentsMargins(13, 9, 13, 9)
            layout.setSpacing(max(5, spacing - 3))
            return card, layout

        # Header / file card
        header_card, header_layout = make_card("headerCard", spacing=9)
        title = QLabel("Gartic OpenCV Drawer")
        title.setObjectName("titleLabel")
        title.setFont(QFont("Segoe UI", 22, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(title)

        subtitle = QLabel("Smart Line • Palette • Custom RGB • SBR")
        subtitle.setObjectName("subtitleLabel")
        subtitle.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(subtitle)

        self.image_label = QLabel("尚未載入圖片")
        self.image_label.setObjectName("pathLabel")
        self.image_label.setWordWrap(True)
        header_layout.addWidget(self.image_label)

        file_buttons = QHBoxLayout()
        file_buttons.setSpacing(8)
        self.load_btn = QPushButton("Load Image 載入圖片")
        self.load_btn.setObjectName("secondaryButton")
        self.load_btn.clicked.connect(self.load_image)
        file_buttons.addWidget(self.load_btn)

        self.detect_btn = QPushButton(DETECT_BUTTON_TEXT)
        self.detect_btn.setObjectName("secondaryButton")
        self.detect_btn.clicked.connect(self.auto_detect_thread)
        file_buttons.addWidget(self.detect_btn)

        self.overlay_btn = QPushButton("Overlay 偵測/拖動校正")
        self.overlay_btn.setObjectName("secondaryButton")
        self.overlay_btn.setCheckable(True)
        self.overlay_btn.clicked.connect(self.toggle_overlay_calibration)
        file_buttons.addWidget(self.overlay_btn)

        self.place_btn = QPushButton("圖片定位")
        self.place_btn.setObjectName("secondaryButton")
        self.place_btn.setCheckable(True)
        self.place_btn.clicked.connect(self.toggle_image_placement_overlay)
        file_buttons.addWidget(self.place_btn)

        header_layout.addLayout(file_buttons)
        root.addWidget(header_card)

        # Quick controls card
        quick_card, quick_layout = make_card("quickCard", spacing=10)
        quick_title = QLabel("Quick Control 快速控制")
        quick_title.setObjectName("cardTitle")
        quick_layout.addWidget(quick_title)

        quick_row = QHBoxLayout()
        quick_row.setSpacing(10)
        quick_row.addWidget(QLabel("CPS"))
        self.cps_spin = QDoubleSpinBox()
        self.cps_spin.setRange(1, 1000)
        self.cps_spin.setSingleStep(10)
        self.cps_spin.setDecimals(0)
        self.cps_spin.setValue(200)
        quick_row.addWidget(self.cps_spin)

        quick_row.addWidget(QLabel("Brush Key (`=0)"))
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(0, 5)
        self.brush_spin.setValue(1)
        quick_row.addWidget(self.brush_spin)

        self.detect_label = QLabel("尚未偵測")
        self.detect_label.setObjectName("statusPill")
        quick_row.addWidget(self.detect_label, 1)
        quick_layout.addLayout(quick_row)

        profile_row = QHBoxLayout()
        profile_row.setSpacing(8)
        profile_row.addWidget(QLabel("畫布設定檔"))
        self.profile_combo = BorderlessProfileCombo()
        self.profile_combo.setObjectName("profileCombo")
        # Remove native combo frame/focus ring; the stylesheet draws the only visible border.
        self.profile_combo.setFrame(False)
        self.profile_combo.setAttribute(Qt.WA_MacShowFocusRect, False)
        self.profile_combo.setMinimumWidth(190)
        self.profile_combo.setMaxVisibleItems(6)
        profile_view = QListView()
        profile_view.setObjectName("profileComboPopup")
        profile_view.setUniformItemSizes(True)
        profile_view.setSpacing(1)
        profile_view.setMouseTracking(True)
        # Remove the native popup frame that appears as an unwanted gray outline.
        profile_view.setFrameShape(QFrame.NoFrame)
        profile_view.setLineWidth(0)
        profile_view.setMidLineWidth(0)
        profile_view.setWindowFlag(Qt.FramelessWindowHint, True)
        profile_view.setFocusPolicy(Qt.NoFocus)
        profile_view.viewport().setAutoFillBackground(False)
        self.profile_combo.setView(profile_view)
        profile_row.addWidget(self.profile_combo, 1)

        self.load_profile_btn = QPushButton("載入")
        self.load_profile_btn.setObjectName("secondaryButton")
        self.load_profile_btn.clicked.connect(self.load_selected_profile)
        profile_row.addWidget(self.load_profile_btn)

        self.save_profile_btn = QPushButton("保存目前")
        self.save_profile_btn.setObjectName("secondaryButton")
        self.save_profile_btn.clicked.connect(self.save_current_profile_dialog)
        profile_row.addWidget(self.save_profile_btn)
        quick_layout.addLayout(profile_row)
        root.addWidget(quick_card)

        # Mode + advanced settings card
        mode_card, mode_outer = make_card("settingsCard", spacing=12)
        mode_title = QLabel("Mode & Parameters 模式與參數")
        mode_title.setObjectName("cardTitle")
        mode_outer.addWidget(mode_title)

        mode_layout = QGridLayout()
        mode_layout.setHorizontalSpacing(10)
        mode_layout.setVerticalSpacing(7)
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)

        def add_mode(text, value, row, col, colspan=1):
            btn = QRadioButton(text)
            btn.setObjectName("modeChip")
            btn.mode_value = value
            btn.toggled.connect(lambda checked, b=btn: self._set_mode_from_button(b, checked))
            self.mode_group.addButton(btn)
            mode_layout.addWidget(btn, row, col, 1, colspan)
            return btn

        self.smart_line_radio = add_mode("Smart Line Art 整合線稿", MODE_SMART_LINE, 0, 0, 1)
        add_mode("Palette Color 全彩色盤", MODE_PALETTE, 0, 1)
        add_mode("Custom RGB 自訂色", MODE_CUSTOM_RGB, 0, 2)
        add_mode("SBR 筆觸渲染", MODE_SBR, 0, 3)
        self.smart_line_radio.setChecked(True)

        self.mode_param_widgets = {}

        def add_param(label_text, widget, row, label_col, key):
            label = QLabel(label_text)
            label.setObjectName("paramLabel")
            mode_layout.addWidget(label, row, label_col)
            mode_layout.addWidget(widget, row, label_col + 1)
            self.mode_param_widgets[key] = (label, widget)
            return widget

        self.detail_spin = add_param("Line Detail", QSpinBox(), 1, 0, "detail")
        self.detail_spin.setRange(1, 5)
        self.detail_spin.setValue(4)

        self.custom_colors_spin = add_param("Custom Colors", QSpinBox(), 1, 2, "custom_colors")
        self.custom_colors_spin.setRange(8, MAX_CUSTOM_COLORS)
        self.custom_colors_spin.setSingleStep(8)
        self.custom_colors_spin.setValue(DEFAULT_CUSTOM_COLORS)

        self.line_move_spin = add_param("Line Move ms", QSpinBox(), 2, 0, "line_move")
        self.line_move_spin.setRange(1, 80)
        self.line_move_spin.setValue(DEFAULT_LINE_MOVE_MS)

        self.sbr_strokes_spin = add_param("SBR Strokes", QSpinBox(), 2, 2, "sbr_strokes")
        self.sbr_strokes_spin.setRange(50, 1500)
        self.sbr_strokes_spin.setSingleStep(50)
        self.sbr_strokes_spin.setValue(DEFAULT_SBR_STROKES)

        self.line_gap_spin = add_param("Line Gap ms", QSpinBox(), 3, 0, "line_gap")
        self.line_gap_spin.setRange(0, 80)
        self.line_gap_spin.setValue(DEFAULT_LINE_GAP_MS)

        self.line_scale_spin = add_param("Image Scale %", QSpinBox(), 3, 2, "line_scale")
        self.line_scale_spin.setRange(40, 100)
        self.line_scale_spin.setSingleStep(5)
        self.line_scale_spin.setValue(DEFAULT_LINE_SCALE)

        self.stroke_step_spin = add_param("Stroke Step", QSpinBox(), 4, 0, "stroke_step")
        self.stroke_step_spin.setRange(1, 5)
        self.stroke_step_spin.setValue(DEFAULT_STROKE_STEP)

        self.rgb_panel_delay_spin = add_param("RGB Panel ms", QSpinBox(), 4, 2, "rgb_panel_delay")
        self.rgb_panel_delay_spin.setRange(0, 500)
        self.rgb_panel_delay_spin.setSingleStep(5)
        self.rgb_panel_delay_spin.setValue(DEFAULT_RGB_PANEL_DELAY_MS)
        mode_outer.addLayout(mode_layout)
        root.addWidget(mode_card)

        self.configure_number_inputs()

        # Options card - compact modern option tiles
        toggle_card, toggle_layout = make_card("toggleCard", spacing=12)
        toggle_header = QHBoxLayout()
        toggle_header.setContentsMargins(0, 0, 0, 0)
        toggle_header.setSpacing(8)
        toggle_title = QLabel("Options")
        toggle_title.setObjectName("cardTitle")
        toggle_header.addWidget(toggle_title)
        toggle_hint = QLabel("常用開關")
        toggle_hint.setObjectName("cardHint")
        toggle_header.addWidget(toggle_hint)
        toggle_header.addStretch(1)
        toggle_layout.addLayout(toggle_header)

        toggle_grid = QGridLayout()
        toggle_grid.setHorizontalSpacing(9)
        toggle_grid.setVerticalSpacing(7)

        self.skip_white_check = OptionToggle(
            "Skip White",
            "跳過白色背景",
            checked=True,
        )
        toggle_grid.addWidget(self.skip_white_check, 0, 0)

        self.eye_detail_check = OptionToggle(
            "Eye Detail",
            "補強眼睛細節",
            checked=True,
        )
        toggle_grid.addWidget(self.eye_detail_check, 0, 1)

        self.spiral_fill_check = OptionToggle(
            "Spiral Fill",
            "大色塊蚊香填色",
            checked=False,
        )
        toggle_grid.addWidget(self.spiral_fill_check, 1, 0)

        self.auto_black_check = OptionToggle(
            "Auto Black",
            "線稿自動選黑色",
            checked=False,
        )
        toggle_grid.addWidget(self.auto_black_check, 0, 2)

        for col in range(3):
            toggle_grid.setColumnStretch(col, 1)

        toggle_layout.addLayout(toggle_grid)
        root.addWidget(toggle_card)
        self.update_mode_parameter_state()

        # Action buttons card
        action_card, action_layout = make_card("actionCard", spacing=10)
        action_row = QHBoxLayout()
        action_row.setSpacing(12)
        self.preview_btn = QPushButton("Preview 預覽效果")
        self.preview_btn.setObjectName("previewButton")
        self.preview_btn.clicked.connect(self.preview_thread)
        action_row.addWidget(self.preview_btn)

        self.draw_btn = AnimatedActionButton("▶  Draw Fast 快速繪製")
        self.draw_btn.setObjectName("drawButton")
        self.draw_btn.setMinimumHeight(38)
        self.draw_btn.clicked.connect(self.draw_thread)
        action_row.addWidget(self.draw_btn, 2)
        action_layout.addLayout(action_row)

        elapsed_row = QHBoxLayout()
        elapsed_row.setSpacing(10)
        elapsed_title = QLabel("Elapsed 本次耗時")
        elapsed_title.setObjectName("elapsedTitle")
        self.elapsed_label = QLabel("00:00.0")
        self.elapsed_label.setObjectName("elapsedValue")
        self.elapsed_label.setAlignment(Qt.AlignCenter)
        elapsed_row.addWidget(elapsed_title)
        elapsed_row.addStretch(1)
        elapsed_row.addWidget(self.elapsed_label)
        action_layout.addLayout(elapsed_row)

        self.stop_btn = QPushButton("■  STOP 停止繪製 (Esc)")
        self.stop_btn.setObjectName("stopButton")
        self.stop_btn.setMinimumHeight(36)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.request_stop)
        action_layout.addWidget(self.stop_btn)
        root.addWidget(action_card)

        # Right-side Log card
        log_card, log_layout = make_card("logCard", spacing=8)
        log_card.setMinimumWidth(345)
        log_card.setMaximumWidth(430)
        log_title = QLabel("Log 輸出紀錄")
        log_title.setObjectName("cardTitle")
        log_layout.addWidget(log_title)
        self.log_box = QTextEdit()
        self.log_box.setObjectName("logBox")
        self.log_box.setReadOnly(True)
        self.log_box.setFont(QFont("Cascadia Mono", 9))
        log_layout.addWidget(self.log_box, 1)
        main_layout.addWidget(log_card, 1)

    def apply_modern_theme(self):
        """Modern dark card layout with soft shadows and toggle switches."""
        self.setObjectName("mainWindow")
        self.setStyleSheet("""
            QMainWindow#mainWindow {
                background: #070B16;
            }
            QWidget {
                color: #E5E7EB;
                font-family: "Segoe UI", "Microsoft JhengHei", Arial;
                font-size: 13px;
            }
            QLabel#titleLabel {
                color: #F8FAFC;
                font-size: 25px;
                font-weight: 850;
                letter-spacing: 0.6px;
                padding-top: 2px;
            }
            QLabel#subtitleLabel {
                color: #8EA0C0;
                font-size: 12px;
                padding-bottom: 4px;
            }
            QLabel#cardTitle {
                color: #D8E7FF;
                font-size: 12px;
                font-weight: 750;
                letter-spacing: 0.3px;
                padding-bottom: 2px;
            }
            QFrame#headerCard, QFrame#quickCard, QFrame#settingsCard,
            QFrame#toggleCard, QFrame#actionCard, QFrame#logCard {
                background: rgba(15, 23, 42, 0.92);
                border: 1px solid rgba(72, 98, 145, 0.55);
                border-radius: 13px;
            }
            QFrame#quickCard, QFrame#toggleCard, QFrame#actionCard {
                background: rgba(13, 22, 40, 0.94);
            }
            QLabel#pathLabel {
                background: #0A1326;
                color: #8FA0B7;
                border: 1px solid #243047;
                border-radius: 11px;
                padding: 7px 10px;
            }
            QLabel#statusPill {
                background: rgba(16, 185, 129, 0.10);
                border: 1px solid rgba(16, 185, 129, 0.30);
                border-radius: 12px;
                padding: 6px 10px;
                color: #22C55E;
                font-weight: 700;
            }
            QLabel#elapsedTitle {
                color: #9FB3D1;
                font-weight: 700;
                letter-spacing: 0.2px;
            }
            QLabel#elapsedValue {
                background: rgba(56, 189, 248, 0.10);
                border: 1px solid rgba(56, 189, 248, 0.34);
                border-radius: 12px;
                color: #7DD3FC;
                font-family: "Cascadia Mono", "Consolas", monospace;
                font-size: 16px;
                font-weight: 900;
                padding: 5px 12px;
                min-width: 105px;
            }
            QPushButton {
                background: #162033;
                border: 1px solid #2B3A55;
                border-radius: 11px;
                padding: 8px 12px;
                color: #E5E7EB;
                font-weight: 700;
            }
            QPushButton:hover {
                background: #1F2A44;
                border-color: #38BDF8;
            }
            QPushButton:pressed {
                background: #0F172A;
                padding-top: 9px;
                padding-bottom: 7px;
            }
            QPushButton:disabled {
                background: #101827;
                color: #64748B;
                border-color: #1F2937;
            }
            QPushButton#secondaryButton, QPushButton#previewButton {
                background: #111C31;
                border-color: #334155;
            }
            QPushButton#secondaryButton:hover, QPushButton#previewButton:hover {
                background: #1C2A46;
                border-color: #38BDF8;
            }
            QPushButton#drawButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #22C55E, stop:0.55 #16A34A, stop:1 #0D9488);
                border: 0;
                color: white;
                font-size: 17px;
                font-weight: 900;
                letter-spacing: 0.4px;
            }
            QPushButton#drawButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #34D399, stop:0.55 #22C55E, stop:1 #14B8A6);
            }
            QPushButton#drawButton:disabled {
                background: #1F2937;
                color: #94A3B8;
            }
            QPushButton#stopButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #EF4444, stop:0.55 #DC2626, stop:1 #B91C1C);
                border: 0;
                color: white;
                font-size: 14px;
                font-weight: 900;
                letter-spacing: 0.4px;
            }
            QPushButton#stopButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #F87171, stop:0.55 #EF4444, stop:1 #DC2626);
            }
            QPushButton#stopButton:disabled {
                background: #1F2937;
                color: #64748B;
            }
            QSpinBox, QDoubleSpinBox {
                background: #0B1222;
                border: 1px solid #334155;
                border-radius: 11px;
                padding: 5px 10px;
                min-height: 21px;
                color: #E5E7EB;
                selection-background-color: #2563EB;
            }
            QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                width: 0px;
                height: 0px;
                border: none;
                background: transparent;
            }
            QSpinBox::up-arrow, QSpinBox::down-arrow,
            QDoubleSpinBox::up-arrow, QDoubleSpinBox::down-arrow {
                width: 0px;
                height: 0px;
                image: none;
            }
            QSpinBox:hover, QDoubleSpinBox:hover {
                border-color: #38BDF8;
                background: #0F1A2D;
            }
            QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #60A5FA;
                background: #0F172A;
            }
            QSpinBox:disabled, QDoubleSpinBox:disabled {
                background: #101827;
                color: #475569;
                border-color: #1F2937;
            }
            QLabel#paramLabel {
                color: #CBD5E1;
                font-weight: 650;
            }
            QLabel#paramLabel:disabled {
                color: #475569;
            }
            QComboBox#profileCombo {
                background: rgba(8, 31, 50, 0.98);
                border: 1px solid rgba(56, 189, 248, 0.58);
                border-radius: 10px;
                padding: 5px 10px;
                color: #E5E7EB;
                min-height: 21px;
                outline: 0px;
                selection-background-color: rgba(56, 189, 248, 0.28);
            }
            QComboBox#profileCombo:hover {
                border-color: rgba(56, 189, 248, 0.82);
                background: rgba(10, 43, 67, 0.98);
                outline: 0px;
            }
            QComboBox#profileCombo:focus,
            QComboBox#profileCombo:on {
                border-color: rgba(56, 189, 248, 0.58);
                background: rgba(8, 31, 50, 0.98);
                outline: 0px;
            }
            QComboBox#profileCombo:disabled {
                background: #101827;
                color: #64748B;
                border-color: rgba(30, 41, 59, 0.65);
                outline: 0px;
            }
            QComboBox#profileCombo::drop-down {
                border: 0px;
                background: transparent;
                width: 0px;
                image: none;
                subcontrol-origin: padding;
                subcontrol-position: top right;
            }
            QComboBox#profileCombo::down-arrow {
                image: none;
                width: 0px;
                height: 0px;
            }
            QComboBox#profileCombo::focus {
                outline: 0px;
            }
            QFrame#profileComboPopupContainer {
                background: #071020;
                border: 0px;
                padding: 0px;
                margin: 0px;
                outline: 0px;
            }
            QListView#profileComboPopup {
                background: #071020;
                border: 0px;
                border-radius: 10px;
                padding: 4px;
                color: #DDE7F7;
                outline: 0px;
                selection-background-color: transparent;
                selection-color: #FFFFFF;
            }
            QListView#profileComboPopup::viewport {
                background: #071020;
                border: 0px;
                outline: 0px;
            }
            QComboBox#profileCombo QAbstractItemView {
                background: #071020;
                border: 0px;
                outline: 0px;
                selection-background-color: transparent;
            }
            QListView#profileComboPopup::item {
                min-height: 22px;
                padding: 4px 8px;
                margin: 0px 1px;
                border-radius: 7px;
                color: #CBD5E1;
                background: transparent;
            }
            QListView#profileComboPopup::item:hover {
                color: #F8FAFC;
                background: rgba(56, 189, 248, 0.12);
                border: 0px;
            }
            QListView#profileComboPopup::item:selected {
                color: #FFFFFF;
                background: rgba(37, 99, 235, 0.35);
                border: 0px;
                outline: 0px;
            }
            QListView#profileComboPopup::item:selected:hover {
                background: rgba(56, 189, 248, 0.22);
                border: 0px;
            }
            QRadioButton#modeChip {
                spacing: 8px;
                padding: 6px 9px;
                border: 1px solid #263249;
                border-radius: 10px;
                background: rgba(8, 14, 28, 0.58);
            }
            QRadioButton#modeChip:hover {
                border-color: #38BDF8;
                background: rgba(30, 41, 59, 0.74);
            }
            QRadioButton#modeChip::indicator {
                width: 16px;
                height: 16px;
                border-radius: 6px;
                border: 1px solid #2B3A55;
                background: #0F172A;
            }
            QRadioButton#modeChip::indicator:checked {
                background: #38BDF8;
                border: 1px solid #38BDF8;
            }
            QLabel#cardHint {
                color: #64748B;
                font-size: 12px;
                font-weight: 650;
                padding-top: 1px;
            }
            QFrame#optionToggleCard {
                background: rgba(8, 14, 28, 0.58);
                border: 1px solid rgba(51, 65, 85, 0.78);
                border-radius: 13px;
            }
            QFrame#optionToggleCard:hover {
                background: rgba(15, 23, 42, 0.95);
                border-color: rgba(56, 189, 248, 0.68);
            }
            QFrame#optionToggleCard[checked="true"] {
                background: rgba(14, 51, 78, 0.48);
                border-color: rgba(56, 189, 248, 0.62);
            }
            QFrame#optionToggleCard:disabled {
                background: rgba(8, 14, 28, 0.36);
                border-color: rgba(31, 41, 55, 0.78);
            }
            QFrame#optionToggleCard:disabled QLabel {
                color: #526176;
            }
            QLabel#optionTitle {
                color: #EAF4FF;
                font-size: 12px;
                font-weight: 800;
                letter-spacing: 0.2px;
            }
            QLabel#optionSubtitle {
                color: #8EA0C0;
                font-size: 10px;
                font-weight: 550;
            }
            QFrame#optionToggleCard:disabled QLabel#optionTitle,
            QFrame#optionToggleCard:disabled QLabel#optionSubtitle {
                color: #475569;
            }
            QTextEdit#logBox {
                background: #050816;
                border: 1px solid #263249;
                border-radius: 14px;
                color: #C7D2FE;
                padding: 10px;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 10px;
                margin: 4px 2px 4px 2px;
            }
            QScrollBar::handle:vertical {
                background: #334155;
                border-radius: 5px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: #475569;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

    def _set_mode_from_button(self, btn, checked):
        if checked:
            self.mode_value = btn.mode_value
            self.update_mode_parameter_state()

    def _set_widget_group_enabled(self, enabled, *widgets):
        enabled = bool(enabled)
        for widget in widgets:
            if widget is None:
                continue
            widget.setEnabled(enabled)
            widget.style().unpolish(widget)
            widget.style().polish(widget)
            widget.update()

    def update_mode_parameter_state(self):
        if not hasattr(self, "mode_param_widgets"):
            return

        mode = self.mode_value
        line_modes = {MODE_LINE, MODE_CLEAN_LINE, MODE_DARK_OUTLINE, MODE_SMART_LINE}
        color_modes = {MODE_PALETTE, MODE_CUSTOM_RGB}

        enabled_map = {
            "detail": mode in line_modes,
            "line_move": True,
            "line_gap": mode in line_modes,
            "line_scale": True,
            "stroke_step": mode in line_modes,
            "custom_colors": mode == MODE_CUSTOM_RGB,
            "rgb_panel_delay": mode == MODE_CUSTOM_RGB,
            "sbr_strokes": mode == MODE_SBR,
        }

        tooltips = {
            "detail": "只影響 Smart Line Art / 線稿模式。",
            "line_move": "所有模式都會用到，控制拖曳筆畫速度。",
            "line_gap": "只影響線稿模式的每筆間隔。",
            "line_scale": "所有模式都會用到；有圖片定位時則作為未定位時的備用比例。",
            "stroke_step": "只影響線稿模式，數值越大越會略過線上點。",
            "custom_colors": "只影響 Custom RGB 自訂色模式。",
            "rgb_panel_delay": "只影響 Custom RGB 自訂色模式。",
            "sbr_strokes": "只影響 SBR 筆觸渲染模式。",
        }

        for key, widgets in self.mode_param_widgets.items():
            enabled = enabled_map.get(key, True)
            self._set_widget_group_enabled(enabled, *widgets)
            for widget in widgets:
                widget.setToolTip(tooltips.get(key, ""))

        if hasattr(self, "auto_black_check"):
            self._set_widget_group_enabled(mode in line_modes, self.auto_black_check)
            self.auto_black_check.setToolTip("只在線稿模式有用，會先自動選最深色。")
        if hasattr(self, "eye_detail_check"):
            self._set_widget_group_enabled(mode in color_modes, self.eye_detail_check)
            self.eye_detail_check.setToolTip("只在 Palette / Custom RGB 上色模式補畫眼睛、嘴巴、臉部小細節。")
        if hasattr(self, "spiral_fill_check"):
            self._set_widget_group_enabled(mode in color_modes, self.spiral_fill_check)
            self.spiral_fill_check.setToolTip("只在 Palette / Custom RGB 上色模式使用。")
        if hasattr(self, "skip_white_check"):
            self._set_widget_group_enabled(mode in (MODE_PALETTE, MODE_CUSTOM_RGB, MODE_SBR), self.skip_white_check)
            self.skip_white_check.setToolTip("線稿模式不使用；上色/SBR 模式會跳過白色背景。")

    def set_draw_progress(self, percent):
        p = max(0, min(100, int(percent)))
        if hasattr(self.draw_btn, "setProgress"):
            self.draw_btn.setProgress(p)

        if p >= 100:
            self.draw_btn.setText("✓  完成 100%")
        else:
            self.draw_btn.setText(f"✦  繪製中 Drawing... {p}%")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.request_stop()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self.request_stop()
        if self.keyboard_listener is not None:
            try:
                self.keyboard_listener.stop()
            except Exception:
                pass
            self.keyboard_listener = None
        try:
            pyautogui.mouseUp()
        except Exception:
            pass
        try:
            self.detection_overlay.hide()
            self.detection_overlay.deleteLater()
        except Exception:
            pass
        event.accept()

    def start_global_hotkey(self):
        if pynput_keyboard is None:
            self.log("全域 ESC 未啟用：請先安裝 pynput：py -m pip install pynput")
            return

        def on_press(key):
            try:
                if key == pynput_keyboard.Key.esc:
                    self.signals.stop_requested.emit()
            except Exception:
                pass

        try:
            self.keyboard_listener = pynput_keyboard.Listener(on_press=on_press)
            self.keyboard_listener.daemon = True
            self.keyboard_listener.start()
            self.log("全域 ESC 已啟用：焦點在 Gartic / 瀏覽器時按 Esc 也會停止。")
        except Exception as e:
            self.keyboard_listener = None
            self.log(f"全域 ESC 啟用失敗：{e}")

    def log(self, msg):
        self.signals.log_message.emit(str(msg))

    def _append_log(self, msg):
        self.log_box.append(msg)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def _set_detect_status(self, text, color):
        self.detect_label.setText(text)
        if color:
            if color in ("green", "#22C55E"):
                self.detect_label.setStyleSheet("color: #22C55E; background: rgba(34, 197, 94, 0.10); border: 1px solid rgba(34, 197, 94, 0.32); border-radius: 11px; padding: 7px 10px; font-weight: 600;")
            elif color in ("red", "#EF4444"):
                self.detect_label.setStyleSheet("color: #EF4444; background: rgba(239, 68, 68, 0.10); border: 1px solid rgba(239, 68, 68, 0.32); border-radius: 11px; padding: 7px 10px; font-weight: 600;")
            else:
                self.detect_label.setStyleSheet(f"color: {color}; background: rgba(245, 158, 11, 0.10); border: 1px solid rgba(245, 158, 11, 0.32); border-radius: 11px; padding: 7px 10px; font-weight: 600;")

    def set_detecting(self, is_detecting):
        self.is_detecting = bool(is_detecting)
        self.detect_btn.setEnabled(not is_detecting)
        self.detect_btn.setText("Detecting 偵測中..." if is_detecting else DETECT_BUTTON_TEXT)

    @staticmethod
    def format_elapsed(seconds):
        seconds = max(0.0, float(seconds))
        minutes = int(seconds // 60)
        secs = seconds - minutes * 60
        if minutes >= 60:
            hours = minutes // 60
            minutes = minutes % 60
            return f"{hours:02d}:{minutes:02d}:{secs:04.1f}"
        return f"{minutes:02d}:{secs:04.1f}"

    def update_elapsed_label(self):
        if self.draw_start_time is None:
            return
        elapsed = time.perf_counter() - self.draw_start_time
        self.last_draw_elapsed = elapsed
        self.elapsed_label.setText(self.format_elapsed(elapsed))

    def start_elapsed_timer(self):
        self.draw_start_time = time.perf_counter()
        self.last_draw_elapsed = 0.0
        self.elapsed_label.setText("00:00.0")
        self.elapsed_timer.start()

    def stop_elapsed_timer(self):
        if self.draw_start_time is None:
            return
        elapsed = time.perf_counter() - self.draw_start_time
        self.last_draw_elapsed = elapsed
        self.elapsed_timer.stop()
        formatted = self.format_elapsed(elapsed)
        self.elapsed_label.setText(formatted)
        self.log(f"本次畫畫耗時：{formatted}")
        self.draw_start_time = None

    def update_action_animation(self):
        self.action_pulse = (self.action_pulse + 7) % 360
        if hasattr(self, "draw_btn") and hasattr(self.draw_btn, "setPulse"):
            self.draw_btn.setPulse(self.action_pulse)

    def set_draw_phase(self, phase):
        phase = str(phase or "idle")
        self.draw_phase = phase
        if hasattr(self.draw_btn, "setPhase"):
            self.draw_btn.setPhase(phase)

        if phase == "computing":
            if hasattr(self.draw_btn, "setProgress"):
                self.draw_btn.setProgress(None)
            self.draw_btn.setText("◌  運算中 Computing...")
            if not self.action_anim_timer.isActive():
                self.action_anim_timer.start()
        elif phase == "drawing":
            self.set_draw_progress(0)
            if not self.action_anim_timer.isActive():
                self.action_anim_timer.start()
        else:
            self.action_anim_timer.stop()
            self.action_pulse = 0
            if hasattr(self.draw_btn, "setPulse"):
                self.draw_btn.setPulse(0)
            if hasattr(self.draw_btn, "setProgress"):
                self.draw_btn.setProgress(None)
            self.draw_btn.setText("▶  Draw Fast 快速繪製")

    def set_drawing(self, is_drawing):
        was_drawing = self.is_drawing
        self.is_drawing = bool(is_drawing)
        self.draw_btn.setEnabled(not is_drawing)
        self.stop_btn.setEnabled(bool(is_drawing))

        if is_drawing:
            self.set_draw_phase("computing")
        else:
            self.set_draw_phase("idle")

        if is_drawing and not was_drawing:
            self.start_elapsed_timer()
        elif (not is_drawing) and was_drawing:
            self.stop_elapsed_timer()

    def set_previewing(self, is_previewing):
        self.is_previewing = bool(is_previewing)
        self.preview_btn.setEnabled(not is_previewing)
        self.preview_btn.setText("Previewing 預覽中..." if is_previewing else "Preview 預覽效果")

    def request_stop(self):
        if self.is_drawing:
            self.stop_event.set()
            try:
                pyautogui.mouseUp()
            except Exception:
                pass
            self.log("已送出停止要求，會在目前小段結束後停止。")

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇圖片",
            "",
            "Image Files (*.png *.jpg *.jpeg *.webp *.bmp);;All Files (*)"
        )
        if not path:
            return
        try:
            self.image = Image.open(path).convert("RGBA")
            self.image_path = path
            self.image_label.setText(path)
            self.image_label.setStyleSheet("color: #94A3B8;")
            self.log(f"已成功載入圖片：{path}")
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"無法載入圖片：{e}")

    def apply_image_placement(self, rect):
        self.image_placement = normalize_image_placement(rect)

    def save_image_placement(self, rect=None):
        if rect is not None:
            self.apply_image_placement(rect)
        if self.image_placement is not None:
            self.image_placement = clamp_rect_to_canvas(self.image_placement, self.canvas)
            x1, y1, x2, y2 = self.image_placement
            self.log(f"圖片定位已保存：位置=({x1}, {y1})，大小={x2 - x1}x{y2 - y1}")
        self.image_placement_overlay.hide()
        self.place_btn.setChecked(False)

    def cancel_image_placement(self):
        self.image_placement = normalize_image_placement(self.image_placement_backup)
        self.image_placement_overlay.hide()
        self.place_btn.setChecked(False)
        self.log("圖片定位已取消，已還原前一次位置。")

    def clear_image_placement(self):
        self.image_placement = None
        self.image_placement_overlay.hide()
        if hasattr(self, "place_btn"):
            self.place_btn.setChecked(False)
        self.log("已清除圖片定位，之後會回到依照 Line Scale 置中繪製。")

    def toggle_image_placement_overlay(self, checked):
        if checked:
            if self.image is None:
                QMessageBox.warning(self, "提示", "請先載入圖片")
                self.place_btn.setChecked(False)
                return
            if self.canvas is None:
                QMessageBox.warning(self, "提示", "請先自動偵測畫布，或載入畫布設定檔。")
                self.place_btn.setChecked(False)
                return
            # 圖片定位與偵測校正分開操作，避免誤拖畫布 / 色盤。
            if self.overlay_btn.isChecked():
                self.overlay_btn.setChecked(False)
                self.detection_overlay.hide()
                self.detection_overlay.set_edit_mode(False)
            self.image_placement_backup = normalize_image_placement(self.image_placement)
            scale = min(100, max(10, int(self.line_scale_spin.value()))) if hasattr(self, "line_scale_spin") else DEFAULT_LINE_SCALE
            self.image_placement_overlay.set_context(self.canvas, self.image, self.image_placement, scale_percent=scale)
            self.image_placement_overlay.show_overlay()
            self.log("圖片定位 Overlay 已開啟：拖動中間移動，拖四角/邊框調整大小，滾輪也可縮放，按保存套用。")
        else:
            self.image_placement_overlay.hide()

    def apply_overlay_adjustment(self, canvas, palette, brush_positions, custom_rgb_controls=None):
        # Called continuously while the user drags overlay markers.
        self.canvas = tuple(canvas) if canvas else None
        self.palette = list(palette or [])
        self.brush_positions = list(brush_positions or [])
        self.custom_rgb_controls = normalize_custom_rgb_controls(custom_rgb_controls)
        if self.custom_rgb_controls is None and self.palette:
            self.custom_rgb_controls = estimate_custom_rgb_controls(self.palette, self.canvas)
        if self.canvas is not None:
            x1, y1, x2, y2 = self.canvas
            rgb_ok = "OK" if self.custom_rgb_controls else "--"
            self._set_detect_status(
                f"已校正 - 畫布: ({x1}, {y1}, {x2}, {y2}) | 色盤: {len(self.palette)} 色 | 筆刷: {len(self.brush_positions)} 顆 | RGB: {rgb_ok}",
                "green"
            )

    def finish_overlay_adjustment(self):
        if self.canvas is None:
            return
        x1, y1, x2, y2 = self.canvas
        self.log(
            f"Overlay 拖動校正完成：Canvas=({x1}, {y1}, {x2}, {y2}) | "
            f"Palette={len(self.palette)} | Brush={len(self.brush_positions)} | "
            f"RGB={self.custom_rgb_controls}"
        )

    def snapshot_detection_state(self):
        palette_copy = []
        for item in self.palette or []:
            palette_copy.append(dict(item) if isinstance(item, dict) else item)
        return {
            "canvas": tuple(self.canvas) if self.canvas else None,
            "palette": palette_copy,
            "brush_positions": list(self.brush_positions or []),
            "custom_rgb_controls": normalize_custom_rgb_controls(self.custom_rgb_controls),
        }

    def restore_detection_state(self, snapshot):
        if not snapshot:
            return
        self.canvas = tuple(snapshot.get("canvas")) if snapshot.get("canvas") else None
        self.palette = list(snapshot.get("palette") or [])
        self.brush_positions = list(snapshot.get("brush_positions") or [])
        self.custom_rgb_controls = normalize_custom_rgb_controls(snapshot.get("custom_rgb_controls"))
        if self.custom_rgb_controls is None and self.palette:
            self.custom_rgb_controls = estimate_custom_rgb_controls(self.palette, self.canvas)
        if self.canvas is not None:
            x1, y1, x2, y2 = self.canvas
            rgb_ok = "OK" if self.custom_rgb_controls else "--"
            self._set_detect_status(
                f"已還原 - 畫布: ({x1}, {y1}, {x2}, {y2}) | 色盤: {len(self.palette)} 色 | 筆刷: {len(self.brush_positions)} 顆 | RGB: {rgb_ok}",
                "green"
            )

    def profile_state_to_json(self, snapshot):
        """Convert current detection/calibration data to JSON-safe profile data."""
        snapshot = snapshot or self.snapshot_detection_state()
        palette_data = []
        for item in snapshot.get("palette") or []:
            if isinstance(item, dict):
                pos = item.get("pos", (0, 0))
                color = item.get("color", (56, 189, 248))
                palette_data.append({
                    "pos": [int(pos[0]), int(pos[1])],
                    "color": [int(color[0]), int(color[1]), int(color[2])],
                })
            else:
                x, y = item
                palette_data.append({"pos": [int(x), int(y)], "color": [56, 189, 248]})

        controls = normalize_custom_rgb_controls(snapshot.get("custom_rgb_controls"))
        if controls:
            controls = {
                "swatch": [int(controls["swatch"][0]), int(controls["swatch"][1])],
                "inputs": [[int(x), int(y)] for x, y in controls.get("inputs", [])[:3]],
                "source": controls.get("source", "profile"),
            }

        canvas = snapshot.get("canvas")
        return {
            "version": 1,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "canvas": [int(v) for v in canvas] if canvas else None,
            "palette": palette_data,
            "brush_positions": [[int(x), int(y)] for x, y in (snapshot.get("brush_positions") or [])],
            "custom_rgb_controls": controls,
        }

    def profile_json_to_state(self, data):
        if not isinstance(data, dict):
            raise ValueError("設定檔格式錯誤")
        canvas = data.get("canvas")
        if canvas:
            canvas = tuple(int(v) for v in canvas[:4])

        palette = []
        for item in data.get("palette") or []:
            pos = item.get("pos", (0, 0)) if isinstance(item, dict) else item
            color = item.get("color", (56, 189, 248)) if isinstance(item, dict) else (56, 189, 248)
            palette.append({
                "pos": (int(pos[0]), int(pos[1])),
                "color": (int(color[0]), int(color[1]), int(color[2])),
            })

        brush_positions = [(int(x), int(y)) for x, y in (data.get("brush_positions") or [])]
        controls = normalize_custom_rgb_controls(data.get("custom_rgb_controls"))
        return {
            "canvas": canvas,
            "palette": palette,
            "brush_positions": brush_positions,
            "custom_rgb_controls": controls,
        }

    def load_profiles(self):
        try:
            if PROFILE_FILE.exists():
                with PROFILE_FILE.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    self.profiles = raw.get("profiles", raw if "profiles" not in raw else {})
                else:
                    self.profiles = {}
            else:
                self.profiles = {}
        except Exception as e:
            self.profiles = {}
            self.log(f"設定檔讀取失敗：{e}")
        self.refresh_profile_combo()

    def write_profiles(self):
        try:
            data = {"version": 1, "profiles": self.profiles}
            PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with PROFILE_FILE.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.critical(self, "設定檔錯誤", f"無法寫入設定檔：{e}")
            raise

    def refresh_profile_combo(self, select_name=None):
        if not hasattr(self, "profile_combo"):
            return
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItem("未選擇", "")
        for name in sorted(self.profiles.keys()):
            self.profile_combo.addItem(name, name)
        if select_name:
            idx = self.profile_combo.findData(select_name)
            if idx >= 0:
                self.profile_combo.setCurrentIndex(idx)
        self.profile_combo.blockSignals(False)

    def default_profile_name(self):
        if self.canvas:
            x1, y1, x2, y2 = self.canvas
            return f"畫布 {abs(x2 - x1)}x{abs(y2 - y1)}"
        return time.strftime("畫布設定 %m%d-%H%M")

    def save_profile_with_name(self, name, snapshot=None):
        name = str(name or "").strip()
        if not name:
            return False
        snapshot = snapshot or self.snapshot_detection_state()
        if not snapshot.get("canvas"):
            QMessageBox.information(self, "尚未偵測", "目前沒有畫布座標可以保存，請先 Auto Detect 或完成 Overlay 校正。")
            return False
        self.profiles[name] = self.profile_state_to_json(snapshot)
        self.write_profiles()
        self.refresh_profile_combo(select_name=name)
        self.log(f"畫布設定檔已保存：{name} → {PROFILE_FILE}")
        return True

    def save_current_profile_dialog(self):
        if self.canvas is None:
            QMessageBox.information(self, "尚未偵測", "請先 Auto Detect，或使用 Overlay 校正後再保存設定檔。")
            return
        current = self.profile_combo.currentData() if hasattr(self, "profile_combo") else ""
        default_name = current or self.default_profile_name()
        name, ok = QInputDialog.getText(self, "保存畫布設定檔", "設定檔名稱：", text=default_name)
        if ok:
            self.save_profile_with_name(name)

    def save_overlay_profile(self, canvas=None, palette=None, brush_positions=None, custom_rgb_controls=None):
        if canvas is not None:
            self.apply_overlay_adjustment(canvas, palette, brush_positions, custom_rgb_controls)
        snapshot = self.snapshot_detection_state()
        default_name = self.default_profile_name()
        # Overlay 是最上層視窗，先暫時隱藏，避免名稱輸入框被蓋住。
        self.detection_overlay.hide()
        name, ok = QInputDialog.getText(self, "保存畫布設定檔", "設定檔名稱：", text=default_name)
        if not ok:
            if self.overlay_btn.isChecked():
                self.detection_overlay.show_overlay()
            return
        if self.save_profile_with_name(name, snapshot):
            self.overlay_calibration_backup = None
            self.close_overlay_calibration_ui()
            self.log("Overlay 校正已保存為設定檔，可在快速控制的畫布設定檔直接載入。")
        elif self.overlay_btn.isChecked():
            self.detection_overlay.show_overlay()

    def load_selected_profile(self):
        if not hasattr(self, "profile_combo"):
            return
        name = self.profile_combo.currentData()
        if not name:
            QMessageBox.information(self, "尚未選擇", "請先選擇一個畫布設定檔。")
            return
        data = self.profiles.get(name)
        if not data:
            QMessageBox.warning(self, "找不到設定檔", f"找不到設定檔：{name}")
            return
        try:
            state = self.profile_json_to_state(data)
            self.restore_detection_state(state)
            self.detection_overlay.set_detection(self.canvas, self.palette, self.brush_positions, self.custom_rgb_controls)
            self.log(f"已載入畫布設定檔：{name}")
        except Exception as e:
            QMessageBox.critical(self, "載入失敗", f"設定檔載入失敗：{e}")

    def close_overlay_calibration_ui(self):
        self.detection_overlay.set_edit_mode(False)
        self.detection_overlay.hide()
        if hasattr(self, "overlay_btn"):
            self.overlay_btn.blockSignals(True)
            self.overlay_btn.setChecked(False)
            self.overlay_btn.blockSignals(False)
            self.overlay_btn.setText("Overlay 偵測/拖動校正")

    def update_detection_overlay(self, canvas, palette, brush_positions, custom_rgb_controls=None):
        self.detection_overlay.set_detection(canvas, palette, brush_positions, custom_rgb_controls)
        if self.overlay_btn.isChecked():
            self.detection_overlay.set_edit_mode(True)
            self.detection_overlay.show_overlay()
            self.overlay_btn.setText("校正中：看 Overlay 保存/取消")

    def toggle_overlay_calibration(self, checked):
        if checked:
            if hasattr(self, "place_btn") and self.place_btn.isChecked():
                self.place_btn.setChecked(False)
                self.image_placement_overlay.hide()
            if self.canvas is None:
                QMessageBox.information(self, "尚未偵測", "請先按 Auto Detect，自動偵測畫布、色盤、筆刷與 RGB 位置。")
                self.overlay_btn.setChecked(False)
                return
            self.overlay_calibration_backup = self.snapshot_detection_state()
            self.detection_overlay.set_detection(self.canvas, self.palette, self.brush_positions, self.custom_rgb_controls)
            self.detection_overlay.set_edit_mode(True)
            self.detection_overlay.show_overlay()
            self.overlay_btn.setText("校正中：看 Overlay 保存/取消")
            self.log("Overlay 校正已啟用：請在 Overlay 上拖動，最後按保存、取消或保存設定檔。")
        else:
            # Main UI button acts as Cancel when closing calibration without pressing Save.
            self.cancel_overlay_calibration()

    def save_overlay_calibration(self, canvas=None, palette=None, brush_positions=None, custom_rgb_controls=None):
        if canvas is not None:
            self.apply_overlay_adjustment(canvas, palette, brush_positions, custom_rgb_controls)
        self.overlay_calibration_backup = None
        self.close_overlay_calibration_ui()
        if self.canvas is not None:
            x1, y1, x2, y2 = self.canvas
            self.log(
                f"Overlay 校正已保存：Canvas=({x1}, {y1}, {x2}, {y2}) | "
                f"Palette={len(self.palette)} | Brush={len(self.brush_positions)} | RGB={self.custom_rgb_controls}"
            )

    def cancel_overlay_calibration(self):
        if self.overlay_calibration_backup is not None:
            self.restore_detection_state(self.overlay_calibration_backup)
            self.detection_overlay.set_detection(self.canvas, self.palette, self.brush_positions, self.custom_rgb_controls)
            self.log("Overlay 校正已取消，已還原到開啟校正前的位置。")
        else:
            self.log("Overlay 校正已關閉。")
        self.overlay_calibration_backup = None
        self.close_overlay_calibration_ui()

    def auto_detect_thread(self):
        if self.is_detecting:
            return
        self.set_detecting(True)
        threading.Thread(target=self.auto_detect, daemon=True).start()

    def auto_detect(self):
        try:
            self.log("正在截圖並辨識畫布與色盤位置...")
            img_rgb, offset_x, offset_y = capture_screen_rgb()
            canvas = detect_canvas(img_rgb)
            if canvas is None:
                self.log("錯誤：找不到畫布！請確保網頁沒有被遮擋，縮放率為 100%。")
                return

            x1, y1, x2, y2 = canvas
            canvas_screen = (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)
            palette = detect_palette(img_rgb, canvas)
            brush_positions = detect_brush_buttons(img_rgb, canvas)

            # Detect RGB controls before converting palette points to screen
            # coordinates.  The screenshot image and canvas are still in the
            # capture coordinate system here.
            palette_capture = [dict(p) if isinstance(p, dict) else p for p in palette]
            rgb_controls = detect_custom_rgb_controls(img_rgb, canvas, palette_capture)
            rgb_controls = offset_custom_rgb_controls(rgb_controls, offset_x, offset_y)

            for p in palette:
                px, py = p["pos"]
                p["pos"] = (int(px + offset_x), int(py + offset_y))

            brush_positions = [(int(px + offset_x), int(py + offset_y)) for px, py in brush_positions]

            self.canvas = canvas_screen
            self.palette = palette
            self.brush_positions = brush_positions
            self.custom_rgb_controls = rgb_controls or estimate_custom_rgb_controls(self.palette, self.canvas)

            self.signals.detect_status.emit(
                f"已就緒 - 畫布: {self.canvas} | 色盤: {len(self.palette)} 色 | 筆刷: {len(self.brush_positions)} 顆",
                "green"
            )
            self.log(f"成功偵測畫布區域：{self.canvas}")
            self.log(f"抓取到色盤數量：{len(self.palette)} 色")
            self.log(f"色盤按鈕座標：{[p['pos'] for p in self.palette]}")
            self.log(f"抓取到筆刷按鈕數量：{len(self.brush_positions)} 顆")
            self.log(f"筆刷按鈕座標：{self.brush_positions}")
            self.log(f"RGB 面板座標：{self.custom_rgb_controls}")
            self.signals.overlay_ready.emit(self.canvas, self.palette, self.brush_positions, self.custom_rgb_controls)
        except Exception as e:
            self.log(f"偵測過程中發生異常：{e}")
        finally:
            self.signals.detecting_changed.emit(False)

    def _collect_preview_params(self):
        cps = max(1.0, float(self.cps_spin.value()))
        brush_key = clamp_brush_key(self.brush_spin.value())
        skip_white = self.skip_white_check.isChecked()
        mode = self.mode_value
        detail = int(self.detail_spin.value())
        line_move_ms = max(1, int(self.line_move_spin.value()))
        line_gap_ms = max(0, int(self.line_gap_spin.value()))
        line_scale = min(100, max(40, int(self.line_scale_spin.value())))
        stroke_step = max(1, int(self.stroke_step_spin.value()))
        custom_colors = int(np.clip(int(self.custom_colors_spin.value()), 8, MAX_CUSTOM_COLORS))
        sbr_strokes = int(np.clip(int(self.sbr_strokes_spin.value()), 50, 1500))
        rgb_panel_delay_ms = int(np.clip(int(self.rgb_panel_delay_spin.value()), 0, 500))
        auto_black = self.auto_black_check.isChecked()
        eye_detail = self.eye_detail_check.isChecked()
        spiral_fill = self.spiral_fill_check.isChecked()
        return (
            cps,
            brush_key,
            skip_white,
            mode,
            detail,
            line_move_ms,
            line_gap_ms,
            line_scale,
            stroke_step,
            custom_colors,
            sbr_strokes,
            rgb_panel_delay_ms,
            auto_black,
            eye_detail,
            spiral_fill,
        )

    def preview_thread(self):
        if self.is_previewing:
            return
        if self.image is None:
            QMessageBox.warning(self, "提示", "請先載入圖片")
            return
        if self.canvas is None:
            QMessageBox.warning(self, "提示", "請先點擊自動偵測畫布")
            return
        args = self._collect_preview_params()
        self.set_previewing(True)
        threading.Thread(target=self.build_preview, args=args, daemon=True).start()

    def build_preview(self, cps, brush_key, skip_white, mode, detail, line_move_ms, line_gap_ms, line_scale, stroke_step, custom_colors, sbr_strokes, rgb_panel_delay_ms, auto_black, eye_detail, spiral_fill):
        try:
            brush_px = gartic_brush_pixels(brush_key)
            x1, y1, x2, y2 = self.canvas
            canvas_w, canvas_h = abs(x2 - x1), abs(y2 - y1)
            target_x, target_y, target_w, target_h, using_placement = target_area_from_placement(
                self.canvas, self.image_placement, line_scale
            )

            if mode == MODE_SBR:
                line_max_w = max(1, int(target_w))
                line_max_h = max(1, int(target_h))
                detected_colors = [p["color"] for p in self.palette]
                mapping_colors = palette_colors_for_mapping(detected_colors)
                strokes, img_size, content = image_to_sbr_strokes(
                    self.image, line_max_w, line_max_h, mapping_colors,
                    stroke_count=sbr_strokes, brush_px=brush_px, skip_white=skip_white
                )
                offset_x = target_x + (target_w - content.size[0]) / 2
                offset_y = target_y + (target_h - content.size[1]) / 2
                preview = compose_canvas_preview_at(content, canvas_w, canvas_h, offset_x, offset_y)
                pos_text = "定位" if using_placement else "置中"
                color_changes = count_color_transitions([stroke[0] for stroke in strokes])
                estimated = estimate_sbr_preview_seconds(
                    strokes,
                    cps,
                    line_move_ms,
                    color_changes,
                    brush_seconds=estimate_brush_select_seconds(brush_key, self.brush_positions),
                )
                stats = preview_stats_text(estimated, len(strokes), color_changes)
                info = f"SBR 筆觸預覽 | {stats} | {pos_text} | 畫布: {canvas_w}x{canvas_h} | 目標區: {int(target_w)}x{int(target_h)} | 圖像: {img_size[0]}x{img_size[1]} | 筆刷鍵: {brush_key}"
            elif mode in (MODE_SMART_LINE, MODE_LINE, MODE_CLEAN_LINE, MODE_DARK_OUTLINE):
                self.log("運算中：正在分析線稿，UI 已改成非阻塞模式。")
                line_max_w = max(1, int(target_w))
                line_max_h = max(1, int(target_h))
                strokes, img_size = image_to_smart_line_strokes(self.image, line_max_w, line_max_h, detail=detail)
                before_air = stroke_air_distance(strokes)
                strokes = optimize_stroke_order(strokes)
                after_air = stroke_air_distance(strokes)
                preview_strokes = [decimate_stroke(stroke, stroke_step) for stroke in strokes]
                content = line_strokes_to_image(preview_strokes, img_size, line_width=brush_px)
                offset_x = target_x + (target_w - content.size[0]) / 2
                offset_y = target_y + (target_h - content.size[1]) / 2
                preview = compose_canvas_preview_at(content, canvas_w, canvas_h, offset_x, offset_y)
                pos_text = "定位" if using_placement else "置中"
                color_changes = 1 if auto_black and self.palette else 0
                estimated = estimate_line_preview_seconds(
                    preview_strokes,
                    line_move_ms,
                    line_gap_ms,
                    color_changes=color_changes,
                    brush_seconds=estimate_brush_select_seconds(brush_key, self.brush_positions),
                )
                stats = preview_stats_text(estimated, len(preview_strokes), color_changes)
                saved = 0.0 if before_air <= 0 else max(0.0, (before_air - after_air) / before_air * 100.0)
                info = f"整合線稿預覽 | {stats} | {pos_text} | 畫布: {canvas_w}x{canvas_h} | 目標區: {int(target_w)}x{int(target_h)} | 圖像: {img_size[0]}x{img_size[1]} | 筆刷鍵: {brush_key} | Step: {stroke_step} | 空移動減少: {saved:.1f}%"
            elif mode in (MODE_PALETTE, MODE_CUSTOM_RGB):
                self.log("運算中：正在量化色彩與排序路徑，UI 已改成非阻塞模式。")
                color_max_w = max(1, int(target_w))
                color_max_h = max(1, int(target_h))
                cell_step_px = color_fill_step(mode, brush_px)
                white_protect_radius = white_protect_radius_for_brush(brush_px, cell_step_px)
                cells_w = max(1, color_max_w // cell_step_px)
                cells_h = max(1, color_max_h // cell_step_px)
                detected_colors = [p["color"] for p in self.palette]
                mapping_colors = palette_colors_for_mapping(detected_colors)

                if mode != MODE_CUSTOM_RGB and not detected_colors:
                    self.log("錯誤：未偵測到色盤顏色，無法生成全彩預覽。")
                    return

                if mode == MODE_CUSTOM_RGB:
                    color_map, mapping_colors, img_size = image_to_custom_rgb_map(
                        self.image, cells_w, cells_h, color_count=custom_colors, skip_white=skip_white,
                        white_protect_radius=white_protect_radius
                    )
                    label = "自訂 RGB 預覽"
                else:
                    color_map, img_size = image_to_palette_map(
                        self.image, mapping_colors, cells_w, cells_h, skip_white=skip_white,
                        white_protect_radius=white_protect_radius
                    )
                    label = "全彩色盤預覽"

                draw_w, draw_h = img_size
                draw_px_w = (draw_w - 1) * cell_step_px + brush_px
                draw_px_h = (draw_h - 1) * cell_step_px + brush_px
                offset_px_x = target_x + (target_w - draw_px_w) / 2
                offset_px_y = target_y + (target_h - draw_px_h) / 2

                if skip_white and mode == MODE_PALETTE:
                    for wi in white_palette_indices(mapping_colors):
                        color_map[color_map == wi] = -1

                hole_area = color_hole_area(mode, brush_px)
                if hole_area > 0:
                    color_map = solidify_color_map(color_map, len(mapping_colors), hole_area)
                color_map, _sealed_preview_gaps = seal_thin_color_gaps(
                    color_map,
                    len(mapping_colors),
                    max_thickness=max(2, int(round(brush_px / max(1, cell_step_px)))),
                    max_area=max(18, int(max(color_map.shape) * 1.5))
                )

                bridge_gap = color_run_bridge_gap(mode, brush_px)
                exact_runs_by_color, _exact_pixel_counts = build_color_runs(color_map, len(mapping_colors))
                spiral_paths_by_color = [[] for _ in range(len(mapping_colors))]
                fill_source_map = color_map
                spiral_stats = None
                spiral_count = 0
                spiral_enabled = bool(spiral_fill and mode in (MODE_PALETTE, MODE_CUSTOM_RGB))

                if spiral_enabled:
                    spiral_paths_by_color, fill_source_map, spiral_stats = build_spiral_fill_paths(
                        color_map,
                        len(mapping_colors),
                        brush_px=brush_px,
                        cell_step_px=cell_step_px,
                        remove_from_fallback=False
                    )
                    spiral_paths_by_color = [optimize_spiral_path_order(paths) for paths in spiral_paths_by_color]
                    spiral_count = sum(len(paths) for paths in spiral_paths_by_color)

                raw_runs_by_color, _pixel_counts = build_color_runs(
                    fill_source_map,
                    len(mapping_colors),
                    bridge_gap=bridge_gap
                )
                runs_by_color = [optimize_run_order(runs) for runs in raw_runs_by_color]

                eye_detail_runs_by_color = None
                eye_detail_offsets = None
                eye_detail_brush_key = 0 if brush_key == 0 else 1
                eye_detail_cell_step_px = 1
                eye_detail_brush_px = gartic_brush_pixels(eye_detail_brush_key)

                if eye_detail and mode in (MODE_PALETTE, MODE_CUSTOM_RGB):
                    eye_scale = min(100, max(line_scale, 95))
                    eye_max_w = max(1, int(target_w if using_placement else canvas_w * eye_scale / 100))
                    eye_max_h = max(1, int(target_h if using_placement else canvas_h * eye_scale / 100))
                    eye_map, eye_img_size = image_to_eye_detail_map(
                        self.image, mapping_colors, eye_max_w, eye_max_h
                    )
                    for wi in white_palette_indices(mapping_colors):
                        eye_map[eye_map == wi] = -1
                    if np.any(eye_map >= 0):
                        raw_eye_runs, _eye_counts = build_color_runs(eye_map, len(mapping_colors), bridge_gap=0)
                        eye_detail_runs_by_color = [optimize_run_order(runs) for runs in raw_eye_runs]
                        eye_w, eye_h = eye_img_size
                        eye_draw_px_w = (eye_w - 1) * eye_detail_cell_step_px + eye_detail_brush_px
                        eye_draw_px_h = (eye_h - 1) * eye_detail_cell_step_px + eye_detail_brush_px
                        eye_detail_offsets = (
                            (target_x + (target_w - eye_draw_px_w) / 2) if using_placement else (canvas_w - eye_draw_px_w) / 2,
                            (target_y + (target_h - eye_draw_px_h) / 2) if using_placement else (canvas_h - eye_draw_px_h) / 2,
                        )

                color_order = color_draw_order(mapping_colors)
                white_indices = white_palette_indices(mapping_colors) if skip_white and mode != MODE_CUSTOM_RGB else set()
                eye_order = []
                if eye_detail_runs_by_color is not None:
                    eye_white_indices = set(white_palette_indices(mapping_colors))
                    eye_order = [idx for idx in color_draw_order(mapping_colors) if idx not in eye_white_indices]

                preview = planned_color_runs_to_canvas_preview(
                    canvas_w,
                    canvas_h,
                    mapping_colors,
                    color_order,
                    runs_by_color,
                    spiral_paths_by_color,
                    offset_px_x,
                    offset_px_y,
                    brush_px,
                    cell_step_px,
                    white_indices=white_indices,
                    eye_order=eye_order,
                    eye_runs_by_color=eye_detail_runs_by_color,
                    eye_offset_x=eye_detail_offsets[0] if eye_detail_offsets else 0,
                    eye_offset_y=eye_detail_offsets[1] if eye_detail_offsets else 0,
                    eye_brush_px=eye_detail_brush_px,
                    eye_cell_step_px=eye_detail_cell_step_px,
                )
                total_ops, color_changes, estimated = color_plan_preview_metrics(
                    mode,
                    mapping_colors,
                    color_order,
                    runs_by_color,
                    spiral_paths_by_color,
                    white_indices,
                    eye_order,
                    eye_detail_runs_by_color,
                    cps,
                    line_move_ms,
                    brush_px,
                    cell_step_px,
                    rgb_panel_delay_ms,
                    eye_brush_px=eye_detail_brush_px,
                    eye_cell_step_px=eye_detail_cell_step_px,
                    brush_seconds=estimate_brush_select_seconds(brush_key, self.brush_positions),
                )
                exact_run_count = sum(len(runs) for runs in exact_runs_by_color)
                planned_run_count = sum(len(runs) for runs in runs_by_color)
                reduced = 0.0 if exact_run_count <= 0 else max(0.0, (exact_run_count - planned_run_count) / exact_run_count * 100.0)
                spiral_state = f"ON/{spiral_count}" if spiral_enabled else "OFF"
                pos_text = "定位" if using_placement else "置中"
                stats = preview_stats_text(estimated, total_ops, color_changes)
                info = f"{label} | {stats} | {pos_text} | 畫布: {canvas_w}x{canvas_h} | 目標區: {int(target_w)}x{int(target_h)} | 網格: {img_size[0]}x{img_size[1]} | 筆刷鍵: {brush_key} | 格距: {cell_step_px} | Spiral: {spiral_state} | Runs: {exact_run_count}->{planned_run_count} ({reduced:.1f}%)"
            else:
                self.log(f"未知模式：{mode}")
                return

            self.signals.preview_ready.emit(preview, info)
        except Exception as e:
            self.log(f"預覽生成失敗：{e}")
        finally:
            self.signals.previewing_changed.emit(False)

    def show_preview(self, preview, info):
        dialog = QDialog(self)
        dialog.setWindowTitle("Preview 預覽")
        layout = QVBoxLayout(dialog)
        info_label = QLabel(info)
        info_label.setWordWrap(True)
        info_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(info_label)

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setStyleSheet("background: white;")
        image_label.setPixmap(pil_to_qpixmap(preview))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.addWidget(image_label)
        scroll.setWidget(container)
        layout.addWidget(scroll)

        dialog.resize(720, 620)
        dialog.show()
        self.preview_windows.append(dialog)

    def draw_thread(self):
        if self.is_drawing:
            return
        if self.image is None or self.canvas is None:
            QMessageBox.warning(self, "欄位遺失", "請確認已載入圖片並完成自動偵測。")
            return
        cps = max(1.0, float(self.cps_spin.value()))
        brush_key = clamp_brush_key(self.brush_spin.value())
        skip_white = self.skip_white_check.isChecked()
        mode = self.mode_value
        detail = int(self.detail_spin.value())
        line_move_ms = max(1, int(self.line_move_spin.value()))
        line_gap_ms = max(0, int(self.line_gap_spin.value()))
        line_scale = min(100, max(40, int(self.line_scale_spin.value())))
        stroke_step = max(1, int(self.stroke_step_spin.value()))
        custom_colors = int(np.clip(int(self.custom_colors_spin.value()), 8, MAX_CUSTOM_COLORS))
        sbr_strokes = int(np.clip(int(self.sbr_strokes_spin.value()), 50, 1500))
        rgb_panel_delay_ms = int(np.clip(int(self.rgb_panel_delay_spin.value()), 0, 500))
        auto_black = self.auto_black_check.isChecked()
        eye_detail = self.eye_detail_check.isChecked()
        spiral_fill = self.spiral_fill_check.isChecked()

        # Drawing should happen on the real Gartic canvas only.  If the image
        # placement overlay is still visible, hide it before moving the mouse so
        # the semi-transparent reference image will not cover the canvas or steal
        # clicks while drawing.
        if hasattr(self, "image_placement_overlay") and self.image_placement_overlay.isVisible():
            self.image_placement_overlay.hide()
        if hasattr(self, "place_btn") and self.place_btn.isChecked():
            self.place_btn.blockSignals(True)
            self.place_btn.setChecked(False)
            self.place_btn.blockSignals(False)

        self.set_drawing(True)
        self.stop_event.clear()

        def _start_worker():
            threading.Thread(
                target=self.draw_fast,
                args=(cps, brush_key, skip_white, mode, detail, line_move_ms, line_gap_ms, line_scale, stroke_step, custom_colors, sbr_strokes, rgb_panel_delay_ms, auto_black, eye_detail, spiral_fill),
                daemon=True
            ).start()

        # 先讓 Qt 有一個 event loop 週期更新按鈕/動畫，再開始重運算。
        QTimer.singleShot(30, _start_worker)

    def wait_or_stop(self, seconds):
        end_time = time.time() + seconds
        while time.time() < end_time:
            raise_if_stopped(self.stop_event)
            time.sleep(min(0.05, max(0, end_time - time.time())))
        raise_if_stopped(self.stop_event)

    def wait_before_drawing(self):
        if COUNTDOWN_SECONDS > 0:
            self.log(f"請在 {COUNTDOWN_SECONDS} 秒內將 Gartic 視窗切換至最上層！")
            self.wait_or_stop(COUNTDOWN_SECONDS)
        else:
            self.log("準備開始繪製。")
            raise_if_stopped(self.stop_event)

    def draw_fast(self, cps, brush_key, skip_white, mode, detail, line_move_ms, line_gap_ms, line_scale, stroke_step, custom_colors, sbr_strokes, rgb_panel_delay_ms, auto_black, eye_detail, spiral_fill):
        try:
            last_progress_emit = -1

            def report_progress(done, total):
                nonlocal last_progress_emit
                total = max(1, int(total))
                done = max(0, min(int(done), total))
                percent = int(round(done * 100.0 / total))
                percent = max(0, min(100, percent))
                if percent != last_progress_emit:
                    self.signals.draw_progress_changed.emit(percent)
                    last_progress_emit = percent

            delay = 1.0 / cps
            rgb_panel_delay = max(0.0, min(0.5, rgb_panel_delay_ms / 1000.0))
            brush_px = gartic_brush_pixels(brush_key)
            x1, y1, x2, y2 = self.canvas
            left, top = min(x1, x2), min(y1, y2)
            canvas_w, canvas_h = abs(x2 - x1), abs(y2 - y1)
            target_x, target_y, target_w, target_h, using_placement = target_area_from_placement(
                self.canvas, self.image_placement, line_scale
            )
            if using_placement:
                self.log(f"使用圖片定位區繪製：offset=({int(target_x)}, {int(target_y)})，size={int(target_w)}x{int(target_h)}")
            hotkey_focus_point = (
                int(round(left + canvas_w / 2)),
                int(round(max(0, top - 42))),
            )

            palette_positions = [p["pos"] for p in self.palette]
            palette_colors = [p["color"] for p in self.palette]
            mapping_colors = palette_colors_for_mapping(palette_colors)

            if mode == MODE_SBR:
                self.log("運算中：正在生成 SBR 筆觸序列，UI 已改成非阻塞模式。")
                if not palette_positions:
                    self.log("錯誤：未偵測到有效色盤，取消 SBR 繪製。")
                    return
                line_max_w = max(1, int(target_w))
                line_max_h = max(1, int(target_h))
                strokes, img_size, _preview = image_to_sbr_strokes(
                    self.image, line_max_w, line_max_h, mapping_colors,
                    stroke_count=sbr_strokes, brush_px=brush_px, skip_white=skip_white
                )
                draw_w, draw_h = img_size
                offset_x = target_x + (target_w - draw_w) / 2
                offset_y = target_y + (target_h - draw_h) / 2
                total_expected_ops = max(1, len(strokes))
                self.log(f"SBR 筆觸序列計算完畢，總筆觸：{len(strokes)}")
                self.signals.draw_phase_changed.emit("drawing")
                report_progress(0, total_expected_ops)
                self.wait_before_drawing()
                select_gartic_brush(brush_key, self.brush_positions, self.stop_event, focus_point=hotkey_focus_point)
                self.log(f"已切換 Gartic 筆刷：{brush_key}")

                current_color = None
                total_ops = 0
                for color_idx, start, end, width in strokes:
                    raise_if_stopped(self.stop_event)
                    if color_idx != current_color:
                        px, py = palette_positions[color_idx]
                        pyautogui.click(px, py)
                        self.wait_or_stop(PALETTE_SELECT_DELAY)
                        current_color = color_idx
                    start_pt = screen_point_px(left, top, offset_x, offset_y, start)
                    end_pt = screen_point_px(left, top, offset_x, offset_y, end)
                    draw_stroke_path([start_pt, end_pt], max(0.001, line_move_ms / 1000.0), self.stop_event)
                    total_ops += 1
                    report_progress(total_ops, total_expected_ops)
                    if total_ops % 10 == 0:
                        self.wait_or_stop(max(delay, 0.001))
                report_progress(total_expected_ops, total_expected_ops)
                self.log(f"SBR 繪製完畢，總筆觸：{total_ops}")

            elif mode in (MODE_SMART_LINE, MODE_LINE, MODE_CLEAN_LINE, MODE_DARK_OUTLINE):
                line_max_w = max(1, int(target_w))
                line_max_h = max(1, int(target_h))
                strokes, img_size = image_to_smart_line_strokes(self.image, line_max_w, line_max_h, detail=detail)
                before_air = stroke_air_distance(strokes)
                strokes = optimize_stroke_order(strokes)
                after_air = stroke_air_distance(strokes)
                draw_w, draw_h = img_size
                offset_x = target_x + (target_w - draw_w) / 2
                offset_y = target_y + (target_h - draw_h) / 2
                saved = 0.0 if before_air <= 0 else max(0.0, (before_air - after_air) / before_air * 100.0)
                total_expected_ops = max(1, len(strokes))
                self.log(f"整合線稿序列計算完畢，總筆畫：{len(strokes)}")
                self.log(f"TSP 路徑排序完成：空移動距離約減少 {saved:.1f}%")
                self.signals.draw_phase_changed.emit("drawing")
                report_progress(0, total_expected_ops)
                self.wait_before_drawing()
                select_gartic_brush(brush_key, self.brush_positions, self.stop_event, focus_point=hotkey_focus_point)
                self.log(f"已切換 Gartic 筆刷：{brush_key}")

                if auto_black and palette_positions:
                    black_idx = darkest_palette_index(palette_colors)
                    px, py = palette_positions[black_idx]
                    pyautogui.click(px, py)
                    self.wait_or_stop(0.10)
                    self.log(f"已自動選黑色：index={black_idx + 1}")

                total_ops = 0
                for stroke in strokes:
                    raise_if_stopped(self.stop_event)
                    draw_stroke = decimate_stroke(stroke, stroke_step)
                    screen_points = [screen_point_px(left, top, offset_x, offset_y, pt) for pt in draw_stroke]
                    draw_stroke_path(screen_points, line_move_ms / 1000.0, self.stop_event)
                    total_ops += 1
                    report_progress(total_ops, total_expected_ops)
                    if line_gap_ms > 0:
                        self.wait_or_stop(line_gap_ms / 1000.0)
                    else:
                        raise_if_stopped(self.stop_event)
                report_progress(total_expected_ops, total_expected_ops)
                self.log(f"線稿繪製完畢，總操作筆畫：{total_ops}")

            else:
                if mode != MODE_CUSTOM_RGB and not palette_colors:
                    self.log("錯誤：未偵測到有效色盤，取消全彩繪製。")
                    return
                color_max_w = max(1, int(target_w))
                color_max_h = max(1, int(target_h))
                cell_step_px = color_fill_step(mode, brush_px)
                white_protect_radius = white_protect_radius_for_brush(brush_px, cell_step_px)
                cells_w = max(1, color_max_w // cell_step_px)
                cells_h = max(1, color_max_h // cell_step_px)

                if mode == MODE_CUSTOM_RGB:
                    color_map, mapping_colors, img_size = image_to_custom_rgb_map(
                        self.image, cells_w, cells_h, color_count=custom_colors, skip_white=skip_white,
                        white_protect_radius=white_protect_radius
                    )
                    self.log(f"自訂 RGB 序列計算完畢，顏色數：{len(mapping_colors)}")
                else:
                    color_map, img_size = image_to_palette_map(
                        self.image, mapping_colors, cells_w, cells_h, skip_white=skip_white,
                        white_protect_radius=white_protect_radius
                    )
                    self.log("全彩色盤最佳化序列計算完畢。")
                draw_w, draw_h = img_size
                draw_px_w = (draw_w - 1) * cell_step_px + brush_px
                draw_px_h = (draw_h - 1) * cell_step_px + brush_px
                offset_px_x = target_x + (target_w - draw_px_w) / 2
                offset_px_y = target_y + (target_h - draw_px_h) / 2

                if skip_white and mode == MODE_PALETTE:
                    for wi in white_palette_indices(mapping_colors):
                        color_map[color_map == wi] = -1

                hole_area = color_hole_area(mode, brush_px)
                if hole_area > 0:
                    before_holes = int(np.sum(color_map < 0))
                    color_map = solidify_color_map(color_map, len(mapping_colors), hole_area)
                    filled_holes = max(0, before_holes - int(np.sum(color_map < 0)))
                    if filled_holes > 0:
                        self.log(f"小縫補色完成：補上 {filled_holes} 格")

                color_map, sealed_gaps = seal_thin_color_gaps(
                    color_map,
                    len(mapping_colors),
                    max_thickness=max(2, int(round(brush_px / max(1, cell_step_px)))),
                    max_area=max(18, int(max(color_map.shape) * 1.5))
                )
                if sealed_gaps > 0:
                    self.log(f"細縫封口完成：補上 {sealed_gaps} 格，減少全彩白色空隙")

                bridge_gap = color_run_bridge_gap(mode, brush_px)
                exact_runs_by_color, _exact_pixel_counts = build_color_runs(color_map, len(mapping_colors))

                spiral_paths_by_color = [[] for _ in range(len(mapping_colors))]
                fill_source_map = color_map
                spiral_stats = None
                spiral_enabled = bool(spiral_fill and mode in (MODE_PALETTE, MODE_CUSTOM_RGB))
                if spiral_enabled:
                    spiral_paths_by_color, fill_source_map, spiral_stats = build_spiral_fill_paths(
                        color_map,
                        len(mapping_colors),
                        brush_px=brush_px,
                        cell_step_px=cell_step_px,
                        remove_from_fallback=False
                    )
                    spiral_paths_by_color = [optimize_spiral_path_order(paths) for paths in spiral_paths_by_color]
                    self.log(
                        "Spiral Fill 蚊香填色已啟用："
                        f"螺旋 {spiral_stats['spiral_paths']} 筆，"
                        f"覆蓋 {spiral_stats['covered_pixels']} 格；"
                        "小細節自動保留長線備援"
                    )

                # 穩定填色：Palette / Custom RGB 都用密集水平線填滿。
                # 原本 Palette 的 contour/RDP 簡化在某些圖會把大色塊變得太稀，
                # 看起來像只畫到零碎色點；這裡改回可靠的 scanline。
                raw_runs_by_color, pixel_counts = build_color_runs(
                    fill_source_map,
                    len(mapping_colors),
                    bridge_gap=bridge_gap
                )

                optimized_runs_by_color = [optimize_run_order(runs) for runs in raw_runs_by_color]
                runs_by_color = optimized_runs_by_color

                exact_run_count = sum(len(runs) for runs in exact_runs_by_color)
                bridged_run_count = sum(len(runs) for runs in raw_runs_by_color)
                spiral_path_count = sum(len(paths) for paths in spiral_paths_by_color)
                original_run_air = sum(run_air_distance(runs) for runs in raw_runs_by_color)
                optimized_run_air = sum(run_air_distance(runs) for runs in optimized_runs_by_color)
                run_saved = 0.0 if original_run_air <= 0 else max(0.0, (original_run_air - optimized_run_air) / original_run_air * 100.0)

                if exact_run_count > 0:
                    if spiral_enabled:
                        total_after = bridged_run_count + spiral_path_count
                        contour_reduced = max(0.0, (exact_run_count - total_after) / exact_run_count * 100.0)
                        self.log(f"穩定填色 + 蚊香補線：runs {exact_run_count} -> 主填色 {bridged_run_count} + 補線 {spiral_path_count}")
                    else:
                        contour_reduced = max(0.0, (exact_run_count - bridged_run_count) / exact_run_count * 100.0)
                        self.log(f"穩定填色啟用：runs {exact_run_count} -> {bridged_run_count}")
                if bridge_gap > 0 and exact_run_count > 0:
                    reduced = max(0.0, (exact_run_count - bridged_run_count) / exact_run_count * 100.0)
                    self.log(f"長線填滿啟用：runs {exact_run_count} -> {bridged_run_count}，約減少 {reduced:.1f}%")
                self.log(f"色塊路徑排序完成：同色空移動距離 {original_run_air:.0f} -> {optimized_run_air:.0f}，約減少 {run_saved:.1f}%")

                eye_detail_runs_by_color = None
                eye_detail_offsets = None
                eye_detail_brush_key = 0 if brush_key == 0 else 1
                eye_detail_cell_step_px = 1
                eye_detail_brush_px = gartic_brush_pixels(eye_detail_brush_key)
                if eye_detail and mode in (MODE_PALETTE, MODE_CUSTOM_RGB):
                    eye_scale = min(100, max(line_scale, 95))
                    eye_max_w = max(1, int(target_w if using_placement else canvas_w * eye_scale / 100))
                    eye_max_h = max(1, int(target_h if using_placement else canvas_h * eye_scale / 100))
                    eye_map, eye_img_size = image_to_eye_detail_map(
                        self.image, mapping_colors, eye_max_w, eye_max_h
                    )
                    for wi in white_palette_indices(mapping_colors):
                        eye_map[eye_map == wi] = -1
                    if np.any(eye_map >= 0):
                        raw_eye_runs, _eye_counts = build_color_runs(eye_map, len(mapping_colors), bridge_gap=0)
                        eye_detail_runs_by_color = [optimize_run_order(runs) for runs in raw_eye_runs]
                        eye_w, eye_h = eye_img_size
                        eye_draw_px_w = (eye_w - 1) * eye_detail_cell_step_px + eye_detail_brush_px
                        eye_draw_px_h = (eye_h - 1) * eye_detail_cell_step_px + eye_detail_brush_px
                        eye_detail_offsets = (
                            (target_x + (target_w - eye_draw_px_w) / 2) if using_placement else (canvas_w - eye_draw_px_w) / 2,
                            (target_y + (target_h - eye_draw_px_h) / 2) if using_placement else (canvas_h - eye_draw_px_h) / 2,
                        )
                        eye_run_count = sum(len(runs) for runs in eye_detail_runs_by_color)
                        self.log(f"臉部細節強化已準備：{eye_run_count} 小段，最後補畫避免被底色吃掉")

                color_order = color_draw_order(mapping_colors)
                white_indices = white_palette_indices(mapping_colors) if skip_white and mode != MODE_CUSTOM_RGB else set()
                eye_order_for_progress = []
                if eye_detail_runs_by_color is not None:
                    eye_white_indices_for_progress = set(white_palette_indices(mapping_colors))
                    eye_order_for_progress = [idx for idx in color_draw_order(mapping_colors) if idx not in eye_white_indices_for_progress]

                total_expected_ops = 0
                for planned_idx in color_order:
                    if planned_idx in white_indices:
                        continue
                    total_expected_ops += len(spiral_paths_by_color[planned_idx]) + len(runs_by_color[planned_idx])
                if eye_detail_runs_by_color is not None:
                    total_expected_ops += sum(len(eye_detail_runs_by_color[idx]) for idx in eye_order_for_progress)
                total_expected_ops = max(1, total_expected_ops)

                self.signals.draw_phase_changed.emit("drawing")
                report_progress(0, total_expected_ops)
                self.wait_before_drawing()
                select_gartic_brush(brush_key, self.brush_positions, self.stop_event, focus_point=hotkey_focus_point)
                self.log(f"已切換 Gartic 筆刷：{brush_key}")

                total_ops = 0

                for color_idx in color_order:
                    if color_idx in white_indices:
                        continue
                    raise_if_stopped(self.stop_event)

                    runs = runs_by_color[color_idx]
                    spiral_paths = spiral_paths_by_color[color_idx] if 'spiral_paths_by_color' in locals() else []
                    if not runs and not spiral_paths:
                        continue

                    if mode == MODE_CUSTOM_RGB:
                        set_custom_rgb_color(mapping_colors[color_idx], self.custom_rgb_controls, self.stop_event, panel_delay=rgb_panel_delay)
                        self.wait_or_stop(PALETTE_SELECT_DELAY)
                    else:
                        px, py = palette_positions[color_idx]
                        pyautogui.click(px, py)
                        self.wait_or_stop(PALETTE_SELECT_DELAY)

                    for path in spiral_paths:
                        raise_if_stopped(self.stop_event)
                        screen_path = [
                            (
                                int(round(left + offset_px_x + px * cell_step_px + brush_px / 2)),
                                int(round(top + offset_px_y + py * cell_step_px + brush_px / 2))
                            )
                            for px, py in path
                        ]
                        draw_spiral_screen_path(screen_path, self.stop_event)
                        total_ops += 1
                        report_progress(total_ops, total_expected_ops)
                        if total_ops % 8 == 0:
                            self.wait_or_stop(max(delay, 0.001))

                    for run in runs:
                        raise_if_stopped(self.stop_event)
                        y, start, end, reverse = normalize_run(run)
                        draw_start = end - 1 if reverse else start
                        draw_end = start if reverse else end - 1
                        sx = int(round(left + offset_px_x + draw_start * cell_step_px + brush_px / 2))
                        ex = int(round(left + offset_px_x + draw_end * cell_step_px + brush_px / 2))
                        sy = int(round(top + offset_px_y + y * cell_step_px + brush_px / 2))

                        draw_color_run_solid(
                            sx, ex, sy,
                            brush_px=brush_px,
                            requested_drag=max(0.001, line_move_ms / 1000.0),
                            stop_event=self.stop_event
                        )
                        total_ops += 1
                        report_progress(total_ops, total_expected_ops)
                        if total_ops % 10 == 0:
                            self.wait_or_stop(max(delay, 0.001))

                if eye_detail_runs_by_color is not None and eye_detail_offsets is not None:
                    self.log("開始補畫臉部細節...")
                    select_gartic_brush(eye_detail_brush_key, self.brush_positions, self.stop_event, focus_point=hotkey_focus_point)
                    eye_offset_x, eye_offset_y = eye_detail_offsets
                    eye_order = eye_order_for_progress

                    for color_idx in eye_order:
                        raise_if_stopped(self.stop_event)
                        eye_runs = eye_detail_runs_by_color[color_idx]
                        if not eye_runs:
                            continue
                        if mode == MODE_CUSTOM_RGB:
                            set_custom_rgb_color(mapping_colors[color_idx], self.custom_rgb_controls, self.stop_event, panel_delay=rgb_panel_delay)
                            self.wait_or_stop(PALETTE_DETAIL_SELECT_DELAY)
                        else:
                            px, py = palette_positions[color_idx]
                            pyautogui.click(px, py)
                            self.wait_or_stop(PALETTE_DETAIL_SELECT_DELAY)

                        for run in eye_runs:
                            raise_if_stopped(self.stop_event)
                            y, start, end, reverse = normalize_run(run)
                            draw_start = end - 1 if reverse else start
                            draw_end = start if reverse else end - 1
                            sx = int(round(left + eye_offset_x + draw_start * eye_detail_cell_step_px + eye_detail_brush_px / 2))
                            ex = int(round(left + eye_offset_x + draw_end * eye_detail_cell_step_px + eye_detail_brush_px / 2))
                            sy = int(round(top + eye_offset_y + y * eye_detail_cell_step_px + eye_detail_brush_px / 2))

                            if end - start <= 1:
                                pyautogui.click(sx, sy)
                            else:
                                pyautogui.moveTo(sx, sy, duration=0)
                                pyautogui.dragTo(
                                    ex, sy,
                                    duration=max(0.001, min(0.008, abs(ex - sx) / 18000)),
                                    button="left"
                                )
                            total_ops += 1
                            report_progress(total_ops, total_expected_ops)
                            if total_ops % 25 == 0:
                                self.wait_or_stop(max(delay, 0.001))

                report_progress(total_expected_ops, total_expected_ops)
                self.log(f"全彩繪製完畢，總渲染區塊數：{total_ops}")

        except pyautogui.FailSafeException:
            self.log("🌟 [安全機制觸發] 已成功緊急停止繪製。")
        except StopDrawingException:
            self.log("已停止繪製。")
        except Exception as e:
            self.log(f"繪製過程中斷或出錯：{e}")
        finally:
            self.signals.drawing_changed.emit(False)


def main():
    app = QApplication([])
    app.setStyle("Fusion")
    window = GarticQtDrawer()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
