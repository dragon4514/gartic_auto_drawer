import threading
import time

import cv2
import mss
import numpy as np
from PIL import Image, ImageDraw
import pyautogui

from PySide6.QtCore import QObject, Qt, Signal, QTimer, QSize, QRectF
from PySide6.QtGui import QColor, QFont, QImage, QPixmap, QPainter, QPen, QLinearGradient, QBrush, QConicalGradient
from PySide6.QtWidgets import (
    QApplication, QAbstractButton, QAbstractSpinBox, QButtonGroup, QCheckBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFrame, QGraphicsDropShadowEffect, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QRadioButton, QScrollArea, QSpinBox, QTextEdit, QVBoxLayout, QWidget
)

try:
    from pynput import keyboard as pynput_keyboard
except Exception:
    pynput_keyboard = None

# 初始化 PyAutoGUI 安全設定
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.001  # 給予極小的安全暫停，確保 FailSafe 能被系統捕捉
pyautogui.MINIMUM_DURATION = 0
if hasattr(pyautogui, "MINIMUM_SLEEP"):
    pyautogui.MINIMUM_SLEEP = 0.001

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
DEFAULT_LINE_MOVE_MS = 10
DEFAULT_LINE_GAP_MS = 0
DEFAULT_LINE_SCALE = 85
DEFAULT_STROKE_STEP = 1
DEFAULT_CUSTOM_COLORS = 48
DEFAULT_SBR_STROKES = 300
PREVIEW_MAX_SIZE = 620
GARTIC_BRUSH_PIXELS = {
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


class StopDrawingException(Exception):
    pass


def capture_screen_rgb():
    with mss.mss() as sct:
        monitor = sct.monitors[0]
        shot = np.array(sct.grab(monitor))
        img_rgb = cv2.cvtColor(shot, cv2.COLOR_BGRA2RGB)
        return img_rgb, monitor["left"], monitor["top"]


def detect_canvas(img_rgb):
    # 找大片白色畫布
    mask = cv2.inRange(
        img_rgb,
        np.array([245, 245, 245], dtype=np.uint8),
        np.array([255, 255, 255], dtype=np.uint8)
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        ratio = w / max(h, 1)

        if area > 100000 and 1.2 < ratio < 2.5:
            if area > best_area:
                best_area = area
                best = (x, y, x + w, y + h)

    return best


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
    return int(np.clip(int(value), 1, 5))


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


def detect_palette(img_rgb, canvas):
    x1, y1, x2, y2 = canvas
    h, w, _ = img_rgb.shape

    rx1 = max(0, x1 - 250)
    rx2 = max(0, x1 - 25)
    ry1 = max(0, y1 + 45)
    ry2 = min(h, y1 + 500)

    crop = img_rgb[ry1:ry2, rx1:rx2]

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 130)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []

    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)

        if 26 <= ww <= 76 and 26 <= hh <= 76 and abs(ww - hh) <= 18:
            cx = rx1 + x + ww // 2
            cy = ry1 + y + hh // 2

            duplicate = False
            for px, py in centers:
                if abs(px - cx) < 16 and abs(py - cy) < 16:
                    duplicate = True
                    break

            if not duplicate:
                centers.append((cx, cy))

    centers = sort_centers_grid(centers)

    if len(centers) >= 18:
        centers = centers[:18]
    else:
        # Fixed 3 x 6 palette layout relative to the white drawing canvas.
        xs = [x1 - 205, x1 - 140, x1 - 75]
        ys = [y1 + 95 + i * 67 for i in range(6)]
        centers = [(x, y) for y in ys for x in xs]

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

    rx1 = max(0, x1 - 40)
    rx2 = min(w, x1 + min(canvas_w, 620))
    ry1 = min(h, max(0, y2 + 25))
    ry2 = min(h, y2 + 220)

    centers = []

    if ry2 > ry1 + 40 and rx2 > rx1 + 120:
        crop = img_rgb[ry1:ry2, rx1:rx2]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(28, int(canvas_w * 0.055)),
            param1=80,
            param2=18,
            minRadius=max(10, int(canvas_w * 0.018)),
            maxRadius=max(28, int(canvas_w * 0.055))
        )

        if circles is not None:
            candidates = []
            for cx, cy, radius in np.round(circles[0]).astype(int):
                gx = rx1 + cx
                gy = ry1 + cy
                if x1 - 20 <= gx <= x1 + 560 and y2 + 20 <= gy <= y2 + 210:
                    candidates.append((gx, gy, radius))

            groups = []
            for candidate in sorted(candidates, key=lambda c: c[1]):
                added = False
                for group in groups:
                    if abs(group[0][1] - candidate[1]) <= 24:
                        group.append(candidate)
                        added = True
                        break
                if not added:
                    groups.append([candidate])

            groups = [sorted(group, key=lambda c: c[0]) for group in groups]
            groups = sorted(groups, key=lambda group: (-len(group), group[0][0]))

            for group in groups:
                if len(group) >= 5:
                    centers = [(gx, gy) for gx, gy, _radius in group[:5]]
                    break

    if len(centers) < 5:
        # Fallback for the current Gartic layout: five brush circles below the canvas.
        spacing = int(np.clip(canvas_w * 0.068, 32, 95))
        start_x = int(x1 + canvas_w * 0.043)
        brush_y = int(min(h - 1, y2 + canvas_h * 0.15))
        centers = [(start_x + i * spacing, brush_y) for i in range(5)]

    return centers[:5]


def estimate_custom_rgb_controls(palette):
    if len(palette) < 18:
        return None

    xs = [palette[i]["pos"][0] for i in range(3)]
    ys = [palette[i * 3]["pos"][1] for i in range(6)]
    col_gap = max(1, float(np.median(np.diff(sorted(xs)))))
    row_gap = max(1, float(np.median(np.diff(sorted(ys)))))
    center_x = float(np.mean(xs))
    last_y = float(ys[-1])

    swatch = (
        int(round(center_x)),
        int(round(last_y + row_gap * 1.20))
    )
    input_y = int(round(swatch[1] + row_gap * 3.50))
    inputs = [
        (int(round(center_x - col_gap * 0.60)), input_y),
        (int(round(center_x + col_gap * 0.50)), input_y),
        (int(round(center_x + col_gap * 1.60)), input_y),
    ]

    return {
        "swatch": swatch,
        "inputs": inputs,
    }


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


def image_to_palette_map(img_rgba, palette_colors, max_w, max_h, skip_white=True):
    img = resize_keep_aspect(img_rgba, max_w, max_h)
    rgb, alpha = pil_rgb_alpha_arrays(img, dtype=np.int32)

    pal = np.asarray(palette_colors, dtype=np.int32)

    diff = rgb[:, :, None, :] - pal[None, None, :, :]
    dist = np.sum(diff * diff, axis=3)
    idx = np.argmin(dist, axis=2).astype(np.int16)

    skip = alpha < 40

    if skip_white:
        # 原圖接近白色的地方直接跳過，利用 Gartic 原本白色畫布。
        near_white = (
            np.min(rgb, axis=2) > 230
        ) & (
            np.max(rgb, axis=2) - np.min(rgb, axis=2) < 35
        )
        skip = skip | near_white

        # 即使像素被量化成 Gartic 白色色盤，也不要畫。
        white_indices = white_palette_indices(palette_colors)
        if white_indices:
            idx[np.isin(idx, list(white_indices))] = -1

    idx[skip] = -1

    return idx, img.size



def detect_eye_detail_mask(rgb, alpha):
    """
    Detect compact anime-eye details only.
    This is intentionally conservative: it looks for small red/brown/dark
    components in the upper-center character area, then adds tiny highlights
    around those components.  It avoids changing broad background handling.
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
    roi = (
        (alpha >= 40)
        & (yy >= int(h * 0.14))
        & (yy <= int(h * 0.66))
        & (xx >= int(w * 0.10))
        & (xx <= int(w * 0.90))
    )

    # Red / brown / pink-ish eye pixels, plus very dark pupils/eyelashes.
    red_channel = rgb[:, :, 0].astype(np.int16)
    green_channel = rgb[:, :, 1].astype(np.int16)
    blue_channel = rgb[:, :, 2].astype(np.int16)
    reddish = (
        (red_channel > green_channel + 12)
        & (red_channel > blue_channel + 6)
        & (sat > 25)
        & (val < 245)
    )
    hue_red_or_brown = ((hue <= 18) | (hue >= 165) | ((hue >= 6) & (hue <= 32))) & (sat > 28) & (val < 245)
    dark_pupil = (gray < 115) & (alpha >= 40)

    candidate = roi & (reddish | hue_red_or_brown | dark_pupil)
    candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, 8)
    keep = np.zeros((h, w), dtype=np.uint8)
    max_area = max(6, int(h * w * 0.010))

    for label in range(1, num_labels):
        x, y, ww, hh, area = stats[label]
        cx, cy = centroids[label]

        if area < 2:
            continue
        if area > max_area:
            continue
        if ww > w * 0.32 or hh > h * 0.18:
            continue
        if cy < h * 0.16 or cy > h * 0.66:
            continue

        keep[labels == label] = 1

    if not np.any(keep):
        return keep.astype(bool)

    # Add tiny eye highlights / eyelid pixels close to detected eye blobs.
    near_eye = cv2.dilate(keep, np.ones((5, 5), np.uint8), iterations=1) > 0
    small_highlight = (
        roi
        & near_eye
        & (np.min(rgb, axis=2) > 205)
        & ((np.max(rgb, axis=2) - np.min(rgb, axis=2)) < 65)
    )
    nearby_colored = roi & near_eye & (sat > 25) & (val < 250)
    nearby_dark = roi & near_eye & (gray < 150)

    detail = (keep > 0) | small_highlight | nearby_colored | nearby_dark
    detail = cv2.morphologyEx(detail.astype(np.uint8), cv2.MORPH_OPEN, np.ones((1, 1), np.uint8))
    return detail.astype(bool)


def image_to_eye_detail_map(img_rgba, palette_colors, max_w, max_h):
    """
    High-resolution final pass just for eyes.
    Returns a color_map where non-eye pixels are -1.  Unlike global Skip White,
    white eye highlights are allowed so they can be restored after red/dark fills.
    """
    img = resize_keep_aspect(img_rgba, max_w, max_h)
    rgb, alpha = pil_rgb_alpha_arrays(img, dtype=np.uint8)

    if not palette_colors:
        return np.full(rgb.shape[:2], -1, dtype=np.int16), img.size

    mask = detect_eye_detail_mask(rgb, alpha)
    if not np.any(mask):
        return np.full(rgb.shape[:2], -1, dtype=np.int16), img.size

    colors = np.asarray(palette_colors, dtype=np.int32)
    rgb_i = rgb.astype(np.int32)
    diff = rgb_i[:, :, None, :] - colors[None, None, :, :]
    dist = np.sum(diff * diff, axis=3)
    idx = np.argmin(dist, axis=2).astype(np.int16)
    idx[~mask] = -1
    idx[alpha < 40] = -1
    return idx, img.size

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


def image_to_custom_rgb_map(img_rgba, max_w, max_h, color_count=24, skip_white=True):
    """
    高還原 Custom RGB：
    - 只移除邊界純白背景，不吃掉臉/衣服高光。
    - 48 色時不做 label medianBlur，避免眼睛、嘴巴、細線被抹掉。
    - 小色塊清理改很保守，保留動漫圖細節。
    """
    img = resize_keep_aspect(img_rgba, max_w, max_h)
    rgb, alpha = pil_rgb_alpha_arrays(img, dtype=np.uint8)

    # 輕微降噪即可；原本太強會把眼睛/臉部小色塊糊掉。
    rgb_smooth = cv2.bilateralFilter(rgb, 5, 28, 28)

    skip = alpha < 40

    if skip_white:
        skip = skip | background_white_mask(rgb_smooth, alpha)

    # 深色線條與五官不准被 skip_white 影響。
    gray = cv2.cvtColor(rgb_smooth, cv2.COLOR_RGB2GRAY)
    dark_detail = (gray < 170) & (alpha >= 40)
    skip[dark_detail] = False

    pixels = rgb_smooth[~skip]

    if len(pixels) == 0:
        return np.full(rgb.shape[:2], -1, dtype=np.int16), [], img.size

    color_count = int(np.clip(color_count, 2, 64))
    k = min(color_count, len(pixels))

    # 用隨機但固定 seed 的 sample，比 linspace 更不容易偏向圖片某一側。
    rng = np.random.default_rng(12345)
    if len(pixels) > 50000:
        sample_idx = rng.choice(len(pixels), 50000, replace=False)
        samples = pixels[sample_idx]
    else:
        samples = pixels

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
    rgb_i = rgb_smooth.astype(np.int32)
    diff = rgb_i[:, :, None, :] - centers_i[None, None, :, :]
    dist = np.sum(diff * diff, axis=3)
    idx = np.argmin(dist, axis=2).astype(np.int16)
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
    if color_count >= 40:
        min_area = 2
    elif color_count >= 32:
        min_area = 3
    elif color_count >= 24:
        min_area = 4
    else:
        min_area = max(4, int(max_w * max_h * 0.00018))

    idx = remove_small_color_regions(idx, k, min_area)

    colors = [tuple(int(v) for v in color) for color in centers_i]
    return idx, colors, img.size

def remove_small_color_regions(color_map, palette_size, min_area):
    cleaned = np.full(color_map.shape, -1, dtype=np.int16)

    for color_idx in range(palette_size):
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

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(invalid, 8)
    max_hole_area = max(1, int(max_hole_area))
    kernel = np.ones((3, 3), np.uint8)

    for label in range(1, num_labels):
        x, y, ww, hh, area = stats[label]

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


def image_to_cartoon_color_map(img_rgba, palette_colors, max_w, max_h, skip_white=True):
    img = resize_keep_aspect(img_rgba, max_w, max_h)
    rgb, alpha = pil_rgb_alpha_arrays(img, dtype=np.uint8)
    rgb = cv2.bilateralFilter(rgb, 7, 45, 45)
    pal = np.asarray(palette_colors, dtype=np.int32)
    rgb_i = rgb.astype(np.int32)
    diff = rgb_i[:, :, None, :] - pal[None, None, :, :]
    dist = np.sum(diff * diff, axis=3)
    idx = np.argmin(dist, axis=2).astype(np.int16)

    skip = alpha < 40

    if skip_white:
        # 原圖白色 / 接近白色背景直接跳過。
        near_white = np.all(rgb > 246, axis=2)
        almost_white = (
            np.min(rgb, axis=2) > 225
        ) & (
            np.max(rgb, axis=2) - np.min(rgb, axis=2) < 40
        )
        skip = skip | near_white | almost_white

        # 被轉換成 Gartic 白色色盤的，也直接跳過。
        white_indices = white_palette_indices(palette_colors)
        if white_indices:
            idx[np.isin(idx, list(white_indices))] = -1

    idx[skip] = -1

    min_area = max(4, int(max_w * max_h * 0.00045))
    idx = remove_small_color_regions(idx, len(palette_colors), min_area)

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

    for _iteration in range(stroke_count):
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
    img = (mask > 0).astype(np.uint8)
    changed = True

    while changed:
        changed = False

        for step in (0, 1):
            remove = []
            h, w = img.shape

            for y in range(1, h - 1):
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
        for nxt in neighbors(node):
            if edge_key(node, nxt) in visited_edges:
                continue
            path = follow_path(node, nxt)
            if len(path) >= min_points:
                strokes.append([(x, y) for y, x in path])

    for point in points:
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

    while remaining:
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

    Brush Key 1 in Gartic is visually thinner than our estimated brush_px,
    so using brush_px * 0.72 leaves horizontal white gaps.  For the smallest
    brush, use a 1 px cell step.  Larger brushes still use overlapping rows
    to keep fills solid while avoiding too many operations.
    """
    brush_px = max(1, int(brush_px))

    if mode in (MODE_PALETTE, MODE_CUSTOM_RGB):
        if brush_px <= 3:
            return 1
        if brush_px <= 6:
            return max(1, int(round(brush_px * 0.50)))
        return max(1, int(round(brush_px * 0.62)))

    return brush_px


def color_hole_area(mode, brush_px):
    if mode in (MODE_PALETTE, MODE_CUSTOM_RGB):
        return max(4, int((max(1, brush_px) ** 2) * 1.5))

    return 0


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
    preview = Image.new("RGB", (preview_w, preview_h), "white")
    draw = ImageDraw.Draw(preview)

    if use_contour:
        runs_by_color, _pixel_counts = build_color_runs_contour(
            color_map,
            len(palette_colors),
            brush_px=brush_px,
            bridge_gap=bridge_gap,
            optimize=False
        )
    else:
        runs_by_color, _pixel_counts = build_color_runs(color_map, len(palette_colors), bridge_gap=bridge_gap)
    # 預覽也用實際繪製順序：亮色先、暗色最後，避免線條被蓋掉。
    color_order = color_draw_order(palette_colors)

    radius = brush_px / 2

    for color_idx in color_order:
        color = tuple(int(v) for v in palette_colors[color_idx])

        if spiral_paths_by_color is not None and color_idx < len(spiral_paths_by_color):
            for path in spiral_paths_by_color[color_idx]:
                if len(path) == 1:
                    px, py = path[0]
                    cx = px * cell_step_px + radius
                    cy = py * cell_step_px + radius
                    draw.ellipse(
                        (cx - radius, cy - radius, cx + radius, cy + radius),
                        fill=color
                    )
                    continue

                if len(path) >= 2:
                    points = [
                        (
                            px * cell_step_px + radius,
                            py * cell_step_px + radius
                        )
                        for px, py in path
                    ]
                    draw.line(points, fill=color, width=brush_px)
                    for px, py in (path[0], path[-1]):
                        cx = px * cell_step_px + radius
                        cy = py * cell_step_px + radius
                        draw.ellipse(
                            (cx - radius, cy - radius, cx + radius, cy + radius),
                            fill=color
                        )

        for run in runs_by_color[color_idx]:
            y, start, end, _reverse = normalize_run(run)
            sx = start * cell_step_px + radius
            ex = (end - 1) * cell_step_px + radius
            sy = y * cell_step_px + radius

            if end - start <= 1:
                draw.ellipse(
                    (sx - radius, sy - radius, sx + radius, sy + radius),
                    fill=color
                )
            else:
                draw.line((sx, sy, ex, sy), fill=color, width=brush_px)
                draw.ellipse(
                    (sx - radius, sy - radius, sx + radius, sy + radius),
                    fill=color
                )
                draw.ellipse(
                    (ex - radius, sy - radius, ex + radius, sy + radius),
                    fill=color
                )

    return preview


def compose_canvas_preview(content, canvas_w, canvas_h):
    canvas_w = max(1, int(canvas_w))
    canvas_h = max(1, int(canvas_h))
    preview = Image.new("RGB", (canvas_w, canvas_h), "white")
    x = (canvas_w - content.size[0]) // 2
    y = (canvas_h - content.size[1]) // 2
    preview.paste(content, (x, y))
    return preview


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


def screen_point_px(left, top, offset_x, offset_y, point):
    x, y = point
    return (
        int(round(left + offset_x + x)),
        int(round(top + offset_y + y))
    )


def raise_if_stopped(stop_event):
    if stop_event is not None and stop_event.is_set():
        raise StopDrawingException()


def select_gartic_brush(brush_key, brush_positions=None, stop_event=None):
    raise_if_stopped(stop_event)
    key = clamp_brush_key(brush_key)

    if brush_positions and len(brush_positions) >= key:
        x, y = brush_positions[key - 1]
        pyautogui.click(x, y)
        time.sleep(0.20)
    else:
        pyautogui.press(str(key))
        time.sleep(0.30)

    raise_if_stopped(stop_event)


def set_custom_rgb_color(rgb, controls, stop_event=None):
    raise_if_stopped(stop_event)

    if not controls:
        raise RuntimeError("尚未取得 RGB 面板位置，請先 Auto Detect，並確認自訂色面板位置沒有被遮擋。")

    pyautogui.click(*controls["swatch"])
    time.sleep(0.18)

    for pos, value in zip(controls["inputs"], rgb):
        raise_if_stopped(stop_event)
        pyautogui.click(*pos)
        time.sleep(0.03)
        pyautogui.hotkey("ctrl", "a")
        pyautogui.write(str(int(np.clip(value, 0, 255))), interval=0)
        time.sleep(0.04)

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
    runs_by_color = [[] for _ in range(palette_size)]
    pixel_counts = [0 for _ in range(palette_size)]
    draw_h, draw_w = color_map.shape
    bridge_gap = max(0, int(bridge_gap))

    for y in range(draw_h):
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
    mask = (mask > 0).astype(np.uint8)
    runs = []
    h, w = mask.shape
    bridge_gap = max(0, int(bridge_gap))
    min_run_len = max(1, int(min_run_len))

    for y in range(h):
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

        pixel_counts[color_idx] = int(np.sum(mask))
        simplified = contour_simplify_mask(mask, min_area=min_area, epsilon_factor=epsilon_factor)
        runs = extract_runs_from_binary_mask(
            simplified,
            bridge_gap=bridge_gap,
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

    for depth in range(int(max_loops)):
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


def build_spiral_fill_paths(color_map, palette_size, brush_px=3, cell_step_px=1, min_area=None):
    """
    Build spiral fill paths for large simple same-color components.
    Returns (spiral_paths_by_color, fallback_color_map, stats). Small / complex
    regions stay in fallback_color_map and use the original optimized scanline.
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

    for color_idx in range(palette_size):
        mask = (color_map == color_idx).astype(np.uint8)
        if not np.any(mask):
            continue
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
        for label in range(1, num_labels):
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
    paths = [p for p in paths if len(p) >= 2]
    if len(paths) <= 2:
        return paths
    remaining = list(range(len(paths)))
    start_pos = max(range(len(remaining)), key=lambda pos: len(paths[remaining[pos]]))
    first_idx = remaining.pop(start_pos)
    ordered = [paths[first_idx]]
    current_end = ordered[-1][-1]
    while remaining:
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

    while remaining:
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
        if self.isChecked():
            track = QColor("#38BDF8")
            border = QColor("#60A5FA")
            knob = QColor("#EFF6FF")
            knob_x = 23.0
        else:
            track = QColor("#111827")
            border = QColor("#334155")
            knob = QColor("#64748B")
            knob_x = 3.0

        if self.underMouse():
            border = QColor("#7DD3FC")
            if not self.isChecked():
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

    def mousePressEvent(self, event):
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



class UiSignals(QObject):
    log_message = Signal(str)
    detect_status = Signal(str, str)
    detecting_changed = Signal(bool)
    drawing_changed = Signal(bool)
    draw_phase_changed = Signal(str)
    previewing_changed = Signal(bool)
    preview_ready = Signal(object, str)
    stop_requested = Signal()


def pil_to_qpixmap(img):
    """Convert PIL image to QPixmap safely."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = fit_preview_image(img)
    w, h = img.size
    data = img.tobytes("raw", "RGB")
    qimg = QImage(data, w, h, w * 3, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)



class AnimatedActionButton(QPushButton):
    """Three-state action button with a soft animated border for the computing stage."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._phase = "idle"
        self._pulse = 0
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(44)
        self.setFont(QFont("Segoe UI", 11, QFont.Bold))

    def setPhase(self, phase):
        self._phase = str(phase or "idle")
        self.update()

    def setPulse(self, pulse):
        self._pulse = int(pulse) % 360
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
            fill = QLinearGradient(rect.topLeft(), rect.topRight())
            fill.setColorAt(0.0, QColor("#22C55E"))
            fill.setColorAt(0.55, QColor("#16A34A"))
            fill.setColorAt(1.0, QColor("#0D9488"))
            text_color = QColor("#FFFFFF")
            painter.setPen(QPen(QColor("#86EFAC"), 1.4))
            painter.setBrush(fill)
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
        self.signals.previewing_changed.connect(self.set_previewing)
        self.signals.preview_ready.connect(self.show_preview)
        self.signals.stop_requested.connect(self.request_stop)

        self.build_ui()
        self.apply_modern_theme()
        self.start_global_hotkey()
        self.log("=== Qt 右側 Log / 動態運算邊框版就緒 ===")
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

        quick_row.addWidget(QLabel("Brush Key"))
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, 5)
        self.brush_spin.setValue(1)
        quick_row.addWidget(self.brush_spin)

        self.detect_label = QLabel("尚未偵測")
        self.detect_label.setObjectName("statusPill")
        quick_row.addWidget(self.detect_label, 1)
        quick_layout.addLayout(quick_row)
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

        mode_layout.addWidget(QLabel("Line Detail"), 1, 0)
        self.detail_spin = QSpinBox()
        self.detail_spin.setRange(1, 5)
        self.detail_spin.setValue(4)
        mode_layout.addWidget(self.detail_spin, 1, 1)

        mode_layout.addWidget(QLabel("Custom Colors"), 1, 2)
        self.custom_colors_spin = QSpinBox()
        self.custom_colors_spin.setRange(8, 48)
        self.custom_colors_spin.setSingleStep(4)
        self.custom_colors_spin.setValue(DEFAULT_CUSTOM_COLORS)
        mode_layout.addWidget(self.custom_colors_spin, 1, 3)

        mode_layout.addWidget(QLabel("Line Move ms"), 2, 0)
        self.line_move_spin = QSpinBox()
        self.line_move_spin.setRange(1, 80)
        self.line_move_spin.setValue(DEFAULT_LINE_MOVE_MS)
        mode_layout.addWidget(self.line_move_spin, 2, 1)

        mode_layout.addWidget(QLabel("SBR Strokes"), 2, 2)
        self.sbr_strokes_spin = QSpinBox()
        self.sbr_strokes_spin.setRange(50, 1500)
        self.sbr_strokes_spin.setSingleStep(50)
        self.sbr_strokes_spin.setValue(DEFAULT_SBR_STROKES)
        mode_layout.addWidget(self.sbr_strokes_spin, 2, 3)

        mode_layout.addWidget(QLabel("Line Gap ms"), 3, 0)
        self.line_gap_spin = QSpinBox()
        self.line_gap_spin.setRange(0, 80)
        self.line_gap_spin.setValue(DEFAULT_LINE_GAP_MS)
        mode_layout.addWidget(self.line_gap_spin, 3, 1)

        mode_layout.addWidget(QLabel("Image Scale %"), 3, 2)
        self.line_scale_spin = QSpinBox()
        self.line_scale_spin.setRange(40, 100)
        self.line_scale_spin.setSingleStep(5)
        self.line_scale_spin.setValue(DEFAULT_LINE_SCALE)
        mode_layout.addWidget(self.line_scale_spin, 3, 3)

        mode_layout.addWidget(QLabel("Stroke Step"), 4, 0)
        self.stroke_step_spin = QSpinBox()
        self.stroke_step_spin.setRange(1, 5)
        self.stroke_step_spin.setValue(DEFAULT_STROKE_STEP)
        mode_layout.addWidget(self.stroke_step_spin, 4, 1)
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
            self.draw_btn.setText("◌  運算中 Computing...")
            if not self.action_anim_timer.isActive():
                self.action_anim_timer.start()
        elif phase == "drawing":
            self.draw_btn.setText("✦  繪製中 Drawing...")
            if not self.action_anim_timer.isActive():
                self.action_anim_timer.start()
        else:
            self.action_anim_timer.stop()
            self.action_pulse = 0
            if hasattr(self.draw_btn, "setPulse"):
                self.draw_btn.setPulse(0)
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

            for p in palette:
                px, py = p["pos"]
                p["pos"] = (int(px + offset_x), int(py + offset_y))

            brush_positions = [(int(px + offset_x), int(py + offset_y)) for px, py in brush_positions]

            self.canvas = canvas_screen
            self.palette = palette
            self.brush_positions = brush_positions
            self.custom_rgb_controls = estimate_custom_rgb_controls(self.palette)

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
        except Exception as e:
            self.log(f"偵測過程中發生異常：{e}")
        finally:
            self.signals.detecting_changed.emit(False)

    def _collect_preview_params(self):
        brush_key = clamp_brush_key(self.brush_spin.value())
        skip_white = self.skip_white_check.isChecked()
        mode = self.mode_value
        detail = int(self.detail_spin.value())
        line_scale = min(100, max(40, int(self.line_scale_spin.value())))
        custom_colors = int(np.clip(int(self.custom_colors_spin.value()), 8, 48))
        sbr_strokes = int(np.clip(int(self.sbr_strokes_spin.value()), 50, 1500))
        eye_detail = self.eye_detail_check.isChecked()
        spiral_fill = self.spiral_fill_check.isChecked()
        return brush_key, skip_white, mode, detail, line_scale, custom_colors, sbr_strokes, eye_detail, spiral_fill

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

    def build_preview(self, brush_key, skip_white, mode, detail, line_scale, custom_colors, sbr_strokes, eye_detail, spiral_fill):
        try:
            brush_px = gartic_brush_pixels(brush_key)
            x1, y1, x2, y2 = self.canvas
            canvas_w, canvas_h = abs(x2 - x1), abs(y2 - y1)

            if mode == MODE_SBR:
                line_max_w = max(1, int(canvas_w * line_scale / 100))
                line_max_h = max(1, int(canvas_h * line_scale / 100))
                detected_colors = [p["color"] for p in self.palette]
                mapping_colors = palette_colors_for_mapping(detected_colors)
                strokes, img_size, content = image_to_sbr_strokes(
                    self.image, line_max_w, line_max_h, mapping_colors,
                    stroke_count=sbr_strokes, brush_px=brush_px, skip_white=skip_white
                )
                preview = compose_canvas_preview(content, canvas_w, canvas_h)
                info = f"SBR 筆觸預覽 | 畫布: {canvas_w}x{canvas_h} | 圖像: {img_size[0]}x{img_size[1]} | 筆觸: {len(strokes)} | 筆刷鍵: {brush_key}"
            elif mode in (MODE_SMART_LINE, MODE_LINE, MODE_CLEAN_LINE, MODE_DARK_OUTLINE):
                line_max_w = max(1, int(canvas_w * line_scale / 100))
                line_max_h = max(1, int(canvas_h * line_scale / 100))
                strokes, img_size = image_to_smart_line_strokes(self.image, line_max_w, line_max_h, detail=detail)
                content = line_strokes_to_image(strokes, img_size, line_width=brush_px)
                preview = compose_canvas_preview(content, canvas_w, canvas_h)
                info = f"整合線稿預覽 | 畫布: {canvas_w}x{canvas_h} | 圖像: {img_size[0]}x{img_size[1]} | 筆刷鍵: {brush_key} | 總筆畫: {len(strokes)}"
            elif mode in (MODE_PALETTE, MODE_CUSTOM_RGB):
                color_max_w = max(1, int(canvas_w * line_scale / 100))
                color_max_h = max(1, int(canvas_h * line_scale / 100))
                cell_step_px = color_fill_step(mode, brush_px)
                cells_w = max(1, color_max_w // cell_step_px)
                cells_h = max(1, color_max_h // cell_step_px)
                detected_colors = [p["color"] for p in self.palette]
                mapping_colors = palette_colors_for_mapping(detected_colors)

                if mode != MODE_CUSTOM_RGB and not detected_colors:
                    self.log("錯誤：未偵測到色盤顏色，無法生成全彩預覽。")
                    return

                if mode == MODE_CUSTOM_RGB:
                    color_map, mapping_colors, img_size = image_to_custom_rgb_map(
                        self.image, cells_w, cells_h, color_count=custom_colors, skip_white=skip_white
                    )
                    label = "自訂 RGB 預覽"
                else:
                    color_map, img_size = image_to_palette_map(
                        self.image, mapping_colors, cells_w, cells_h, skip_white=skip_white
                    )
                    label = "全彩色盤預覽"

                if eye_detail and mode in (MODE_PALETTE, MODE_CUSTOM_RGB):
                    eye_map, _eye_size = image_to_eye_detail_map(
                        self.image, mapping_colors, color_map.shape[1], color_map.shape[0]
                    )
                    for wi in white_palette_indices(mapping_colors):
                        eye_map[eye_map == wi] = -1
                    eye_mask = eye_map >= 0
                    if np.any(eye_mask):
                        color_map[eye_mask] = eye_map[eye_mask]

                hole_area = color_hole_area(mode, brush_px)
                if hole_area > 0:
                    color_map = solidify_color_map(color_map, len(mapping_colors), hole_area)

                preview_color_map = color_map
                spiral_paths_by_color = None
                spiral_count = 0
                spiral_enabled = bool(spiral_fill and mode in (MODE_PALETTE, MODE_CUSTOM_RGB))

                if spiral_enabled:
                    spiral_paths_by_color, preview_color_map, _spiral_stats = build_spiral_fill_paths(
                        color_map,
                        len(mapping_colors),
                        brush_px=brush_px,
                        cell_step_px=cell_step_px
                    )
                    spiral_paths_by_color = [optimize_spiral_path_order(paths) for paths in spiral_paths_by_color]
                    spiral_count = sum(len(paths) for paths in spiral_paths_by_color)

                content = color_map_to_gartic_preview(
                    preview_color_map, mapping_colors, brush_px,
                    reverse_order=(mode == MODE_CUSTOM_RGB),
                    bridge_gap=color_run_bridge_gap(mode, brush_px),
                    use_contour=(mode != MODE_CUSTOM_RGB),
                    cell_step_px=cell_step_px,
                    spiral_paths_by_color=spiral_paths_by_color
                )
                preview = compose_canvas_preview(content, canvas_w, canvas_h)
                spiral_state = f"ON/{spiral_count}" if spiral_enabled else "OFF"
                info = f"{label} | 畫布: {canvas_w}x{canvas_h} | 圖像: {content.size[0]}x{content.size[1]} | 網格: {img_size[0]}x{img_size[1]} | 筆刷鍵: {brush_key} | 格距: {cell_step_px} | Spiral: {spiral_state}"
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
        custom_colors = int(np.clip(int(self.custom_colors_spin.value()), 8, 48))
        sbr_strokes = int(np.clip(int(self.sbr_strokes_spin.value()), 50, 1500))
        auto_black = self.auto_black_check.isChecked()
        eye_detail = self.eye_detail_check.isChecked()
        spiral_fill = self.spiral_fill_check.isChecked()

        self.set_drawing(True)
        self.stop_event.clear()
        threading.Thread(
            target=self.draw_fast,
            args=(cps, brush_key, skip_white, mode, detail, line_move_ms, line_gap_ms, line_scale, stroke_step, custom_colors, sbr_strokes, auto_black, eye_detail, spiral_fill),
            daemon=True
        ).start()

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

    def draw_fast(self, cps, brush_key, skip_white, mode, detail, line_move_ms, line_gap_ms, line_scale, stroke_step, custom_colors, sbr_strokes, auto_black, eye_detail, spiral_fill):
        try:
            delay = 1.0 / cps
            brush_px = gartic_brush_pixels(brush_key)
            x1, y1, x2, y2 = self.canvas
            left, top = min(x1, x2), min(y1, y2)
            canvas_w, canvas_h = abs(x2 - x1), abs(y2 - y1)

            palette_positions = [p["pos"] for p in self.palette]
            palette_colors = [p["color"] for p in self.palette]
            mapping_colors = palette_colors_for_mapping(palette_colors)

            if mode == MODE_SBR:
                if not palette_positions:
                    self.log("錯誤：未偵測到有效色盤，取消 SBR 繪製。")
                    return
                line_max_w = max(1, int(canvas_w * line_scale / 100))
                line_max_h = max(1, int(canvas_h * line_scale / 100))
                strokes, img_size, _preview = image_to_sbr_strokes(
                    self.image, line_max_w, line_max_h, mapping_colors,
                    stroke_count=sbr_strokes, brush_px=brush_px, skip_white=skip_white
                )
                draw_w, draw_h = img_size
                offset_x = (canvas_w - draw_w) / 2
                offset_y = (canvas_h - draw_h) / 2
                self.log(f"SBR 筆觸序列計算完畢，總筆觸：{len(strokes)}")
                self.signals.draw_phase_changed.emit("drawing")
                self.wait_before_drawing()
                select_gartic_brush(brush_key, self.brush_positions, self.stop_event)
                self.log(f"已切換 Gartic 筆刷：{brush_key}")

                current_color = None
                total_ops = 0
                for color_idx, start, end, width in strokes:
                    raise_if_stopped(self.stop_event)
                    if color_idx != current_color:
                        px, py = palette_positions[color_idx]
                        pyautogui.click(px, py)
                        self.wait_or_stop(0.05)
                        current_color = color_idx
                    start_pt = screen_point_px(left, top, offset_x, offset_y, start)
                    end_pt = screen_point_px(left, top, offset_x, offset_y, end)
                    draw_stroke_path([start_pt, end_pt], max(0.001, line_move_ms / 1000.0), self.stop_event)
                    total_ops += 1
                    if total_ops % 10 == 0:
                        self.wait_or_stop(max(delay, 0.001))
                self.log(f"SBR 繪製完畢，總筆觸：{total_ops}")

            elif mode in (MODE_SMART_LINE, MODE_LINE, MODE_CLEAN_LINE, MODE_DARK_OUTLINE):
                line_max_w = max(1, int(canvas_w * line_scale / 100))
                line_max_h = max(1, int(canvas_h * line_scale / 100))
                strokes, img_size = image_to_smart_line_strokes(self.image, line_max_w, line_max_h, detail=detail)
                before_air = stroke_air_distance(strokes)
                strokes = optimize_stroke_order(strokes)
                after_air = stroke_air_distance(strokes)
                draw_w, draw_h = img_size
                offset_x = (canvas_w - draw_w) / 2
                offset_y = (canvas_h - draw_h) / 2
                saved = 0.0 if before_air <= 0 else max(0.0, (before_air - after_air) / before_air * 100.0)
                self.log(f"整合線稿序列計算完畢，總筆畫：{len(strokes)}")
                self.log(f"TSP 路徑排序完成：空移動距離約減少 {saved:.1f}%")
                self.signals.draw_phase_changed.emit("drawing")
                self.wait_before_drawing()
                select_gartic_brush(brush_key, self.brush_positions, self.stop_event)
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
                    if line_gap_ms > 0:
                        self.wait_or_stop(line_gap_ms / 1000.0)
                    else:
                        raise_if_stopped(self.stop_event)
                self.log(f"線稿繪製完畢，總操作筆畫：{total_ops}")

            else:
                if mode != MODE_CUSTOM_RGB and not palette_colors:
                    self.log("錯誤：未偵測到有效色盤，取消全彩繪製。")
                    return
                color_max_w = max(1, int(canvas_w * line_scale / 100))
                color_max_h = max(1, int(canvas_h * line_scale / 100))
                cell_step_px = color_fill_step(mode, brush_px)
                cells_w = max(1, color_max_w // cell_step_px)
                cells_h = max(1, color_max_h // cell_step_px)

                if mode == MODE_CUSTOM_RGB:
                    color_map, mapping_colors, img_size = image_to_custom_rgb_map(
                        self.image, cells_w, cells_h, color_count=custom_colors, skip_white=skip_white
                    )
                    self.log(f"自訂 RGB 序列計算完畢，顏色數：{len(mapping_colors)}")
                else:
                    color_map, img_size = image_to_palette_map(
                        self.image, mapping_colors, cells_w, cells_h, skip_white=skip_white
                    )
                    self.log("全彩色盤最佳化序列計算完畢。")
                draw_w, draw_h = img_size
                draw_px_w = (draw_w - 1) * cell_step_px + brush_px
                draw_px_h = (draw_h - 1) * cell_step_px + brush_px
                offset_px_x = (canvas_w - draw_px_w) / 2
                offset_px_y = (canvas_h - draw_px_h) / 2

                if skip_white:
                    for wi in white_palette_indices(mapping_colors):
                        color_map[color_map == wi] = -1

                hole_area = color_hole_area(mode, brush_px)
                if hole_area > 0:
                    before_holes = int(np.sum(color_map < 0))
                    color_map = solidify_color_map(color_map, len(mapping_colors), hole_area)
                    filled_holes = max(0, before_holes - int(np.sum(color_map < 0)))
                    if filled_holes > 0:
                        self.log(f"小縫補色完成：補上 {filled_holes} 格")

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
                        cell_step_px=cell_step_px
                    )
                    spiral_paths_by_color = [optimize_spiral_path_order(paths) for paths in spiral_paths_by_color]
                    self.log(
                        "Spiral Fill 蚊香填色已啟用："
                        f"螺旋 {spiral_stats['spiral_paths']} 筆，"
                        f"覆蓋 {spiral_stats['covered_pixels']} 格；"
                        "小細節自動保留長線備援"
                    )

                # 先保留未排序 runs，再產生最佳化版本；實際繪製與 Log 都使用這套排序。
                if mode == MODE_CUSTOM_RGB:
                    raw_runs_by_color, pixel_counts = build_color_runs(
                        fill_source_map,
                        len(mapping_colors),
                        bridge_gap=bridge_gap
                    )
                else:
                    raw_runs_by_color, pixel_counts = build_color_runs_contour(
                        fill_source_map,
                        len(mapping_colors),
                        brush_px=brush_px,
                        bridge_gap=bridge_gap,
                        optimize=False
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
                        self.log(f"輪廓 + 蚊香填色：runs {exact_run_count} -> 螺旋 {spiral_path_count} + 備援 {bridged_run_count}，約減少 {contour_reduced:.1f}%")
                    else:
                        contour_reduced = max(0.0, (exact_run_count - bridged_run_count) / exact_run_count * 100.0)
                        self.log(f"輪廓/RDP 填色啟用：runs {exact_run_count} -> {bridged_run_count}，約減少 {contour_reduced:.1f}%")
                if bridge_gap > 0 and exact_run_count > 0:
                    reduced = max(0.0, (exact_run_count - bridged_run_count) / exact_run_count * 100.0)
                    self.log(f"長線填滿啟用：runs {exact_run_count} -> {bridged_run_count}，約減少 {reduced:.1f}%")
                self.log(f"色塊路徑排序完成：同色空移動距離 {original_run_air:.0f} -> {optimized_run_air:.0f}，約減少 {run_saved:.1f}%")

                eye_detail_runs_by_color = None
                eye_detail_offsets = None
                eye_detail_cell_step_px = 1
                eye_detail_brush_px = gartic_brush_pixels(1)
                if eye_detail and mode in (MODE_PALETTE, MODE_CUSTOM_RGB):
                    eye_scale = min(100, max(line_scale, 95))
                    eye_max_w = max(1, int(canvas_w * eye_scale / 100))
                    eye_max_h = max(1, int(canvas_h * eye_scale / 100))
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
                            (canvas_w - eye_draw_px_w) / 2,
                            (canvas_h - eye_draw_px_h) / 2,
                        )
                        eye_run_count = sum(len(runs) for runs in eye_detail_runs_by_color)
                        self.log(f"眼睛細節強化已準備：{eye_run_count} 小段，最後補畫避免被底色吃掉")

                self.signals.draw_phase_changed.emit("drawing")
                self.wait_before_drawing()
                select_gartic_brush(brush_key, self.brush_positions, self.stop_event)
                self.log(f"已切換 Gartic 筆刷：{brush_key}")

                total_ops = 0
                color_order = color_draw_order(mapping_colors)
                white_indices = white_palette_indices(mapping_colors) if skip_white and mode != MODE_CUSTOM_RGB else set()

                for color_idx in color_order:
                    if color_idx in white_indices:
                        continue
                    raise_if_stopped(self.stop_event)

                    runs = runs_by_color[color_idx]
                    spiral_paths = spiral_paths_by_color[color_idx] if 'spiral_paths_by_color' in locals() else []
                    if not runs and not spiral_paths:
                        continue

                    if mode == MODE_CUSTOM_RGB:
                        set_custom_rgb_color(mapping_colors[color_idx], self.custom_rgb_controls, self.stop_event)
                        self.wait_or_stop(0.05)
                    else:
                        px, py = palette_positions[color_idx]
                        pyautogui.click(px, py)
                        self.wait_or_stop(0.05)

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

                        if end - start <= 1:
                            pyautogui.click(sx, sy)
                        else:
                            pyautogui.moveTo(sx, sy, duration=0)
                            drag_distance = abs(ex - sx)
                            requested_drag = max(0.001, line_move_ms / 1000.0)
                            if brush_px <= 3:
                                min_drag_duration = 0.0012 if line_move_ms <= 2 else 0.0025 if line_move_ms <= 5 else 0.006
                                max_drag_duration = min(0.018, max(min_drag_duration, requested_drag))
                            elif brush_px <= 6:
                                min_drag_duration = 0.001 if line_move_ms <= 3 else 0.0025
                                max_drag_duration = min(0.012, max(min_drag_duration, requested_drag))
                            else:
                                min_drag_duration = 0.0008
                                max_drag_duration = min(0.008, max(min_drag_duration, requested_drag))
                            pyautogui.dragTo(
                                ex, sy,
                                duration=max(min_drag_duration, min(max_drag_duration, drag_distance / 15000)),
                                button="left"
                            )
                        total_ops += 1
                        if total_ops % 10 == 0:
                            self.wait_or_stop(max(delay, 0.001))

                if eye_detail_runs_by_color is not None and eye_detail_offsets is not None:
                    self.log("開始補畫眼睛細節...")
                    select_gartic_brush(1, self.brush_positions, self.stop_event)
                    eye_offset_x, eye_offset_y = eye_detail_offsets
                    eye_white_indices = set(white_palette_indices(mapping_colors))
                    eye_order = [idx for idx in color_draw_order(mapping_colors) if idx not in eye_white_indices]

                    for color_idx in eye_order:
                        raise_if_stopped(self.stop_event)
                        eye_runs = eye_detail_runs_by_color[color_idx]
                        if not eye_runs:
                            continue
                        if mode == MODE_CUSTOM_RGB:
                            set_custom_rgb_color(mapping_colors[color_idx], self.custom_rgb_controls, self.stop_event)
                            self.wait_or_stop(0.035)
                        else:
                            px, py = palette_positions[color_idx]
                            pyautogui.click(px, py)
                            self.wait_or_stop(0.035)

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
                            if total_ops % 25 == 0:
                                self.wait_or_stop(max(delay, 0.001))

                self.log(f"全彩繪製完畢，總渲染區塊數：{total_ops}")

        except pyautogui.FailSafeException:
            self.log("🌟 [安全機制觸發] 已成功緊急停止繪製。")
        except StopDrawingException:
            self.log("已停止繪製。")
        except Exception as e:
            self.log(f"繪製過程中斷或出錯：{e}")
        finally:
            self.signals.drawing_changed.emit(False)


if __name__ == "__main__":
    app = QApplication([])
    app.setStyle("Fusion")
    window = GarticQtDrawer()
    window.show()
    app.exec()
