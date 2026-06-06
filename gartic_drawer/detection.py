"""Screen, canvas, palette, brush, and RGB-control detection."""

import cv2
import mss
import numpy as np

from .common import ResponsiveYield
from .config import FIXED_GARTIC_COLORS, GARTIC_BRUSH_PIXELS

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
