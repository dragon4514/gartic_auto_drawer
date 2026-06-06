"""Image quantization, line extraction, stroke planning, and preview rendering."""

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .automation import contour_simplify_mask
from .common import ResponsiveYield
from .config import *
from .detection import nearest_color_index_map, white_palette_indices

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


def protected_white_shape_mask(rgb, alpha, white_threshold=235, chroma_limit=48, min_area=None, dilate_px=0):
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
        min_area = max(16, int(h * w * 0.00025))

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

        # 被量化成 Gartic 白色色盤的地方仍不直接畫白，後面會用
        # seal_thin_color_gaps() 把非保護區的小縫補回鄰近顏色。
        white_indices = white_palette_indices(palette_colors)
        if white_indices:
            idx[np.isin(idx, list(white_indices))] = -1

    idx[skip] = -1
    idx[protected_white] = PROTECTED_WHITE

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
    responsive = ResponsiveYield()

    for label in range(1, num_labels):
        responsive.maybe()
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
        # 原圖白色 / 接近白色背景直接跳過。
        near_white = np.all(rgb > 246, axis=2)
        almost_white = (
            np.min(rgb, axis=2) > 225
        ) & (
            np.max(rgb, axis=2) - np.min(rgb, axis=2) < 40
        )
        protected_white = protected_white_shape_mask(
            rgb,
            alpha,
            white_threshold=225,
            chroma_limit=55,
            dilate_px=white_protect_radius
        )
        skip = skip | near_white | almost_white

        # 被轉換成 Gartic 白色色盤的，也直接跳過。
        white_indices = white_palette_indices(palette_colors)
        if white_indices:
            idx[np.isin(idx, list(white_indices))] = -1

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
