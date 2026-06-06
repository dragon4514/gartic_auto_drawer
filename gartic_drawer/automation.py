"""Mouse/keyboard automation and draw-path construction."""

import time

import cv2
import numpy as np
import pyautogui

from .common import ResponsiveYield, raise_if_stopped
from .config import *
from .detection import clamp_brush_key, normalize_custom_rgb_controls

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
