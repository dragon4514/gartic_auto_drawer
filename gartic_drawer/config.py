"""Runtime constants and portable project paths."""

import sys
from pathlib import Path

import pyautogui


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
MAX_CUSTOM_COLORS = 384
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
if getattr(sys, "frozen", False):
    PROJECT_DIR = Path(sys.executable).resolve().parent
else:
    PROJECT_DIR = Path(__file__).resolve().parent.parent
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
