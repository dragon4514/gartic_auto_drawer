"""Main Qt application window."""

import json
import threading
import time

import cv2
import numpy as np
import pyautogui
from PIL import Image
from PySide6.QtCore import QObject, Qt, Signal, QTimer
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QButtonGroup,
    QCheckBox,
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

from .automation import *
from .common import *
from .config import *
from .detection import *
from .image_processing import *
from .ui.overlays import DetectionOverlay, ImagePlacementOverlay
from .ui.widgets import AnimatedActionButton, BorderlessProfileCombo, OptionToggle, pil_to_qpixmap

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
            self.eye_detail_check.setToolTip("只在 Palette / Custom RGB 上色模式補畫眼睛細節。")
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
        brush_key = clamp_brush_key(self.brush_spin.value())
        skip_white = self.skip_white_check.isChecked()
        mode = self.mode_value
        detail = int(self.detail_spin.value())
        line_scale = min(100, max(40, int(self.line_scale_spin.value())))
        custom_colors = int(np.clip(int(self.custom_colors_spin.value()), 8, MAX_CUSTOM_COLORS))
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
                info = f"SBR 筆觸預覽 | {pos_text} | 畫布: {canvas_w}x{canvas_h} | 目標區: {int(target_w)}x{int(target_h)} | 圖像: {img_size[0]}x{img_size[1]} | 筆觸: {len(strokes)} | 筆刷鍵: {brush_key}"
            elif mode in (MODE_SMART_LINE, MODE_LINE, MODE_CLEAN_LINE, MODE_DARK_OUTLINE):
                self.log("運算中：正在分析線稿，UI 已改成非阻塞模式。")
                line_max_w = max(1, int(target_w))
                line_max_h = max(1, int(target_h))
                strokes, img_size = image_to_smart_line_strokes(self.image, line_max_w, line_max_h, detail=detail)
                content = line_strokes_to_image(strokes, img_size, line_width=brush_px)
                offset_x = target_x + (target_w - content.size[0]) / 2
                offset_y = target_y + (target_h - content.size[1]) / 2
                preview = compose_canvas_preview_at(content, canvas_w, canvas_h, offset_x, offset_y)
                pos_text = "定位" if using_placement else "置中"
                info = f"整合線稿預覽 | {pos_text} | 畫布: {canvas_w}x{canvas_h} | 目標區: {int(target_w)}x{int(target_h)} | 圖像: {img_size[0]}x{img_size[1]} | 筆刷鍵: {brush_key} | 總筆畫: {len(strokes)}"
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

                if eye_detail and mode in (MODE_PALETTE, MODE_CUSTOM_RGB):
                    eye_map, _eye_size = image_to_eye_detail_map(
                        self.image, mapping_colors, color_map.shape[1], color_map.shape[0]
                    )
                    eye_map = resize_label_map_to_shape(eye_map, color_map.shape)
                    for wi in white_palette_indices(mapping_colors):
                        eye_map[eye_map == wi] = -1
                    eye_mask = eye_map >= 0
                    if np.any(eye_mask):
                        color_map[eye_mask] = eye_map[eye_mask]

                hole_area = color_hole_area(mode, brush_px)
                if hole_area > 0:
                    color_map = solidify_color_map(color_map, len(mapping_colors), hole_area)
                color_map, _sealed_preview_gaps = seal_thin_color_gaps(
                    color_map,
                    len(mapping_colors),
                    max_thickness=max(2, int(round(brush_px / max(1, cell_step_px)))),
                    max_area=max(18, int(max(color_map.shape) * 1.5))
                )

                preview_color_map = color_map
                spiral_paths_by_color = None
                spiral_count = 0
                spiral_enabled = bool(spiral_fill and mode in (MODE_PALETTE, MODE_CUSTOM_RGB))

                if spiral_enabled:
                    spiral_paths_by_color, preview_color_map, _spiral_stats = build_spiral_fill_paths(
                        color_map,
                        len(mapping_colors),
                        brush_px=brush_px,
                        cell_step_px=cell_step_px,
                        remove_from_fallback=False
                    )
                    spiral_paths_by_color = [optimize_spiral_path_order(paths) for paths in spiral_paths_by_color]
                    spiral_count = sum(len(paths) for paths in spiral_paths_by_color)

                content = color_map_to_gartic_preview(
                    preview_color_map, mapping_colors, brush_px,
                    reverse_order=(mode == MODE_CUSTOM_RGB),
                    bridge_gap=color_run_bridge_gap(mode, brush_px),
                    use_contour=False,
                    cell_step_px=cell_step_px,
                    spiral_paths_by_color=spiral_paths_by_color
                )
                offset_x = target_x + (target_w - content.size[0]) / 2
                offset_y = target_y + (target_h - content.size[1]) / 2
                preview = compose_canvas_preview_at(content, canvas_w, canvas_h, offset_x, offset_y)
                spiral_state = f"ON/{spiral_count}" if spiral_enabled else "OFF"
                pos_text = "定位" if using_placement else "置中"
                info = f"{label} | {pos_text} | 畫布: {canvas_w}x{canvas_h} | 目標區: {int(target_w)}x{int(target_h)} | 圖像: {content.size[0]}x{content.size[1]} | 網格: {img_size[0]}x{img_size[1]} | 筆刷鍵: {brush_key} | 格距: {cell_step_px} | Spiral: {spiral_state}"
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
                        self.log(f"眼睛細節強化已準備：{eye_run_count} 小段，最後補畫避免被底色吃掉")

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
                    self.log("開始補畫眼睛細節...")
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
