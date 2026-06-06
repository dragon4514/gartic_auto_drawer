"""Reusable Qt widgets and image conversion helpers."""

from PIL import Image
from PySide6.QtCore import Qt, QTimer, QSize, QRectF
from PySide6.QtGui import QColor, QFont, QImage, QPixmap, QPainter, QPen, QLinearGradient, QBrush, QConicalGradient
from PySide6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ..image_processing import fit_preview_image

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
