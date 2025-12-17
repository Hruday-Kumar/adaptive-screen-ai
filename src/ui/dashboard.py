"""PySide6 dashboard for Adaptive Screen AI."""
from __future__ import annotations

import json
import os
import sys
import threading
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, cast

LOW_LIGHT_DOC_THRESHOLD = 18

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from eye_tracking import (  # noqa: E402
    EyeTracker,
    EyeTrackingMetrics,
    read_calibration_config,
    write_calibration_config,
)


@dataclass
class GaugeConfig:
    label: str
    color: QtGui.QColor
    min_value: int = 0
    max_value: int = 100


class GaugeWidget(QtWidgets.QWidget):
    def __init__(self, config: GaugeConfig, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.config = config
        self._value = 0
        self.setMinimumHeight(120)
        self._label_point_size = 10

    def set_value(self, value: int):
        self._value = max(self.config.min_value, min(self.config.max_value, int(value)))
        self.update()

    def set_label_point_size(self, point_size: int):
        self._label_point_size = max(8, point_size)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent):  # noqa: N802 - Qt API
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        rect = self.rect().adjusted(10, 10, -10, -10)
        pen = QtGui.QPen(QtGui.QColor("#333"), 6)
        painter.setPen(pen)
        painter.drawArc(rect, 225 * 16, 90 * 16)

        span = int((self._value / (self.config.max_value - self.config.min_value)) * 90)
        pen.setColor(self.config.color)
        painter.setPen(pen)
        painter.drawArc(rect, 225 * 16, -span * 16)

        painter.setPen(QtGui.QColor("white"))
        font = painter.font()
        font.setBold(True)
        font.setPointSize(self._label_point_size)
        painter.setFont(font)
        text = f"{self.config.label}\n{self._value:02d}"
        painter.drawText(rect, QtCore.Qt.AlignCenter, text)


class VideoWidget(QtWidgets.QLabel):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(640, 400)
        self.setStyleSheet("background-color: #111; border: 1px solid #333;")

    def update_frame(self, frame: np.ndarray):
        height, width, _ = frame.shape
        bytes_per_line = 3 * width
        qt_image = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_BGR888)
        pixmap = QtGui.QPixmap.fromImage(qt_image).scaled(
            self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.setPixmap(pixmap)


class DashboardWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Adaptive Screen AI")
        self.setStyleSheet("background-color: #1a1a1a; color: white;")
        self.tracker = EyeTracker()
        self.stop_event = threading.Event()
        self._openness_history: deque[float] = deque(maxlen=300)
        self._focus_assist_enabled = False
        self._onboarding_shown = False

        app = QtWidgets.QApplication.instance()
        base_font = QtGui.QFont(app.font() if app else QtGui.QFont())
        self._default_app_font = QtGui.QFont(base_font)
        self._large_app_font = QtGui.QFont(base_font)
        if self._large_app_font.pointSize() > 0:
            self._large_app_font.setPointSize(int(self._large_app_font.pointSize() * 1.35))
        else:
            self._large_app_font.setPointSize(14)
        self._base_font_size = self._default_app_font.pointSize() or 12
        self._large_font_size = int(self._base_font_size * 1.35)
        self._last_metrics_for_doc: Optional[EyeTrackingMetrics] = None
        self._active_mode = "Custom"
        self._mode_changing = False
        self._focus_dim_effect: Optional[QtWidgets.QGraphicsOpacityEffect] = None

        self.video_widget = VideoWidget()
        self.status_label = QtWidgets.QLabel("Initializing camera…")
        self.status_label.setAlignment(QtCore.Qt.AlignLeft)
        status_font = max(14, int(self._base_font_size * 1.1))
        self.status_label.setStyleSheet(f"padding: 8px; font-size: {status_font}px;")

        self.profile_combo = QtWidgets.QComboBox()
        self.profile_combo.currentTextChanged.connect(self.change_profile)
        self.new_profile_button = QtWidgets.QPushButton("New profile…")
        self.new_profile_button.clicked.connect(self.prompt_new_profile)

        self.refresh_profiles_button = QtWidgets.QPushButton("Reload profiles")
        self.refresh_profiles_button.clicked.connect(self.refresh_profiles)

        self.calibrate_button = QtWidgets.QPushButton("Calibrate…")
        self.calibrate_button.clicked.connect(self.open_calibration_dialog)

        self.comfort_hold_button = QtWidgets.QPushButton("Save comfort hold")
        self.comfort_hold_button.clicked.connect(self.toggle_comfort_hold)

        self.auto_brightness_toggle = QtWidgets.QCheckBox("Adaptive brightness")
        self.auto_brightness_toggle.setChecked(self.tracker.auto_brightness)
        self.auto_brightness_toggle.stateChanged.connect(self.toggle_auto_brightness)

        self.manual_brightness_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.manual_brightness_slider.setRange(10, 100)
        self.manual_brightness_slider.setValue(self.tracker.current_brightness)
        self.manual_brightness_slider.valueChanged.connect(self.manual_brightness_changed)
        self.manual_brightness_slider.setEnabled(not self.tracker.auto_brightness)

        self.font_label = QtWidgets.QLabel("Font preview")
        self.font_label.setAlignment(QtCore.Qt.AlignCenter)
        self.font_label.setStyleSheet(
            "background: #111; border: 1px solid #333; padding: 20px; font-size: 24px;"
        )

        self.coach_label = QtWidgets.QLabel("")
        self.coach_label.setWordWrap(True)
        self._coach_style_template = "color: {color}; background-color: {bg}; padding: 6px; font-size: {size}px;"
        self.coach_label.setStyleSheet(
            self._coach_style_template.format(color="#f5c16c", bg="transparent", size=14)
        )
        self.coach_label.hide()

        self.mode_banner = QtWidgets.QLabel("")
        self.mode_banner.setWordWrap(True)
        self.mode_banner.setStyleSheet(
            "color: #80dfff; background: rgba(32, 64, 96, 0.35); padding: 6px; border: 1px solid #294a66;"
        )
        self.mode_banner.hide()

        self.doc_assist_panel = QtWidgets.QGroupBox("Doc assist reader")
        self.doc_assist_panel.setStyleSheet(
            "QGroupBox { border: 1px solid #333; margin-top: 12px; }"
            "QGroupBox::title { subcontrol-origin: margin; padding: 4px 8px; }"
        )
        doc_assist_layout = QtWidgets.QVBoxLayout(self.doc_assist_panel)
        self.doc_assist_text = QtWidgets.QPlainTextEdit()
        self.doc_assist_text.setReadOnly(True)
        self.doc_assist_text.setStyleSheet("background: #111; border: 1px solid #444; padding: 8px;")
        doc_assist_layout.addWidget(self.doc_assist_text)
        self.doc_assist_panel.hide()
        self._banner_timer = QtCore.QTimer(self)
        self._banner_timer.setSingleShot(True)
        self._banner_timer.timeout.connect(self.mode_banner.hide)

        self.ambient_bar = QtWidgets.QProgressBar()
        self.ambient_bar.setRange(0, 100)
        self.ambient_bar.setFormat("Ambient light: %p%")
        self.ambient_bar.setStyleSheet("QProgressBar { background: #222; color: white; }"
                                       " QProgressBar::chunk { background: #3fa9f5; }")

        gauge_layout = QtWidgets.QHBoxLayout()
        self.brightness_gauge = GaugeWidget(GaugeConfig(label="Brightness", color=QtGui.QColor("#f9d976")))
        self.comfort_gauge = GaugeWidget(GaugeConfig(label="Comfort", color=QtGui.QColor("#9be15d")))
        self.openness_gauge = GaugeWidget(
            GaugeConfig(label="Openness", color=QtGui.QColor("#5cf7f7"), max_value=35)
        )
        gauge_layout.addWidget(self.brightness_gauge)
        gauge_layout.addWidget(self.comfort_gauge)
        gauge_layout.addWidget(self.openness_gauge)

        profile_layout = QtWidgets.QHBoxLayout()
        profile_layout.addWidget(QtWidgets.QLabel("Profile"))
        profile_layout.addWidget(self.profile_combo)
        profile_layout.addWidget(self.new_profile_button)
        profile_layout.addWidget(self.calibrate_button)
        profile_layout.addWidget(self.refresh_profiles_button)
        profile_layout.addStretch()
        profile_layout.addWidget(self.comfort_hold_button)

        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.addWidget(self.auto_brightness_toggle)
        controls_layout.addWidget(QtWidgets.QLabel("Manual brightness"))
        controls_layout.addWidget(self.manual_brightness_slider)

        accessibility_layout = QtWidgets.QHBoxLayout()
        accessibility_layout.addWidget(QtWidgets.QLabel("Accessibility"))
        self.high_contrast_toggle = QtWidgets.QCheckBox("High contrast")
        self.high_contrast_toggle.stateChanged.connect(self.update_accessibility)
        self.large_text_toggle = QtWidgets.QCheckBox("Larger text")
        self.large_text_toggle.stateChanged.connect(self.update_accessibility)
        self.doc_assist_toggle = QtWidgets.QCheckBox("Doc assist")
        self.doc_assist_toggle.stateChanged.connect(self.toggle_doc_assist)
        self.focus_assist_toggle = QtWidgets.QCheckBox("Focus assist")
        self.focus_assist_toggle.stateChanged.connect(self.toggle_focus_assist)
        accessibility_layout.addWidget(self.high_contrast_toggle)
        accessibility_layout.addWidget(self.large_text_toggle)
        accessibility_layout.addWidget(self.doc_assist_toggle)
        accessibility_layout.addWidget(self.focus_assist_toggle)
        accessibility_layout.addStretch()

        presets_layout = QtWidgets.QHBoxLayout()
        presets_layout.addWidget(QtWidgets.QLabel("Comfort mode"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Select…", "Relaxation", "Focus", "Presentation"])
        self.mode_combo.setToolTip("Swap between preset brightness, contrast, and assist settings.")
        self.mode_combo.currentTextChanged.connect(self.apply_comfort_mode)
        presets_layout.addWidget(self.mode_combo)
        presets_layout.addStretch()

        self.wellness_group = QtWidgets.QGroupBox("Wellness reminders")
        self.wellness_group.setStyleSheet(
            "QGroupBox { border: 1px solid #333; margin-top: 12px; }"
            "QGroupBox::title { subcontrol-origin: margin; padding: 4px 8px; }"
        )
        wellness_group_layout = QtWidgets.QVBoxLayout(self.wellness_group)
        self._wellness_reminders: Dict[str, Dict[str, object]] = {}
        for key, label_text, default_value, message in (
            ("micro", "Micro-break", 20, "Give your eyes a 20-second rest and refocus on a distant point."),
            ("stretch", "Stretch", 60, "Stand up, stretch your shoulders, and roll your neck gently."),
            ("hydrate", "Hydration", 90, "Grab a sip of water to stay refreshed."),
        ):
            row = QtWidgets.QHBoxLayout()
            title = QtWidgets.QLabel(label_text)
            title.setMinimumWidth(110)
            spin = QtWidgets.QSpinBox()
            spin.setRange(0, 240)
            spin.setSingleStep(5)
            spin.setValue(default_value)
            countdown = QtWidgets.QLabel("Off")
            countdown.setMinimumWidth(90)
            action_button = QtWidgets.QPushButton("Start")

            row.addWidget(title)
            row.addWidget(spin)
            row.addWidget(QtWidgets.QLabel("minutes"))
            row.addWidget(countdown)
            row.addWidget(action_button)
            row.addStretch()
            wellness_group_layout.addLayout(row)

            timer = QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda key=key: self._handle_wellness_timeout(key))
            self._wellness_reminders[key] = {
                "spin": spin,
                "timer": timer,
                "label": countdown,
                "button": action_button,
                "message": message,
                "due": None,
            }
            spin.valueChanged.connect(lambda _value, key=key: self._schedule_wellness_timer(key))
            action_button.clicked.connect(lambda _checked=False, key=key: self._schedule_wellness_timer(key, restart=True))

        wellness_group_layout.addStretch()
        self._wellness_countdown_timer = QtCore.QTimer(self)
        self._wellness_countdown_timer.setInterval(1000)
        self._wellness_countdown_timer.timeout.connect(self._update_wellness_countdowns)
        self._wellness_countdown_timer.start()

        sharing_layout = QtWidgets.QHBoxLayout()
        sharing_layout.addWidget(QtWidgets.QLabel("Profiles"))
        export_button = QtWidgets.QPushButton("Export…")
        export_button.clicked.connect(self.export_profiles)
        import_button = QtWidgets.QPushButton("Import…")
        import_button.clicked.connect(self.import_profiles)
        sharing_layout.addWidget(export_button)
        sharing_layout.addWidget(import_button)
        sharing_layout.addStretch()

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll_content = QtWidgets.QWidget()
        self.scroll_area.setWidget(self.scroll_content)
        scroll_layout = QtWidgets.QVBoxLayout(self.scroll_content)
        scroll_layout.addLayout(profile_layout)
        scroll_layout.addLayout(gauge_layout)
        scroll_layout.addWidget(self.coach_label)
        scroll_layout.addWidget(self.mode_banner)
        scroll_layout.addWidget(self.doc_assist_panel)
        scroll_layout.addWidget(self.font_label)
        scroll_layout.addLayout(controls_layout)
        scroll_layout.addLayout(accessibility_layout)
        scroll_layout.addLayout(presets_layout)
        scroll_layout.addWidget(self.wellness_group)
        scroll_layout.addLayout(sharing_layout)
        scroll_layout.addWidget(self.ambient_bar)
        scroll_layout.addStretch()

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(self.video_widget)
        main_layout.addWidget(self.scroll_area)
        main_layout.addWidget(self.status_label)

        self.update_timer = QtCore.QTimer()
        self.update_timer.setInterval(40)
        self.update_timer.timeout.connect(self.process_latest_metrics)
        self.update_timer.start()

        self._metrics_lock = threading.Lock()
        self._latest_metrics: Optional[EyeTrackingMetrics] = None

        self.worker_thread = threading.Thread(target=self._run_tracker, daemon=True)
        self.worker_thread.start()

        self.update_profile_combo()
        for key in self._wellness_reminders:
            self._schedule_wellness_timer(key)

        QtCore.QTimer.singleShot(400, self.show_onboarding)

    def closeEvent(self, event: QtGui.QCloseEvent):  # noqa: N802 - Qt API
        self.stop_event.set()
        self.worker_thread.join(timeout=2.0)
        return super().closeEvent(event)

    def _run_tracker(self) -> None:
        def callback(metrics: EyeTrackingMetrics):
            with self._metrics_lock:
                self._latest_metrics = metrics

        self.tracker.run(callback, self.stop_event)

    def process_latest_metrics(self):
        with self._metrics_lock:
            metrics = self._latest_metrics
            self._latest_metrics = None
        if not metrics:
            return

        if metrics.frame is not None:
            self.video_widget.update_frame(metrics.frame)
        self.status_label.setText(metrics.status)
        self.comfort_hold_button.setText(
            "Resume adaptive" if metrics.locked else "Save comfort hold"
        )
        self.comfort_hold_button.setStyleSheet(
            "background-color: #2a6b2a; color: white;" if metrics.locked else ""
        )

        assigned_font_size = max(18, min(72, metrics.font_size))
        self.font_label.setStyleSheet(
            f"background: #111; border: 1px solid #333; padding: 20px; font-size: {assigned_font_size}px;"
        )
        self.font_label.setText("Eye-Friendly Preview")

        self.brightness_gauge.set_value(metrics.brightness)
        self.comfort_gauge.set_value(metrics.comfort_score)
        self.openness_gauge.set_value(int(metrics.openness))

        self.auto_brightness_toggle.blockSignals(True)
        self.auto_brightness_toggle.setChecked(metrics.auto_brightness)
        self.auto_brightness_toggle.blockSignals(False)
        self.manual_brightness_slider.setEnabled(not metrics.auto_brightness)

        self.manual_brightness_slider.blockSignals(True)
        self.manual_brightness_slider.setValue(metrics.brightness)
        self.manual_brightness_slider.blockSignals(False)

        if metrics.openness > 0:
            self._openness_history.append(metrics.openness)

        if metrics.coaching_message:
            self.coach_label.setText(metrics.coaching_message)
            self.coach_label.show()
        else:
            self.coach_label.hide()

        self.ambient_bar.setValue(int(metrics.ambient_level))
        self._update_doc_assist(metrics)

    @QtCore.Slot(int)
    def toggle_auto_brightness(self, state: int):
        enabled = state == QtCore.Qt.CheckState.Checked
        self.tracker.set_auto_brightness(enabled)
        self.manual_brightness_slider.setEnabled(not enabled)
        if not self._mode_changing:
            self._active_mode = "Custom"
            self._show_banner("")
            self._flash_widget(self.auto_brightness_toggle)
            if not enabled:
                self._flash_widget(self.manual_brightness_slider)
            self._show_banner("")
            message = "Adaptive brightness turned {}.".format("on" if enabled else "off")
            self._update_doc_assist(None, message)

    @QtCore.Slot(int)
    def manual_brightness_changed(self, value: int):
        self.tracker.set_manual_brightness(value, apply_now=True)
        if not self._mode_changing:
            self._active_mode = "Custom"
            self._flash_widget(self.manual_brightness_slider)
            self._show_banner("")
            self._update_doc_assist(None, f"Manual brightness adjusted to {value}%.")

    @QtCore.Slot(str)
    def change_profile(self, profile: str):
        if not profile:
            return
        success, error = self.tracker.set_active_profile(profile)
        if not success and error:
            QtWidgets.QMessageBox.warning(self, "Profile", error)
            self.update_profile_combo()

    @QtCore.Slot()
    def prompt_new_profile(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "New profile", "Profile name:")
        if not ok or not name.strip():
            return
        success, error = self.tracker.create_profile(name)
        if not success and error:
            QtWidgets.QMessageBox.warning(self, "Profile", error)
            return
        self.update_profile_combo()

    @QtCore.Slot()
    def refresh_profiles(self):
        self.tracker.reload_calibration()
        self.update_profile_combo()

    @QtCore.Slot()
    def toggle_comfort_hold(self):
        if self.tracker.locked:
            self.tracker.resume_adaptive()
        else:
            self.tracker.save_comfort_settings()
        self.update_profile_combo()

    @QtCore.Slot()
    def open_calibration_dialog(self):
        dialog = CalibrationDialog(self)
        dialog.exec()

    def get_recent_openness_samples(self, sample_count: int = 60) -> List[float]:
        samples = list(self._openness_history)
        if not samples:
            return []
        if sample_count <= 0 or len(samples) <= sample_count:
            return samples
        return samples[-sample_count:]

    def apply_calibration_from_ui(self, open_value: float, squint_value: float) -> Tuple[bool, Optional[str]]:
        success, error = self.tracker.update_calibration(open_value, squint_value, self.tracker.active_profile, activate=True)
        if success:
            self.update_profile_combo()
        return success, error

    def update_profile_combo(self):
        profiles = self.tracker.available_profiles()
        active = self.tracker.active_profile
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItems(profiles)
        index = self.profile_combo.findText(active)
        if index >= 0:
            self.profile_combo.setCurrentIndex(index)
        self.profile_combo.blockSignals(False)
        self.comfort_hold_button.setText(
            "Resume adaptive" if self.tracker.locked else "Save comfort hold"
        )

    @QtCore.Slot()
    def show_onboarding(self):
        if self._onboarding_shown:
            return
        self._onboarding_shown = True
        dialog = OnboardingDialog(self)
        dialog.exec()

    @QtCore.Slot()
    def update_accessibility(self):
        if not self._mode_changing:
            self._active_mode = "Custom"
        base_styles = ["color: white;"]
        if self.high_contrast_toggle.isChecked():
            base_styles.append("background-color: #000000; color: #f0f0f0;")
        else:
            base_styles.append("background-color: #1a1a1a;")
        self.setStyleSheet(" ".join(base_styles))

        content_font_px = self._large_font_size if self.large_text_toggle.isChecked() else self._base_font_size
        self.scroll_content.setStyleSheet(f"font-size: {content_font_px}px;")

        status_font = max(14, int(content_font_px * 0.9))
        self.status_label.setStyleSheet(f"padding: 8px; font-size: {status_font}px;")

        preview_size = 32 if self.large_text_toggle.isChecked() else 24
        card_bg = "#050505" if self.high_contrast_toggle.isChecked() else "#111"
        card_border = "#555" if self.high_contrast_toggle.isChecked() else "#333"
        card_text = "#f0f0f0" if self.high_contrast_toggle.isChecked() else "#ffffff"
        self.font_label.setStyleSheet(
            f"background: {card_bg}; border: 1px solid {card_border}; padding: 20px; font-size: {preview_size}px; color: {card_text};"
        )

        coach_size = 18 if self.large_text_toggle.isChecked() else 14
        coach_color = "#ffe08a" if self.high_contrast_toggle.isChecked() else "#f5c16c"
        coach_bg = "#000000" if self.high_contrast_toggle.isChecked() else "transparent"
        self.coach_label.setStyleSheet(
            self._coach_style_template.format(color=coach_color, bg=coach_bg, size=coach_size)
        )

        gauge_font_size = 14 if self.large_text_toggle.isChecked() else 10
        for gauge in (self.brightness_gauge, self.comfort_gauge, self.openness_gauge):
            gauge.set_label_point_size(gauge_font_size)

        doc_bg = "#000" if self.high_contrast_toggle.isChecked() else "#111"
        doc_fg = "#f5f5f5" if self.high_contrast_toggle.isChecked() else "#ffffff"
        self.doc_assist_text.setStyleSheet(
            f"background: {doc_bg}; border: 1px solid #444; padding: 8px; color: {doc_fg}; font-size: {content_font_px}px;"
        )

        self.toggle_focus_assist(self.focus_assist_toggle.checkState())
        if not self._mode_changing:
            self._update_doc_assist(None, "Accessibility preferences updated.")

    @QtCore.Slot(int)
    def toggle_focus_assist(self, state: int):
        enabled = state == QtCore.Qt.CheckState.Checked
        previous = self._focus_assist_enabled
        self._focus_assist_enabled = enabled
        if enabled:
            if self._focus_dim_effect is None:
                effect = QtWidgets.QGraphicsOpacityEffect(self.scroll_area)
                effect.setOpacity(0.55)
                self._focus_dim_effect = effect
            self.scroll_area.setGraphicsEffect(self._focus_dim_effect)
            self.video_widget.setStyleSheet("background-color: #000; border: 3px solid #4da6ff;")
        else:
            if self._focus_dim_effect is not None:
                self.scroll_area.setGraphicsEffect(None)
                self._focus_dim_effect = None
            self.video_widget.setStyleSheet("background-color: #111; border: 1px solid #333;")
        if enabled != previous:
            if enabled:
                self._flash_widget(self.video_widget)
                self._flash_widget(self.focus_assist_toggle)
                if not self._mode_changing:
                    self._active_mode = "Custom"
                    self._show_banner("")
                self._update_doc_assist(
                    None,
                    "Focus assist on — the control panel is dimmed so you can concentrate on the feed.",
                )
            else:
                if not self._mode_changing:
                    self._active_mode = "Custom"
                    self._show_banner("")
                self._update_doc_assist(None, "Focus assist off — full dashboard controls restored.")

    @QtCore.Slot(int)
    def toggle_doc_assist(self, state: int):
        enabled = state == QtCore.Qt.CheckState.Checked
        self.doc_assist_panel.setVisible(enabled)
        if not enabled:
            self.doc_assist_text.clear()
            if not self._mode_changing:
                self._active_mode = "Custom"
                self._show_banner("")
        elif enabled:
            self._flash_widget(self.doc_assist_toggle)
            self._flash_widget(self.doc_assist_panel)
            self._update_doc_assist(
                self._last_metrics_for_doc,
                "Doc assist enabled — changes will be narrated here.",
            )

    def _update_doc_assist(
        self,
        metrics: Optional[EyeTrackingMetrics],
        context_message: Optional[str] = None,
    ):
        if metrics is not None:
            self._last_metrics_for_doc = metrics
        if not self.doc_assist_toggle.isChecked():
            return

        data = self._last_metrics_for_doc
        lines: List[str] = []

        if context_message:
            lines.append(context_message)
            lines.append("")

        lines.append(f"Comfort mode: {self._active_mode}")
        lines.append(
            "Adaptive brightness: {}".format(
                "on (auto adjusting)" if self.auto_brightness_toggle.isChecked() else "off (use the slider)"
            )
        )
        current_brightness = (
            data.brightness if data is not None else self.manual_brightness_slider.value()
        )
        lines.append(f"Brightness level: {current_brightness}%")
        lines.append(f"Focus assist: {'on' if self.focus_assist_toggle.isChecked() else 'off'}")
        lines.append(f"Doc assist: {'on' if self.doc_assist_toggle.isChecked() else 'off'}")

        ambient_value = (
            int(data.ambient_level) if data is not None else int(self.ambient_bar.value())
        )
        lines.append(f"Ambient light: {ambient_value}%")

        if data is None:
            lines.append("")
            lines.append("Tracking feed is warming up…")
        else:
            lines.append("")
            lines.append(f"Status: {data.status}")
            lines.append(f"Comfort score: {data.comfort_score}/100")
            if data.coaching_message:
                lines.append("")
                lines.append("Coach tip:")
                lines.append(data.coaching_message)
            elif ambient_value <= LOW_LIGHT_DOC_THRESHOLD:
                lines.append("")
                lines.append("Coach tip:")
                lines.append("Lighting is low—try night lighting or increase screen brightness for better tracking.")

        new_text = "\n".join(lines)
        if self.doc_assist_text.toPlainText() != new_text:
            self.doc_assist_text.setPlainText(new_text)

    def _show_banner(self, message: str, duration_ms: int = 4000):
        if not message:
            self.mode_banner.hide()
            return
        self.mode_banner.setText(message)
        self.mode_banner.show()
        self._banner_timer.start(duration_ms)

    def _flash_widget(self, widget: Optional[QtWidgets.QWidget], duration_ms: int = 1200):
        if widget is None:
            return
        effect = QtWidgets.QGraphicsColorizeEffect(widget)
        effect.setColor(QtGui.QColor("#4da6ff"))
        effect.setStrength(0.9)
        widget.setGraphicsEffect(effect)

        def _clear_effect():
            if widget.graphicsEffect() is effect:
                widget.setGraphicsEffect(None)

        QtCore.QTimer.singleShot(duration_ms, _clear_effect)

    @QtCore.Slot(str)
    def apply_comfort_mode(self, mode: str):
        if mode == "Select…":
            return
        self._mode_changing = True
        changed_widgets: List[QtWidgets.QWidget] = []
        banner_message = ""
        doc_message = ""

        try:
            if mode == "Relaxation":
                self.tracker.resume_adaptive()
                if not self.auto_brightness_toggle.isChecked():
                    self.auto_brightness_toggle.setChecked(True)
                    changed_widgets.append(self.auto_brightness_toggle)
                if self.high_contrast_toggle.isChecked():
                    self.high_contrast_toggle.setChecked(False)
                    changed_widgets.append(self.high_contrast_toggle)
                if self.focus_assist_toggle.isChecked():
                    self.focus_assist_toggle.setChecked(False)
                    changed_widgets.append(self.focus_assist_toggle)
                if not self.doc_assist_toggle.isChecked():
                    self.doc_assist_toggle.setChecked(True)
                    changed_widgets.append(self.doc_assist_toggle)
                banner_message = "Relaxation mode: adaptive brightness and doc assist are enabled for a softer session."
                doc_message = "Relaxation mode enabled — adaptive brightness is back on and high contrast is off."

            elif mode == "Focus":
                if self.auto_brightness_toggle.isChecked():
                    self.auto_brightness_toggle.setChecked(False)
                    changed_widgets.append(self.auto_brightness_toggle)
                self.tracker.set_manual_brightness(65, apply_now=True)
                self.manual_brightness_slider.blockSignals(True)
                self.manual_brightness_slider.setValue(65)
                self.manual_brightness_slider.blockSignals(False)
                changed_widgets.append(self.manual_brightness_slider)
                if self.tracker.locked:
                    self.tracker.toggle_lock()
                if not self.focus_assist_toggle.isChecked():
                    self.focus_assist_toggle.setChecked(True)
                    changed_widgets.append(self.focus_assist_toggle)
                if self.high_contrast_toggle.isChecked():
                    self.high_contrast_toggle.setChecked(False)
                    changed_widgets.append(self.high_contrast_toggle)
                if self.doc_assist_toggle.isChecked():
                    self.doc_assist_toggle.setChecked(False)
                    changed_widgets.append(self.doc_assist_toggle)
                banner_message = "Focus mode: brightness pinned at 65%. Use the slider to fine-tune or re-enable adaptive brightness."
                doc_message = "Focus mode active — brightness is steady, focus assist is on, and doc assist is paused."

            elif mode == "Presentation":
                if self.auto_brightness_toggle.isChecked():
                    self.auto_brightness_toggle.setChecked(False)
                    changed_widgets.append(self.auto_brightness_toggle)
                self.tracker.set_manual_brightness(85, apply_now=True)
                self.manual_brightness_slider.blockSignals(True)
                self.manual_brightness_slider.setValue(85)
                self.manual_brightness_slider.blockSignals(False)
                changed_widgets.append(self.manual_brightness_slider)
                if not self.high_contrast_toggle.isChecked():
                    self.high_contrast_toggle.setChecked(True)
                    changed_widgets.append(self.high_contrast_toggle)
                if self.focus_assist_toggle.isChecked():
                    self.focus_assist_toggle.setChecked(False)
                    changed_widgets.append(self.focus_assist_toggle)
                if self.doc_assist_toggle.isChecked():
                    self.doc_assist_toggle.setChecked(False)
                    changed_widgets.append(self.doc_assist_toggle)
                if not self.tracker.locked:
                    self.tracker.toggle_lock()
                banner_message = "Presentation mode: screen brightness raised and contrast boosted."
                doc_message = "Presentation mode active — brightness is locked high and high contrast is on for visibility."

        finally:
            self._mode_changing = False

        self._active_mode = mode
        for widget in changed_widgets:
            self._flash_widget(widget)
        if banner_message:
            self._show_banner(banner_message)
        if doc_message:
            self._update_doc_assist(None, doc_message)
        self.mode_combo.setCurrentIndex(0)

    def _schedule_wellness_timer(self, key: str, restart: bool = False):
        reminder = self._wellness_reminders.get(key)
        if not reminder:
            return

        timer = cast(QtCore.QTimer, reminder["timer"])
        spin = cast(QtWidgets.QSpinBox, reminder["spin"])
        label = cast(QtWidgets.QLabel, reminder["label"])
        button = cast(QtWidgets.QPushButton, reminder["button"])

        timer.stop()
        minutes = spin.value()
        if minutes <= 0:
            reminder["due"] = None
            label.setText("Off")
            button.setText("Start")
            return

        due = QtCore.QDateTime.currentDateTime().addSecs(minutes * 60)
        reminder["due"] = due
        timer.start(minutes * 60 * 1000)
        label.setText("Due in {} min".format(minutes))
        button.setText("Restart")

    def _handle_wellness_timeout(self, key: str):
        reminder = self._wellness_reminders.get(key)
        if not reminder:
            return

        message = reminder.get("message", "Time for a quick break.")
        titles = {
            "micro": "Micro-break",
            "stretch": "Stretch",
            "hydrate": "Hydration",
        }
        title = titles.get(key, "Wellness reminder")
        QtWidgets.QMessageBox.information(self, title, message)
        # Reschedule immediately to keep the cadence going
        self._schedule_wellness_timer(key)

    def _update_wellness_countdowns(self):
        now = QtCore.QDateTime.currentDateTime()
        for reminder in self._wellness_reminders.values():
            label = cast(QtWidgets.QLabel, reminder["label"])
            due = cast(Optional[QtCore.QDateTime], reminder.get("due"))
            if not due:
                continue
            remaining = now.msecsTo(due)
            if remaining <= 0:
                label.setText("Due now")
            else:
                minutes = remaining // 60000
                seconds = (remaining % 60000) // 1000
                if minutes >= 1:
                    label.setText(f"{minutes}m {seconds:02d}s")
                else:
                    label.setText(f"{seconds}s")

    @QtCore.Slot()
    def export_profiles(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export profiles",
            "calibration-profiles.json",
            "JSON files (*.json)",
        )
        if not path:
            return
        config = read_calibration_config()
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(self, "Export", f"Could not save profiles: {exc}")

    @QtCore.Slot()
    def import_profiles(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import profiles",
            "",
            "JSON files (*.json)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            QtWidgets.QMessageBox.warning(self, "Import", f"Could not read profiles: {exc}")
            return
        write_calibration_config(data)
        self.tracker.reload_calibration()
        self.update_profile_combo()
        QtWidgets.QMessageBox.information(self, "Import", "Profiles imported successfully.")


class OnboardingDialog(QtWidgets.QDialog):
    def __init__(self, dashboard: DashboardWindow):
        super().__init__(dashboard)
        self.dashboard = dashboard
        self.setWindowTitle("Welcome")
        self.setModal(True)
        self.setStyleSheet("background-color: #202020; color: white;")

        intro = QtWidgets.QLabel(
            "Welcome to Adaptive Screen AI. We'll walk you through a quick comfort setup."
        )
        intro.setWordWrap(True)

        step_list = QtWidgets.QLabel(
            "1. Make sure your face is well lit.\n"
            "2. Click 'Start quick calibration' and follow the prompts.\n"
            "3. Adjust brightness or comfort modes any time from the dashboard."
        )
        step_list.setWordWrap(True)

        buttons = QtWidgets.QDialogButtonBox()
        start_btn = buttons.addButton("Start quick calibration", QtWidgets.QDialogButtonBox.AcceptRole)
        skip_btn = buttons.addButton("Skip for now", QtWidgets.QDialogButtonBox.RejectRole)
        buttons.accepted.connect(self.start_calibration)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(step_list)
        layout.addWidget(buttons)

    def start_calibration(self):
        self.accept()
        QtCore.QTimer.singleShot(0, self.dashboard.open_calibration_dialog)


class CalibrationDialog(QtWidgets.QDialog):
    def __init__(self, dashboard: DashboardWindow):
        super().__init__(dashboard)
        self.dashboard = dashboard
        self.setWindowTitle("Calibrate Profile")
        self.setModal(True)
        self.setStyleSheet("background-color: #202020; color: white;")

        self.open_value: Optional[float] = None
        self.squint_value: Optional[float] = None

        intro = QtWidgets.QLabel(
            "Look at the screen and keep your face centered. Capture open eyes first, then squint."
        )
        intro.setWordWrap(True)

        self.open_label = QtWidgets.QLabel("Open: not captured")
        self.squint_label = QtWidgets.QLabel("Squint: not captured")

        self.status_label = QtWidgets.QLabel("Waiting for capture…")

        capture_open_btn = QtWidgets.QPushButton("Capture open")
        capture_open_btn.clicked.connect(self.capture_open)

        capture_squint_btn = QtWidgets.QPushButton("Capture squint")
        capture_squint_btn.clicked.connect(self.capture_squint)

        self.save_button = QtWidgets.QPushButton("Save calibration")
        self.save_button.clicked.connect(self.save_calibration)

        cancel_button = QtWidgets.QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        buttons_layout = QtWidgets.QHBoxLayout()
        buttons_layout.addWidget(capture_open_btn)
        buttons_layout.addWidget(capture_squint_btn)
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.save_button)
        buttons_layout.addWidget(cancel_button)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(self.open_label)
        layout.addWidget(self.squint_label)
        layout.addWidget(self.status_label)
        layout.addLayout(buttons_layout)

    def capture_open(self):
        value = self._capture_average()
        if value is None:
            QtWidgets.QMessageBox.warning(self, "Calibration", "Need a face in view before capturing.")
            return
        self.open_value = value
        self.open_label.setText(f"Open: {value:.2f}")
        self.status_label.setText("Open level captured. Now squint and capture again.")

    def capture_squint(self):
        value = self._capture_average()
        if value is None:
            QtWidgets.QMessageBox.warning(self, "Calibration", "Need a face in view before capturing.")
            return
        self.squint_value = value
        self.squint_label.setText(f"Squint: {value:.2f}")
        self.status_label.setText("Review values, then save.")

    def save_calibration(self):
        if self.open_value is None or self.squint_value is None:
            QtWidgets.QMessageBox.information(self, "Calibration", "Capture both open and squint values first.")
            return

        success, error = self.dashboard.apply_calibration_from_ui(self.open_value, self.squint_value)
        if not success:
            QtWidgets.QMessageBox.warning(self, "Calibration", error or "Unable to save calibration.")
            return

        self.accept()

    def _capture_average(self) -> Optional[float]:
        samples = [s for s in self.dashboard.get_recent_openness_samples(80) if s > 0]
        if len(samples) < 5:
            return None
        return float(np.mean(samples))


def run_dashboard():
    app = QtWidgets.QApplication(sys.argv)
    window = DashboardWindow()
    window.resize(960, 900)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run_dashboard())
