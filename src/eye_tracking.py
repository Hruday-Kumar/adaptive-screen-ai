from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

try:  # pragma: no cover - optional dependency
    import screen_brightness_control as sbc
except ImportError:  # pragma: no cover - optional dependency
    sbc = None


CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "calibration.json"

LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

FACE_MISSING_DIM_DELAY = 6.0
FACE_MISSING_BRIGHTNESS = 25
BRIGHTNESS_SMOOTH_FACTOR = 0.25
FONT_SMOOTH_FACTOR = 0.3
BLINK_RATIO = 0.8
BLINK_MIN_FRAMES = 2
BRIGHTNESS_HYSTERESIS = 5
BRIGHTNESS_UPDATE_INTERVAL = 5.0
SQUINT_ALERT_SECONDS = 6.0
COACH_COOLDOWN_SECONDS = 30.0
BLINK_IDLE_SECONDS = 8.0
AMBIENT_SMOOTH_FACTOR = 0.2
LOW_LIGHT_AMBIENT_THRESHOLD = 15.0

DEFAULT_PROFILE_NAME = "default"
DEFAULT_OPEN = 7.0
DEFAULT_SQUINT = 2.5
DEFAULT_BRIGHTNESS = 60
DEFAULT_FONT_SIZE = 24
MIN_FONT_SIZE = 18
MAX_FONT_SIZE = 72


def _default_config() -> Dict[str, object]:
    return {
        "active_profile": DEFAULT_PROFILE_NAME,
        "profiles": {
            DEFAULT_PROFILE_NAME: {
                "open": DEFAULT_OPEN,
                "squint": DEFAULT_SQUINT,
                "comfort": {
                    "brightness": DEFAULT_BRIGHTNESS,
                    "font_size": DEFAULT_FONT_SIZE,
                },
            }
        },
    }


def _sanitize_profile(profile: Dict[str, object]) -> Dict[str, object]:
    open_val = float(profile.get("open", DEFAULT_OPEN))
    squint_val = float(profile.get("squint", DEFAULT_SQUINT))
    if open_val <= squint_val:
        open_val = squint_val + 1.0

    sanitized: Dict[str, object] = {"open": open_val, "squint": squint_val}

    comfort = profile.get("comfort")
    if isinstance(comfort, dict):
        brightness = int(max(10, min(100, comfort.get("brightness", DEFAULT_BRIGHTNESS))))
        font_size = int(max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, comfort.get("font_size", DEFAULT_FONT_SIZE))))
        sanitized["comfort"] = {"brightness": brightness, "font_size": font_size}

    return sanitized


def _sanitize_config(raw: Dict[str, object]) -> Dict[str, object]:
    config = _default_config()
    if not isinstance(raw, dict):
        return config

    profiles = raw.get("profiles")
    if isinstance(profiles, dict) and profiles:
        sanitized_profiles: Dict[str, Dict[str, object]] = {}
        for name, data in profiles.items():
            if isinstance(name, str) and isinstance(data, dict):
                sanitized_profiles[name] = _sanitize_profile(data)
        if sanitized_profiles:
            config["profiles"] = sanitized_profiles
    elif "open" in raw and "squint" in raw:
        config["profiles"][DEFAULT_PROFILE_NAME] = _sanitize_profile(raw)

    active_profile = raw.get("active_profile")
    if isinstance(active_profile, str) and active_profile in config["profiles"]:
        config["active_profile"] = active_profile
    else:
        config["active_profile"] = next(iter(config["profiles"]))

    return config


def _write_config_to_disk(config: Dict[str, object]) -> None:
    CALIBRATION_PATH.write_text(json.dumps(config, indent=2))


def read_calibration_config() -> Dict[str, object]:
    if not CALIBRATION_PATH.exists():
        config = _default_config()
        _write_config_to_disk(config)
        return config

    try:
        raw = json.loads(CALIBRATION_PATH.read_text())
    except Exception:
        raw = {}
    config = _sanitize_config(raw)
    if config != raw:
        _write_config_to_disk(config)
    return config


def write_calibration_config(config: Dict[str, object]) -> Dict[str, object]:
    sanitized = _sanitize_config(config if isinstance(config, dict) else {})
    _write_config_to_disk(sanitized)
    return sanitized


def map_range(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    if in_max <= in_min:
        return out_min
    value = max(min(value, in_max), in_min)
    ratio = (value - in_min) / (in_max - in_min)
    return out_min + ratio * (out_max - out_min)


@dataclass
class EyeTrackingMetrics:
    frame: Optional[np.ndarray]
    openness: float
    brightness: int
    font_size: int
    comfort_score: int
    locked: bool
    status: str
    blink_detected: bool
    auto_brightness: bool
    coaching_message: str
    ambient_level: float


class EyeTracker:
    def __init__(self, camera_index: int = 0):
        self._camera_index = camera_index
        self._lock = threading.Lock()

        self._auto_brightness = sbc is not None
        self._manual_brightness = DEFAULT_BRIGHTNESS
        self._locked = False
        self._locked_values = {
            "brightness": DEFAULT_BRIGHTNESS,
            "font_size": DEFAULT_FONT_SIZE,
            "openness": DEFAULT_OPEN,
        }
        self._smoothed_brightness: Optional[float] = None
        self._smoothed_font: Optional[float] = None
        self._last_brightness: Optional[int] = None
        self._last_brightness_ts = 0.0
        self._brightness_error_logged = False
        self._away_dimmed = False
        self._last_face_time = time.time()
        self._blink_counter = 0
        self._current_brightness = DEFAULT_BRIGHTNESS
        self._latest_font_size = DEFAULT_FONT_SIZE
        self._squint_since: Optional[float] = None
        self._last_coach_prompt = 0.0
        self._ambient_level = 50.0
        self._last_blink_time = time.time()

        self._config = read_calibration_config()
        self._open_level = DEFAULT_OPEN
        self._squint_level = DEFAULT_SQUINT
        self._apply_profile(self._config["active_profile"])

    @property
    def auto_brightness(self) -> bool:
        with self._lock:
            return self._auto_brightness

    @property
    def locked(self) -> bool:
        with self._lock:
            return self._locked

    @property
    def current_brightness(self) -> int:
        with self._lock:
            return int(self._current_brightness)

    @property
    def active_profile(self) -> str:
        with self._lock:
            return self._config["active_profile"]

    def available_profiles(self) -> List[str]:
        with self._lock:
            return list(self._config["profiles"].keys())

    def toggle_lock(self) -> bool:
        with self._lock:
            self._locked = not self._locked
            if self._locked:
                self._locked_values["brightness"] = int(self._current_brightness)
                self._locked_values["font_size"] = int(self._latest_font_size)
                self._locked_values["openness"] = float(self._open_level)
            else:
                self._away_dimmed = False
        return self.locked

    def set_auto_brightness(self, enabled: bool):
        with self._lock:
            self._auto_brightness = bool(enabled) and sbc is not None
            if self._auto_brightness:
                self._smoothed_brightness = None
            else:
                self._manual_brightness = int(self._current_brightness)

    def set_manual_brightness(self, value: int, apply_now: bool = True) -> int:
        value = int(max(10, min(100, value)))
        with self._lock:
            self._manual_brightness = value
        if not self.auto_brightness and apply_now:
            self._apply_brightness(value, force=True)
            with self._lock:
                self._current_brightness = value
        return value

    def reload_calibration(self):
        with self._lock:
            self._config = read_calibration_config()
            profile = self._config["active_profile"]
            self._apply_profile(profile)

    def create_profile(self, profile_name: str) -> Tuple[bool, Optional[str]]:
        name = profile_name.strip()
        if not name:
            return False, "Profile name cannot be empty."
        with self._lock:
            if name in self._config["profiles"]:
                return False, f"Profile '{name}' already exists."
            base_profile = self._config["profiles"][self._config["active_profile"]]
            self._config["profiles"][name] = {
                "open": base_profile["open"],
                "squint": base_profile["squint"],
            }
            self._config["active_profile"] = name
            self._apply_profile(name)
            self._config = write_calibration_config(self._config)
        return True, None

    def set_active_profile(self, profile_name: str) -> Tuple[bool, Optional[str]]:
        name = profile_name.strip()
        if not name:
            return False, "Profile name cannot be empty."
        with self._lock:
            if name not in self._config["profiles"]:
                return False, f"Profile '{name}' does not exist."
            self._config["active_profile"] = name
            self._apply_profile(name)
            self._config = write_calibration_config(self._config)
        return True, None

    def save_comfort_settings(self) -> None:
        with self._lock:
            profile = self._config["profiles"][self._config["active_profile"]]
            profile["comfort"] = {
                "brightness": int(self._current_brightness),
                "font_size": int(self._latest_font_size),
            }
            self._locked = True
            self._locked_values["brightness"] = int(self._current_brightness)
            self._locked_values["font_size"] = int(self._latest_font_size)
            self._locked_values["openness"] = float(self._open_level)
            self._config = write_calibration_config(self._config)

    def resume_adaptive(self) -> None:
        with self._lock:
            self._locked = False
            self._away_dimmed = False
            self._smoothed_brightness = None
            self._smoothed_font = None

    def update_calibration(
        self,
        open_value: float,
        squint_value: float,
        profile_name: Optional[str] = None,
        activate: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        try:
            new_open = float(open_value)
            new_squint = float(squint_value)
        except (TypeError, ValueError):
            return False, "Calibration values must be numeric."

        if not np.isfinite(new_open) or not np.isfinite(new_squint):
            return False, "Calibration values must be finite numbers."

        if new_open <= new_squint:
            new_open = new_squint + 1.0

        name = (profile_name or self._config["active_profile"]).strip() or self._config["active_profile"]

        with self._lock:
            if name not in self._config["profiles"]:
                self._config["profiles"][name] = {
                    "open": DEFAULT_OPEN,
                    "squint": DEFAULT_SQUINT,
                }

            profile = self._config["profiles"][name]
            comfort = profile.get("comfort") if isinstance(profile, dict) else None

            self._config["profiles"][name] = {
                "open": new_open,
                "squint": new_squint,
            }
            if isinstance(comfort, dict):
                self._config["profiles"][name]["comfort"] = comfort

            if activate:
                self._config["active_profile"] = name

            self._config = write_calibration_config(self._config)
            self._apply_profile(self._config["active_profile"])

        return True, None

    def run(self, callback: Callable[[EyeTrackingMetrics], None], stop_event: threading.Event):
        cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            metrics = EyeTrackingMetrics(
                frame=placeholder,
                openness=0.0,
                brightness=self.current_brightness,
                font_size=self._latest_font_size,
                comfort_score=0,
                locked=self.locked,
                status="Camera not available",
                blink_detected=False,
                auto_brightness=self.auto_brightness,
                coaching_message="",
                ambient_level=self._ambient_level,
            )
            callback(metrics)
            return

        openness_history = deque(maxlen=8)
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        try:
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                frame = cv2.flip(frame, 1)
                display_frame = frame.copy()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                with self._lock:
                    locked = self._locked
                    auto_brightness = self._auto_brightness
                    manual_brightness = self._manual_brightness
                    open_level = self._open_level
                    squint_level = self._squint_level

                status_message = ""
                comfort_score = 0
                blink_detected = False
                current_openness = 0.0
                coach_message = ""
                ambient_level = self._update_ambient_level(frame)
                low_light = ambient_level <= LOW_LIGHT_AMBIENT_THRESHOLD

                if results.multi_face_landmarks:
                    self._last_face_time = time.time()
                    self._away_dimmed = False

                    landmarks = results.multi_face_landmarks[0].landmark
                    current_openness = self._iris_openness(landmarks, frame.shape[:2])
                    openness_history.append(current_openness)
                    avg_openness = float(np.mean(openness_history))

                    now = time.time()
                    if current_openness < self._blink_threshold:
                        self._blink_counter += 1
                    else:
                        if 0 < self._blink_counter <= BLINK_MIN_FRAMES * 3:
                            blink_detected = True
                            self._last_blink_time = now
                        self._blink_counter = 0

                    strain_ratio = 0.35 if not low_light else 0.22
                    if avg_openness <= (squint_level + (open_level - squint_level) * strain_ratio):
                        if self._squint_since is None:
                            self._squint_since = now
                        elif (
                            now - self._squint_since >= SQUINT_ALERT_SECONDS
                            and now - self._last_coach_prompt >= COACH_COOLDOWN_SECONDS
                        ):
                            coach_message = "Looks like you are straining. Take a short blink break or recalibrate."
                            self._last_coach_prompt = now
                    else:
                        self._squint_since = None

                    if (
                        not coach_message
                        and (now - self._last_blink_time) >= BLINK_IDLE_SECONDS
                        and (now - self._last_coach_prompt) >= COACH_COOLDOWN_SECONDS
                    ):
                        coach_message = "Try a gentle blink to keep your eyes comfortable."
                        self._last_coach_prompt = now
                        self._last_blink_time = now
                    elif (
                        low_light
                        and not coach_message
                        and (now - self._last_coach_prompt) >= COACH_COOLDOWN_SECONDS
                    ):
                        coach_message = "Room lighting is very low. Consider night mode or a gentle lamp to help tracking."
                        self._last_coach_prompt = now

                    if locked:
                        brightness_value = self._locked_values["brightness"]
                        font_size = self._locked_values["font_size"]
                        display_openness = self._locked_values["openness"]
                        status_message = "Comfort hold active"
                    else:
                        target_font_size = int(map_range(avg_openness, squint_level, open_level, 60, MIN_FONT_SIZE))
                        target_font_size = max(MIN_FONT_SIZE, min(MAX_FONT_SIZE, target_font_size))
                        if self._smoothed_font is None:
                            self._smoothed_font = float(target_font_size)
                        self._smoothed_font = (
                            (1.0 - FONT_SMOOTH_FACTOR) * self._smoothed_font
                            + FONT_SMOOTH_FACTOR * target_font_size
                        )
                        font_size = int(self._smoothed_font)

                        target_brightness = int(map_range(avg_openness, squint_level, open_level, 100, 25))
                        target_brightness = max(10, min(target_brightness, 100))

                        if auto_brightness:
                            ambient_adjust = int(map_range(ambient_level, 0, 100, -10, 10))
                            target_brightness = max(10, min(100, target_brightness + ambient_adjust))
                            if low_light:
                                target_brightness = max(15, target_brightness)

                        if auto_brightness:
                            if self._smoothed_brightness is None:
                                self._smoothed_brightness = float(target_brightness)
                            self._smoothed_brightness = (
                                (1.0 - BRIGHTNESS_SMOOTH_FACTOR) * self._smoothed_brightness
                                + BRIGHTNESS_SMOOTH_FACTOR * target_brightness
                            )
                            brightness_value = int(self._smoothed_brightness)
                            self._apply_brightness(brightness_value)
                            status_message = "Adaptive brightness active"
                        else:
                            brightness_value = manual_brightness
                            self._smoothed_brightness = float(brightness_value)
                            status_message = "Manual brightness mode"

                        display_openness = avg_openness
                        self._locked_values.update(
                            {
                                "brightness": brightness_value,
                                "font_size": font_size,
                                "openness": display_openness,
                            }
                        )

                    comfort_score = int(map_range(display_openness, squint_level, open_level, 40, 100))
                    comfort_score = max(0, min(100, comfort_score))
                    self._latest_font_size = font_size
                    self._current_brightness = brightness_value

                    self._draw_eye_guides(display_frame, landmarks)
                    cv2.putText(
                        display_frame,
                        f"Openness {display_openness:.2f}",
                        (16, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 200),
                        2,
                    )
                    cv2.putText(
                        display_frame,
                        f"Comfort {comfort_score}",
                        (16, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    if blink_detected:
                        cv2.putText(
                            display_frame,
                            "Blink detected",
                            (16, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 200, 255),
                            2,
                        )
                else:
                    elapsed = time.time() - self._last_face_time
                    status_message = "Face not detected"
                    if auto_brightness and elapsed >= FACE_MISSING_DIM_DELAY and not self._away_dimmed:
                        dimmed_value = FACE_MISSING_BRIGHTNESS
                        if self._apply_brightness(dimmed_value, force=True):
                            self._current_brightness = dimmed_value
                        self._away_dimmed = True
                        status_message = "User away - dimming display"
                    current_openness = 0.0
                    comfort_score = 0
                    self._squint_since = None
                    coach_message = ""
                    self._last_blink_time = time.time()

                metrics = EyeTrackingMetrics(
                    frame=display_frame,
                    openness=current_openness,
                    brightness=int(self._current_brightness),
                    font_size=int(self._latest_font_size),
                    comfort_score=int(comfort_score),
                    locked=locked,
                    status=status_message,
                    blink_detected=blink_detected,
                    auto_brightness=auto_brightness,
                    coaching_message=coach_message,
                    ambient_level=float(ambient_level),
                )
                callback(metrics)
                time.sleep(0.005)
        finally:
            cap.release()
            face_mesh.close()

    def _apply_profile(self, profile_name: str) -> None:
        profile = self._config["profiles"][profile_name]
        self._open_level = float(profile["open"])
        self._squint_level = float(profile["squint"])
        self._update_thresholds()

        comfort = profile.get("comfort", {})
        brightness = int(comfort.get("brightness", DEFAULT_BRIGHTNESS))
        font_size = int(comfort.get("font_size", DEFAULT_FONT_SIZE))

        self._locked_values = {
            "brightness": brightness,
            "font_size": font_size,
            "openness": self._open_level,
        }
        self._manual_brightness = brightness
        self._current_brightness = brightness
        self._latest_font_size = font_size
        self._smoothed_brightness = None
        self._smoothed_font = None
        self._last_brightness = None
        self._last_brightness_ts = 0.0
        self._away_dimmed = False
        self._locked = False

    def _update_thresholds(self) -> None:
        self._blink_threshold = max(self._squint_level * BLINK_RATIO, self._squint_level - 1.0)

    def _iris_openness(self, landmarks, image_shape) -> float:
        height, width = image_shape
        left_top = np.array([landmarks[LEFT_EYE_TOP].x * width, landmarks[LEFT_EYE_TOP].y * height])
        left_bottom = np.array([landmarks[LEFT_EYE_BOTTOM].x * width, landmarks[LEFT_EYE_BOTTOM].y * height])
        right_top = np.array([landmarks[RIGHT_EYE_TOP].x * width, landmarks[RIGHT_EYE_TOP].y * height])
        right_bottom = np.array([landmarks[RIGHT_EYE_BOTTOM].x * width, landmarks[RIGHT_EYE_BOTTOM].y * height])

        left_openness = np.linalg.norm(left_top - left_bottom)
        right_openness = np.linalg.norm(right_top - right_bottom)
        return float((left_openness + right_openness) / 2.0)

    def _draw_eye_guides(self, frame, landmarks) -> None:
        height, width = frame.shape[:2]
        for top_idx, bottom_idx, color in (
            (LEFT_EYE_TOP, LEFT_EYE_BOTTOM, (0, 200, 255)),
            (RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, (0, 200, 255)),
        ):
            top = (
                int(landmarks[top_idx].x * width),
                int(landmarks[top_idx].y * height),
            )
            bottom = (
                int(landmarks[bottom_idx].x * width),
                int(landmarks[bottom_idx].y * height),
            )
            cv2.line(frame, top, bottom, color, 2)
            cv2.circle(frame, top, 3, color, -1)
            cv2.circle(frame, bottom, 3, color, -1)

    def _apply_brightness(self, value: int, force: bool = False) -> bool:
        if sbc is None:
            return False

        value = int(max(10, min(100, value)))
        now = time.time()
        should_send = force
        if not should_send:
            if self._last_brightness is None:
                should_send = True
            elif abs(value - self._last_brightness) >= BRIGHTNESS_HYSTERESIS:
                should_send = True
            elif (now - self._last_brightness_ts) >= BRIGHTNESS_UPDATE_INTERVAL:
                should_send = True

        if not should_send:
            return False

        try:
            sbc.set_brightness(value)
            self._last_brightness = value
            self._last_brightness_ts = now
            return True
        except Exception as exc:  # pragma: no cover - hardware dependent
            if not self._brightness_error_logged:
                print(f"Brightness control skipped: {exc}")
                self._brightness_error_logged = True
            return False

    def _update_ambient_level(self, frame) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        level = float(np.mean(gray)) / 255.0 * 100.0
        if not np.isfinite(level):
            return self._ambient_level
        self._ambient_level = (
            (1.0 - AMBIENT_SMOOTH_FACTOR) * self._ambient_level + AMBIENT_SMOOTH_FACTOR * level
        )
        return self._ambient_level


def start_eye_tracking():
    tracker = EyeTracker()
    stop_event = threading.Event()

    def printer(metrics: EyeTrackingMetrics):
        print(
            f"openness={metrics.openness:.2f} brightness={metrics.brightness} "
            f"font={metrics.font_size} comfort={metrics.comfort_score} status={metrics.status} "
            f"coach={metrics.coaching_message} ambient={metrics.ambient_level:.1f}"
        )

    try:
        tracker.run(printer, stop_event)
    except KeyboardInterrupt:
        stop_event.set()


__all__ = [
    "CALIBRATION_PATH",
    "DEFAULT_PROFILE_NAME",
    "EyeTracker",
    "EyeTrackingMetrics",
    "read_calibration_config",
    "write_calibration_config",
    "start_eye_tracking",
]


if __name__ == "__main__":
    start_eye_tracking()
