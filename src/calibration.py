from __future__ import annotations

import argparse

import cv2
import mediapipe as mp
import numpy as np

from eye_tracking import (  # noqa: E402
    CALIBRATION_PATH,
    DEFAULT_PROFILE_NAME,
    read_calibration_config,
    write_calibration_config,
)

LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374


def get_iris_openness(landmarks, image_shape):
    height, width = image_shape
    left_top = np.array([landmarks[LEFT_EYE_TOP].x * width, landmarks[LEFT_EYE_TOP].y * height])
    left_bottom = np.array([landmarks[LEFT_EYE_BOTTOM].x * width, landmarks[LEFT_EYE_BOTTOM].y * height])
    right_top = np.array([landmarks[RIGHT_EYE_TOP].x * width, landmarks[RIGHT_EYE_TOP].y * height])
    right_bottom = np.array([landmarks[RIGHT_EYE_BOTTOM].x * width, landmarks[RIGHT_EYE_BOTTOM].y * height])

    left_openness = np.linalg.norm(left_top - left_bottom)
    right_openness = np.linalg.norm(right_top - right_bottom)
    return float((left_openness + right_openness) / 2.0)


def collect_samples(cap, face_mesh, label, sample_frames=40):
    values = []
    for i in range(sample_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()
        cv2.putText(
            display,
            f"{label.upper()} calibration {i + 1}/{sample_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(display, "Press ESC to cancel", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.imshow("Calibration", display)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            values.append(get_iris_openness(landmarks, frame.shape[:2]))

        if cv2.waitKey(1) & 0xFF == 27:
            return None

    return float(np.mean(values)) if values else None


def persist_profile(open_value: float, squint_value: float, profile_name: str, activate: bool):
    config = read_calibration_config()
    target = profile_name.strip() or DEFAULT_PROFILE_NAME
    is_new_profile = target not in config["profiles"]
    profile = config["profiles"].get(target, {})
    comfort = profile.get("comfort") if isinstance(profile, dict) else None

    config["profiles"][target] = {
        "open": float(open_value),
        "squint": float(squint_value),
    }
    if isinstance(comfort, dict):
        config["profiles"][target]["comfort"] = comfort

    if activate or is_new_profile or config["active_profile"] == target:
        config["active_profile"] = target

    config = write_calibration_config(config)
    print(f"Calibration saved for profile '{config['active_profile']}' in {CALIBRATION_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate eye openness thresholds.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index to use")
    parser.add_argument("--frames", type=int, default=40, help="Frames to average per capture")
    parser.add_argument("--profile", type=str, default=None, help="Profile name to update")
    parser.add_argument(
        "--activate",
        action="store_true",
        help="Mark the calibrated profile as active after saving",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = read_calibration_config()
    target_profile = (args.profile or config["active_profile"]).strip() or DEFAULT_PROFILE_NAME

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot access the camera.")

    calibration = {}
    mp_face_mesh = mp.solutions.face_mesh

    print(f"Calibrating profile '{target_profile}'. Press C to capture OPEN, S for SQUINT, Q to save and exit.")

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            display = frame.copy()
            cv2.putText(display, "Press C (open), S (squint), Q (save & quit)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if "open" in calibration:
                cv2.putText(display, f"Open: {calibration['open']:.2f}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
            if "squint" in calibration:
                cv2.putText(display, f"Squint: {calibration['squint']:.2f}", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)

            cv2.imshow("Calibration", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("c"):
                sample = collect_samples(cap, face_mesh, "open", sample_frames=args.frames)
                if sample is not None:
                    calibration["open"] = sample
                    print(f"Open eye average: {sample:.2f}")
            elif key == ord("s"):
                sample = collect_samples(cap, face_mesh, "squint", sample_frames=args.frames)
                if sample is not None:
                    calibration["squint"] = sample
                    print(f"Squint eye average: {sample:.2f}")
            elif key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()

    if "open" in calibration and "squint" in calibration:
        if calibration["open"] <= calibration["squint"]:
            calibration["open"] = calibration["squint"] + 1.0
        persist_profile(calibration["open"], calibration["squint"], target_profile, args.activate)
    else:
        print("Calibration incomplete; nothing saved.")


if __name__ == "__main__":
    main()
