#!/usr/bin/env python3
"""Run the trained sign model on a saved JSONL clip or a live camera sample."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import cv2
import joblib
import mediapipe as mp

from sign_features import features_from_rows


MODEL_PATH = Path(__file__).with_name("sign_model_finetuned.joblib")
TASK_PATH = Path(__file__).with_name("hand_landmarker.task")
TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14),
    (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def predict_clip(model: object, rows: list[dict]) -> str:
    if len(rows) < 10:
        raise ValueError("mostra prea scurtă; înregistrează gestul complet")
    hand_frames = sum(bool(row.get("hands")) for row in rows)
    if hand_frames / len(rows) < 0.4:
        raise ValueError("mâna a fost detectată în prea puține cadre; încearcă din nou")
    vector = features_from_rows(rows)
    return str(model.predict(vector.reshape(1, -1))[0])


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ensure_task_model() -> Path:
    if not TASK_PATH.exists():
        print("Descarc modelul MediaPipe necesar pentru detectarea mâinilor…")
        urllib.request.urlretrieve(TASK_URL, TASK_PATH)
    return TASK_PATH


def run_camera(model: object, camera_index: int) -> None:
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise SystemExit(
            f"Nu pot deschide camera {camera_index}. Pe Mac, verifică permisiunea pentru Camera în setările Terminalului."
        )

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    Options = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    options = Options(
        base_options=BaseOptions(model_asset_path=str(ensure_task_model())),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    rows: list[dict] = []
    recording = False
    last_prediction = ""
    last_timestamp_ms = -1
    print("Apasă SPACE, fă un singur semn complet, apoi SPACE pentru predicție. Q sau ESC închide.")
    try:
        with HandLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Nu am mai primit cadre de la cameră.")
                    break
                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = max(int(time.monotonic() * 1000), last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms
                result = detector.detect_for_video(image, timestamp_ms)

                hands = []
                for landmarks in result.hand_landmarks:
                    points = [{"x": point.x, "y": point.y, "z": point.z} for point in landmarks]
                    hands.append({"landmarks": points})
                    coords = [(int(point["x"] * width), int(point["y"] * height)) for point in points]
                    for start, end in HAND_CONNECTIONS:
                        cv2.line(frame, coords[start], coords[end], (80, 220, 80), 2)
                    for x, y in coords:
                        cv2.circle(frame, (x, y), 3, (30, 80, 255), -1)

                if recording:
                    rows.append({"hands": hands})
                    state_text = f"ÎNREGISTREZ: {len(rows)} cadre"
                    state_color = (0, 0, 255)
                else:
                    state_text = "SPACE: începe mostra"
                    state_color = (255, 255, 255)
                cv2.putText(frame, state_text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
                cv2.putText(frame, f"Mâini detectate: {len(hands)}", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if last_prediction:
                    cv2.putText(frame, f"Predicție: {last_prediction}", (12, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("Recunoaștere semne", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == 32:
                    if not recording:
                        rows = []
                        recording = True
                        last_prediction = ""
                    else:
                        recording = False
                        try:
                            last_prediction = predict_clip(model, rows)
                            print(f"Predicție: {last_prediction}")
                        except ValueError as exc:
                            last_prediction = str(exc)
                            print(last_prediction)
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--clip", type=Path, help="classify an existing .jsonl sample instead of opening the camera")
    args = parser.parse_args()
    if not args.model.exists():
        raise SystemExit(f"Nu găsesc modelul antrenat: {args.model}")
    artifact = joblib.load(args.model)
    model = artifact["model"]

    if args.clip:
        try:
            print(f"Predicție: {predict_clip(model, read_jsonl(args.clip))}")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise SystemExit(f"Nu am putut clasifica mostra: {exc}") from exc
    else:
        run_camera(model, args.camera)


if __name__ == "__main__":
    main()
