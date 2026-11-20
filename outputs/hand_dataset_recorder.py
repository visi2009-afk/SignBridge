#!/usr/bin/env python3
"""Capture labeled sign-language samples as video + per-frame hand landmarks."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_FILE = Path(__file__).with_name("hand_landmarker.task")
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def ensure_model() -> Path:
    if not MODEL_FILE.exists():
        print("Descarc modelul MediaPipe pentru detectarea mâinilor…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    return MODEL_FILE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Eticheta semnului, ex. salut")
    parser.add_argument("--camera", type=int, default=0, help="Camera (implicit: 0)")
    parser.add_argument("--out", type=Path, default=Path("hand_dataset"))
    parser.add_argument("--max-hands", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(f"Nu pot deschide camera {args.camera}.")

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    Options = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode
    options = Options(
        base_options=BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=RunningMode.VIDEO,
        num_hands=args.max_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("Apasă SPACE pentru start/stop înregistrare; Q sau ESC pentru ieșire.")
    recording = False
    json_file = None
    session_id = None
    frame_index = 0
    start_clock = 0.0
    last_timestamp_ms = -1

    try:
        with HandLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Nu am mai primit cadre de la cameră.")
                    break

                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = max(int(time.monotonic() * 1000), last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms
                result = detector.detect_for_video(mp_image, timestamp_ms)

                hands = []
                for hand_i, landmarks in enumerate(result.hand_landmarks):
                    category = result.handedness[hand_i][0] if hand_i < len(result.handedness) and result.handedness[hand_i] else None
                    world = result.hand_world_landmarks[hand_i] if hand_i < len(result.hand_world_landmarks) else []
                    hands.append({
                        "handedness": category.category_name if category else None,
                        "handedness_score": category.score if category else None,
                        "landmarks": [{"x": p.x, "y": p.y, "z": p.z} for p in landmarks],
                        "world_landmarks": [{"x": p.x, "y": p.y, "z": p.z} for p in world],
                    })
                    points = [(int(p.x * width), int(p.y * height)) for p in landmarks]
                    for a, b in HAND_CONNECTIONS:
                        cv2.line(frame, points[a], points[b], (80, 220, 80), 2)
                    for x, y in points:
                        cv2.circle(frame, (x, y), 3, (30, 80, 255), -1)

                status = "REC" if recording else "PAUZĂ"
                cv2.putText(frame, f"Etichetă: {args.label} | {status} | mâini: {len(hands)}",
                            (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 0, 255) if recording else (255, 255, 255), 2)
                cv2.imshow("Recorder dataset pentru semne", frame)

                if recording:
                    json_file.write(json.dumps({
                        "session_id": session_id,
                        "label": args.label,
                        "frame_index": frame_index,
                        "timestamp_ms": timestamp_ms,
                        "elapsed_ms": int((time.monotonic() - start_clock) * 1000),
                        "image_width": width,
                        "image_height": height,
                        "hands": hands,
                    }, ensure_ascii=False) + "\n")
                    frame_index += 1

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == 32:
                    if not recording:
                        session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        index = 1
                        while (args.out / f"{args.label}{index}.jsonl").exists():
                            index += 1
                        stem = f"{args.label}{index}"
                        json_file = (args.out / f"{stem}.jsonl").open("w", encoding="utf-8")
                        frame_index = 0
                        start_clock = time.monotonic()
                        recording = True
                        print(f"Înregistrez: {stem}")
                    else:
                        recording = False
                        json_file.close()
                        json_file = None
                        print(f"Salvat: {frame_index} cadre în {args.out}")
    finally:
        if json_file is not None:
            json_file.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
