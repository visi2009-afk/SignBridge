"""Shared JSONL/landmark feature conversion used during training and prediction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


N_STEPS = 32
N_HANDS = 2
FEATURES_PER_HAND = 67  # 21 wrist-relative XYZ points, wrist XYZ, detection flag


def features_from_rows(rows: list[dict]) -> np.ndarray:
    if not rows:
        raise ValueError("clip gol")

    frame_values = np.zeros((len(rows), N_HANDS, FEATURES_PER_HAND), dtype=np.float32)
    frame_valid = np.zeros((len(rows), N_HANDS), dtype=bool)
    for frame_i, row in enumerate(rows):
        hands = sorted(
            row.get("hands", []),
            key=lambda hand: np.mean([point["x"] for point in hand.get("landmarks", [])])
            if len(hand.get("landmarks", [])) == 21 else 2.0,
        )[:N_HANDS]
        for hand_i, hand in enumerate(hands):
            points = hand.get("landmarks", [])
            if len(points) != 21:
                continue
            xyz = np.asarray([[point["x"], point["y"], point["z"]] for point in points], dtype=np.float32)
            if not np.isfinite(xyz).all():
                continue
            wrist = xyz[0]
            relative = xyz - wrist
            frame_values[frame_i, hand_i, :63] = relative.reshape(-1)
            frame_values[frame_i, hand_i, 63:66] = wrist
            frame_values[frame_i, hand_i, 66] = 1.0
            frame_valid[frame_i, hand_i] = True

    source_t = np.linspace(0.0, 1.0, len(rows), dtype=np.float32)
    target_t = np.linspace(0.0, 1.0, N_STEPS, dtype=np.float32)
    sampled = np.zeros((N_STEPS, N_HANDS, FEATURES_PER_HAND), dtype=np.float32)
    for hand_i in range(N_HANDS):
        valid = frame_valid[:, hand_i]
        if valid.any():
            for feature_i in range(FEATURES_PER_HAND - 1):
                sampled[:, hand_i, feature_i] = np.interp(
                    target_t, source_t[valid], frame_values[valid, hand_i, feature_i]
                )
        sampled[:, hand_i, -1] = np.interp(
            target_t, source_t, frame_valid[:, hand_i].astype(np.float32)
        )

    pose = sampled.reshape(N_STEPS, -1)
    motion = np.diff(pose, axis=0, prepend=pose[:1])
    return np.concatenate([pose.reshape(-1), motion.reshape(-1)]).astype(np.float32)


def read_clip(path: Path) -> tuple[str, np.ndarray]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("fișier gol")
    label = rows[0].get("label")
    if not label or any(row.get("label") != label for row in rows):
        raise ValueError("etichetă lipsă sau diferită în interiorul fișierului")
    return label, features_from_rows(rows)
