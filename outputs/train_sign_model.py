#!/usr/bin/env python3
"""Train a small-vocabulary sign classifier from the recorder's JSONL clips."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sign_features import FEATURES_PER_HAND, N_HANDS, N_STEPS, read_clip
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_classifier(
    n_components: int = 12,
    C: float = 2.0,
    kernel: str = "rbf",
    gamma: str | float = "scale",
) -> object:
    # PCA reduces the many correlated coordinates before the RBF SVM.
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components, svd_solver="randomized", random_state=42),
        SVC(C=C, kernel=kernel, gamma=gamma, class_weight="balanced", random_state=42),
    )


def augment_features(vector: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Make one mildly perturbed copy of a clip's fixed-length landmark sequence."""
    frame_width = N_HANDS * FEATURES_PER_HAND
    pose_size = N_STEPS * frame_width
    pose = vector[:pose_size].reshape(N_STEPS, N_HANDS, FEATURES_PER_HAND).copy()
    timeline = np.linspace(0.0, 1.0, N_STEPS, dtype=np.float32)

    # A small time warp changes speed while keeping the clip's start and end.
    query = timeline ** rng.uniform(0.92, 1.08)
    warped = np.empty_like(pose)
    for hand_i in range(N_HANDS):
        for feature_i in range(FEATURES_PER_HAND):
            warped[:, hand_i, feature_i] = np.interp(
                query, timeline, pose[:, hand_i, feature_i]
            )
    pose = warped

    # Rotate and scale hand pose slightly in the image plane; shift the wrist trajectory.
    angle = np.deg2rad(rng.uniform(-5.0, 5.0))
    scale = rng.uniform(0.97, 1.03)
    cosine, sine = np.cos(angle), np.sin(angle)
    for hand_i in range(N_HANDS):
        relative = pose[:, hand_i, :63].reshape(N_STEPS, 21, 3)
        x = relative[:, :, 0].copy()
        y = relative[:, :, 1].copy()
        relative[:, :, 0] = cosine * x - sine * y
        relative[:, :, 1] = sine * x + cosine * y
        relative *= scale

        wrist = pose[:, hand_i, 63:66]
        present = pose[:, hand_i, 66] > 0.05
        wrist[present, 0] = np.clip(wrist[present, 0] + rng.uniform(-0.015, 0.015), 0.0, 1.0)
        wrist[present, 1] = np.clip(wrist[present, 1] + rng.uniform(-0.015, 0.015), 0.0, 1.0)

        # Small sensor-like landmark noise and smooth micro-variation in relative depth.
        noise = rng.normal(0.0, [0.0015, 0.0015, 0.002], size=relative.shape)
        relative += noise.astype(np.float32) * present[:, None, None]
        relative[:, 0, :] = 0.0  # wrist remains the local origin
        amplitude = rng.uniform(0.0, 0.0025)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        relative[:, 1:, 2] += (
            amplitude * np.sin(2.0 * np.pi * timeline + phase) * present
        )[:, None]

    pose_vector = pose.reshape(-1)
    motion = np.diff(pose, axis=0, prepend=pose[:1]).reshape(-1)
    return np.concatenate([pose_vector, motion]).astype(np.float32)


def cross_validated_predictions(
    X: np.ndarray, y: np.ndarray, augmentation_copies: int
) -> tuple[np.ndarray, np.ndarray]:
    baseline = np.empty(y.shape, dtype=y.dtype)
    augmented = np.empty(y.shape, dtype=y.dtype)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_indices, test_indices) in enumerate(cv.split(X, y)):
        X_train, y_train = X[train_indices], y[train_indices]
        baseline_model = make_classifier().fit(X_train, y_train)
        baseline[test_indices] = baseline_model.predict(X[test_indices])

        if augmentation_copies:
            rng = np.random.default_rng(4000 + fold)
            augmented_x = [sample for sample in X_train]
            augmented_y = [label for label in y_train]
            for sample, label in zip(X_train, y_train):
                for _ in range(augmentation_copies):
                    augmented_x.append(augment_features(sample, rng))
                    augmented_y.append(label)
            X_augmented = np.stack(augmented_x)
            y_augmented = np.asarray(augmented_y)
        else:
            X_augmented, y_augmented = X_train, y_train
        augmented_model = make_classifier().fit(X_augmented, y_augmented)
        augmented[test_indices] = augmented_model.predict(X[test_indices])
    return baseline, augmented


def main() -> None:
    default_data = Path(__file__).with_name("hand_dataset")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=default_data, help="folder containing .jsonl clips")
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("sign_model_augmented.joblib"))
    parser.add_argument("--augmentation-copies", type=int, default=3,
                        help="synthetic copies per training clip in each fold (default: 3)")
    args = parser.parse_args()
    if args.augmentation_copies < 0:
        raise SystemExit("--augmentation-copies trebuie să fie cel puțin 0.")

    clips: list[np.ndarray] = []
    labels: list[str] = []
    skipped: list[str] = []
    for path in sorted(args.data.glob("*.jsonl")):
        try:
            label, features = read_clip(path)
            clips.append(features)
            labels.append(label)
        except (OSError, ValueError) as exc:
            skipped.append(f"{path.name}: {exc}")

    counts = Counter(labels)
    if len(counts) < 2:
        raise SystemExit("Am nevoie de cel puțin două etichete pentru antrenare.")
    if min(counts.values()) < 5:
        raise SystemExit("Am nevoie de cel puțin 5 mostre pentru fiecare etichetă, ca să pot evalua modelul.")

    X = np.stack(clips)
    y = np.asarray(labels)
    baseline_predictions, augmented_predictions = cross_validated_predictions(
        X, y, args.augmentation_copies
    )

    print(f"Mostre folosite: {len(y)} din {len(list(args.data.glob('*.jsonl')))} fișiere")
    print("Mostre pe etichetă:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")
    print(f"Acuratețe de bază, CV 5 părți: {accuracy_score(y, baseline_predictions):.1%}")
    print(f"Acuratețe cu augmentare, CV 5 părți: {accuracy_score(y, augmented_predictions):.1%}")
    print(f"Acuratețe echilibrată cu augmentare: {balanced_accuracy_score(y, augmented_predictions):.1%}")
    print("\nRaport pe etichete:")
    print(classification_report(y, augmented_predictions, labels=sorted(counts), zero_division=0))
    print("Matrice de confuzie (rând=real, coloană=prezis; ordinea etichetelor:", ", ".join(sorted(counts)), ")")
    print(confusion_matrix(y, augmented_predictions, labels=sorted(counts)))
    if skipped:
        print("\nFișiere ignorate:")
        for item in skipped:
            print(f"  {item}")

    rng = np.random.default_rng(1234)
    augmented_x = [sample for sample in X]
    augmented_y = [label for label in y]
    for sample, label in zip(X, y):
        for _ in range(args.augmentation_copies):
            augmented_x.append(augment_features(sample, rng))
            augmented_y.append(label)
    final_model = make_classifier()
    final_model.fit(np.stack(augmented_x), np.asarray(augmented_y))
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": final_model,
        "labels": sorted(counts),
        "n_steps": N_STEPS,
        "n_hands": N_HANDS,
        "feature_format": "wrist-relative XYZ landmarks + wrist position + detection flag, pose and motion",
        "training_counts": dict(counts),
        "augmentation_copies": args.augmentation_copies,
        "augmentation": "small in-plane rotation/scale/translation, mild landmark jitter, smooth relative-depth variation, slight time warp",
        "baseline_cross_validation_accuracy": float(accuracy_score(y, baseline_predictions)),
        "cross_validation_accuracy": float(accuracy_score(y, augmented_predictions)),
        "cross_validation_balanced_accuracy": float(balanced_accuracy_score(y, augmented_predictions)),
    }, args.model)
    print(f"\nModel antrenat și salvat în: {args.model}")


if __name__ == "__main__":
    main()
