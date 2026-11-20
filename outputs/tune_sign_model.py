#!/usr/bin/env python3
"""Tune the SVM/PCA settings and train on the current landmark dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from sign_features import read_clip
from train_sign_model import augment_features, make_classifier


def training_fold_with_augmentation(
    X: np.ndarray, y: np.ndarray, indices: np.ndarray, copies: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features: list[np.ndarray] = []
    labels: list[str] = []
    for index in indices:
        features.append(X[index])
        labels.append(str(y[index]))
        for _ in range(copies):
            features.append(augment_features(X[index], rng))
            labels.append(str(y[index]))
    return np.stack(features), np.asarray(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("hand_dataset"))
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("sign_model_finetuned.joblib"))
    parser.add_argument("--copies-per-real", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.copies_per_real < 0 or args.folds < 2:
        raise SystemExit("copies-per-real trebuie să fie cel puțin 0, iar folds cel puțin 2.")

    samples: list[np.ndarray] = []
    labels: list[str] = []
    skipped: list[str] = []
    for path in sorted(args.data.glob("*.jsonl")):
        try:
            label, features = read_clip(path)
            samples.append(features)
            labels.append(label)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            skipped.append(f"{path.name}: {exc}")

    counts = Counter(labels)
    if len(counts) < 2:
        raise SystemExit(f"Am găsit {len(counts)} etichete; sunt necesare cel puțin două în {args.data}.")
    if min(counts.values()) < args.folds:
        raise SystemExit(f"Fiecare etichetă are nevoie de cel puțin {args.folds} clipuri reale pentru validarea în {args.folds} părți.")

    X = np.stack(samples)
    y = np.asarray(labels)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    candidates = []
    for n_components in (8, 12, 16):
        for C in (0.5, 2.0, 8.0):
            candidates.append({"n_components": n_components, "C": C, "kernel": "rbf", "gamma": "scale"})
    for n_components in (8, 12):
        for C in (0.5, 2.0):
            candidates.append({"n_components": n_components, "C": C, "kernel": "linear", "gamma": "scale"})

    # Precompute the exact same fold splits and augmented training copies for every candidate.
    folds = []
    for fold_index, (train_indices, test_indices) in enumerate(cv.split(X, y)):
        X_train, y_train = training_fold_with_augmentation(
            X, y, train_indices, args.copies_per_real, args.seed + fold_index
        )
        folds.append((X_train, y_train, test_indices))

    results = []
    for config in candidates:
        fold_scores = []
        all_true: list[str] = []
        all_predicted: list[str] = []
        for X_train, y_train, test_indices in folds:
            classifier = make_classifier(**config).fit(X_train, y_train)
            predicted = classifier.predict(X[test_indices])
            fold_scores.append(accuracy_score(y[test_indices], predicted))
            all_true.extend(y[test_indices].tolist())
            all_predicted.extend(predicted.tolist())
        results.append({
            **config,
            "mean_accuracy": float(np.mean(fold_scores)),
            "std_accuracy": float(np.std(fold_scores)),
            "balanced_accuracy": float(balanced_accuracy_score(all_true, all_predicted)),
        })

    # Prefer simpler/lower-variance settings when the CV means tie.
    results.sort(key=lambda item: (
        item["mean_accuracy"],
        -item["std_accuracy"],
        -item["n_components"],
        -item["C"],
    ), reverse=True)
    best = results[0]

    print(f"Clipuri reale: {len(y)} | etichete: {len(counts)} | augmentări per clip de antrenare: {args.copies_per_real}")
    print("Clipuri reale per etichetă:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")
    print("\nRezultatele reglării (medie acuratețe CV ± variația între fold-uri):")
    for result in results:
        print(
            f"  PCA={result['n_components']:2}, C={result['C']:>3}, {result['kernel']:6}: "
            f"{result['mean_accuracy']:.1%} ± {result['std_accuracy']:.1%}"
        )
    print(f"\nSetarea aleasă: PCA={best['n_components']}, C={best['C']}, kernel={best['kernel']}, gamma={best['gamma']}")
    print(f"Acuratețe CV a setării alese: {best['mean_accuracy']:.1%}; echilibrată: {best['balanced_accuracy']:.1%}")

    X_final, y_final = training_fold_with_augmentation(
        X, y, np.arange(len(y)), args.copies_per_real, args.seed + 10000
    )
    final_model = make_classifier(**{key: best[key] for key in ("n_components", "C", "kernel", "gamma")})
    final_model.fit(X_final, y_final)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": final_model,
        "labels": sorted(counts),
        "n_steps": 32,
        "n_hands": 2,
        "feature_format": "wrist-relative XYZ landmarks + wrist position + detection flag, pose and motion",
        "training_counts": dict(counts),
        "augmentation_copies_per_real": args.copies_per_real,
        "tuned_parameters": {key: best[key] for key in ("n_components", "C", "kernel", "gamma")},
        "cross_validation_accuracy": best["mean_accuracy"],
        "cross_validation_std": best["std_accuracy"],
        "cross_validation_balanced_accuracy": best["balanced_accuracy"],
    }, args.model)
    print(f"\nModel final salvat în: {args.model}")
    if skipped:
        print("Fișiere ignorate:")
        for item in skipped:
            print(f"  {item}")


if __name__ == "__main__":
    main()
