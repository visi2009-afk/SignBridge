#!/usr/bin/env python3
"""Train with five real examples per sign and four synthetic variations per example."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from train_sign_model import augment_features, make_classifier
from sign_features import read_clip


def choose_training_set(
    by_label: dict[str, list[tuple[str, np.ndarray]]],
    examples_per_label: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[str, np.ndarray]], list[tuple[str, np.ndarray]]]:
    training: list[tuple[str, np.ndarray]] = []
    testing: list[tuple[str, np.ndarray]] = []
    for label in sorted(by_label):
        items = by_label[label]
        selected_indices = set(rng.choice(len(items), size=examples_per_label, replace=False).tolist())
        training.extend((label, vector) for i, (_, vector) in enumerate(items) if i in selected_indices)
        testing.extend((label, vector) for i, (_, vector) in enumerate(items) if i not in selected_indices)
    return training, testing


def augment_training(
    training: list[tuple[str, np.ndarray]], copies: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    vectors: list[np.ndarray] = []
    labels: list[str] = []
    for label, vector in training:
        vectors.append(vector)
        labels.append(label)
        for _ in range(copies):
            vectors.append(augment_features(vector, rng))
            labels.append(label)
    return np.stack(vectors), np.asarray(labels)


def main() -> None:
    default_data = Path(__file__).with_name("hand_dataset")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("sign_model_5x5.joblib"))
    parser.add_argument("--real-per-label", type=int, default=5)
    parser.add_argument("--copies-per-real", type=int, default=4)
    parser.add_argument("--evaluation-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.real_per_label < 1 or args.copies_per_real < 0 or args.evaluation_trials < 1:
        raise SystemExit("Folosește cel puțin o mostră reală și un trial de evaluare; variațiile pot fi 0 sau mai multe.")

    by_label: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    skipped: list[str] = []
    for path in sorted(args.data.glob("*.jsonl")):
        try:
            label, vector = read_clip(path)
            by_label[label].append((path.name, vector))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            skipped.append(f"{path.name}: {exc}")

    if len(by_label) < 2:
        raise SystemExit("Am nevoie de cel puțin două etichete.")
    if any(len(items) < args.real_per_label for items in by_label.values()):
        counts = ", ".join(f"{label}: {len(items)}" for label, items in sorted(by_label.items()))
        raise SystemExit(f"Unele etichete au mai puține mostre decât --real-per-label={args.real_per_label}. Găsite: {counts}")

    total_original = sum(map(len, by_label.values()))
    print(f"Etichete: {len(by_label)} | clipuri reale disponibile: {total_original}")
    for label, items in sorted(by_label.items()):
        print(f"  {label}: {len(items)} reale")

    can_evaluate = all(len(items) > args.real_per_label for items in by_label.values())
    accuracy_scores: list[float] = []
    balanced_scores: list[float] = []
    if can_evaluate:
        for trial in range(args.evaluation_trials):
            rng = np.random.default_rng(args.seed + trial)
            training, testing = choose_training_set(by_label, args.real_per_label, rng)
            X_train, y_train = augment_training(
                training, args.copies_per_real, args.seed + 10000 + trial
            )
            X_test = np.stack([vector for _, vector in testing])
            y_test = np.asarray([label for label, _ in testing])
            classifier = make_classifier().fit(X_train, y_train)
            predictions = classifier.predict(X_test)
            accuracy_scores.append(accuracy_score(y_test, predictions))
            balanced_scores.append(balanced_accuracy_score(y_test, predictions))
        print(
            f"Evaluare pe clipuri reale nefolosite la antrenare ({args.evaluation_trials} alegeri): "
            f"{np.mean(accuracy_scores):.1%} ± {np.std(accuracy_scores):.1%}"
        )
        print(f"Acuratețe echilibrată medie: {np.mean(balanced_scores):.1%}")
    else:
        print("Nu există mostre reale în plus pentru evaluare; antrenez modelul, dar scorul de test nu poate fi calculat.")

    final_rng = np.random.default_rng(args.seed + 50000)
    final_training, _ = choose_training_set(by_label, args.real_per_label, final_rng)
    X_final, y_final = augment_training(final_training, args.copies_per_real, args.seed + 60000)
    final_model = make_classifier().fit(X_final, y_final)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": final_model,
        "labels": sorted(by_label),
        "n_steps": 32,
        "n_hands": 2,
        "feature_format": "wrist-relative XYZ landmarks + wrist position + detection flag, pose and motion",
        "real_samples_per_label": args.real_per_label,
        "augmentation_copies_per_real": args.copies_per_real,
        "training_counts": {label: args.real_per_label for label in sorted(by_label)},
        "cross_validation_accuracy": float(np.mean(accuracy_scores)) if accuracy_scores else None,
        "cross_validation_balanced_accuracy": float(np.mean(balanced_scores)) if balanced_scores else None,
        "selection_seed": args.seed + 50000,
    }, args.model)
    print(
        f"Model salvat: {args.model} | {args.real_per_label} reale + "
        f"{args.copies_per_real} variații × {len(by_label)} semne"
    )
    if skipped:
        print("Fișiere ignorate:")
        for item in skipped:
            print(f"  {item}")


if __name__ == "__main__":
    main()
