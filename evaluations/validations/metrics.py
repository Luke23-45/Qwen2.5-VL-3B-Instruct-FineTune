from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import pandas as pd


def classification_report(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    labels = sorted(set(y_true) | set(y_pred))
    total = len(y_true)
    correct = sum(1 for expected, predicted in zip(y_true, y_pred) if expected == predicted)

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    precision_values: list[float] = []
    recall_values: list[float] = []

    for label in labels:
        tp = sum(1 for expected, predicted in zip(y_true, y_pred) if expected == label and predicted == label)
        fp = sum(1 for expected, predicted in zip(y_true, y_pred) if expected != label and predicted == label)
        fn = sum(1 for expected, predicted in zip(y_true, y_pred) if expected == label and predicted != label)
        support = sum(1 for expected in y_true if expected == label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return {
        "num_examples": total,
        "num_classes": len(labels),
        "accuracy": correct / total if total else 0.0,
        "macro_precision": sum(precision_values) / len(precision_values) if precision_values else 0.0,
        "macro_recall": sum(recall_values) / len(recall_values) if recall_values else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "class_distribution": dict(Counter(y_true)),
        "prediction_distribution": dict(Counter(y_pred)),
        "per_class": per_class,
    }


def confusion_matrix(y_true: list[str], y_pred: list[str]) -> pd.DataFrame:
    labels = sorted(set(y_true) | set(y_pred))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {label: 0 for label in labels})
    for expected, predicted in zip(y_true, y_pred):
        counts[expected][predicted] += 1
    return pd.DataFrame.from_dict(counts, orient="index", columns=labels).fillna(0).astype(int)
