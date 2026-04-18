from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


@dataclass
class ProbabilityMetrics:
    threshold: float
    accuracy: float
    auc: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]


def compute_probability_metrics(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    *,
    threshold: float,
) -> ProbabilityMetrics:
    y = np.asarray(list(y_true), dtype=np.int64)
    probs = np.asarray(list(probabilities), dtype=np.float32)
    preds = (probs >= threshold).astype(np.int64)
    cm = confusion_matrix(y, preds, labels=[0, 1])

    return ProbabilityMetrics(
        threshold=float(threshold),
        accuracy=float(accuracy_score(y, preds)),
        auc=float(roc_auc_score(y, probs)),
        precision=float(precision_score(y, preds, zero_division=0)),
        recall=float(recall_score(y, preds, zero_division=0)),
        f1=float(f1_score(y, preds, zero_division=0)),
        confusion_matrix=cm.astype(int).tolist(),
    )


def find_best_accuracy_threshold(
    y_true: Iterable[int],
    probabilities: Iterable[float],
    *,
    steps: int = 401,
    default_threshold: float = 0.5,
) -> float:
    y = np.asarray(list(y_true), dtype=np.int64)
    probs = np.asarray(list(probabilities), dtype=np.float32)
    candidates = np.linspace(0.0, 1.0, num=steps)

    best_threshold = float(default_threshold)
    best_accuracy = -1.0

    for threshold in candidates:
        preds = (probs >= threshold).astype(np.int64)
        accuracy = float(accuracy_score(y, preds))
        distance_to_default = abs(float(threshold) - default_threshold)
        best_distance = abs(best_threshold - default_threshold)

        if accuracy > best_accuracy + 1e-12:
            best_accuracy = accuracy
            best_threshold = float(threshold)
            continue

        if abs(accuracy - best_accuracy) <= 1e-12 and distance_to_default < best_distance:
            best_threshold = float(threshold)

    return best_threshold


def average_probability_sets(probability_sets: Iterable[Iterable[float]]) -> np.ndarray:
    arrays = [np.asarray(list(probs), dtype=np.float32) for probs in probability_sets]
    if not arrays:
        raise ValueError("Need at least one probability set to average.")

    first_shape = arrays[0].shape
    if any(arr.shape != first_shape for arr in arrays[1:]):
        raise ValueError("All probability sets must have the same shape.")

    return np.mean(np.stack(arrays, axis=0), axis=0)
