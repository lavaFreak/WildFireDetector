import numpy as np

from src.eval_utils import average_probability_sets, compute_probability_metrics, find_best_accuracy_threshold


def test_average_probability_sets_means_matching_vectors() -> None:
    averaged = average_probability_sets([[0.2, 0.8], [0.4, 0.6]])
    assert np.allclose(averaged, np.array([0.3, 0.7], dtype=np.float32))


def test_find_best_accuracy_threshold_prefers_closest_to_default_on_tie() -> None:
    y_true = [0, 1]
    probs = [0.4, 0.6]

    threshold = find_best_accuracy_threshold(y_true, probs, steps=5, default_threshold=0.5)

    assert threshold == 0.5


def test_compute_probability_metrics_returns_expected_confusion_matrix() -> None:
    metrics = compute_probability_metrics([0, 0, 1, 1], [0.1, 0.7, 0.4, 0.9], threshold=0.5)

    assert metrics.confusion_matrix == [[1, 1], [1, 1]]
    assert metrics.accuracy == 0.5
    assert 0.0 <= metrics.auc <= 1.0
