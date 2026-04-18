from pathlib import Path

import cv2
import numpy as np

from src.data_utils import load_splits


def _write_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), array)
    assert ok


def _build_synthetic_splits(root: Path) -> Path:
    splits_root = root / "splits"
    class_values = {
        "fire": 240,
        "no_fire": 32,
    }

    for split in ("train", "val", "test"):
        for cls_name, value in class_values.items():
            for index in range(2):
                image = np.full((12, 12, 3), value + index, dtype=np.uint8)
                _write_image(splits_root / split / cls_name / f"{cls_name}_{index}.png", image)

    return splits_root


def test_load_splits_grayscale_shapes_and_range(tmp_path: Path) -> None:
    splits_root = _build_synthetic_splits(tmp_path)

    ds = load_splits(splits_root=splits_root, size=8, grayscale=True, shuffle_train=False)

    assert ds.X_train.shape == (4, 64)
    assert ds.X_val.shape == (4, 64)
    assert ds.X_test.shape == (4, 64)
    assert set(ds.y_train.tolist()) == {0, 1}
    assert 0.0 <= ds.X_train.min() <= 1.0
    assert 0.0 <= ds.X_train.max() <= 1.0


def test_load_splits_rgb_shapes(tmp_path: Path) -> None:
    splits_root = _build_synthetic_splits(tmp_path)

    ds = load_splits(splits_root=splits_root, size=6, grayscale=False, shuffle_train=False)

    assert ds.X_train.shape == (4, 6 * 6 * 3)
    assert ds.X_val.shape == (4, 6 * 6 * 3)
    assert ds.X_test.shape == (4, 6 * 6 * 3)


def test_train_shuffle_is_seeded(tmp_path: Path) -> None:
    splits_root = _build_synthetic_splits(tmp_path)

    ds_a = load_splits(splits_root=splits_root, size=8, grayscale=True, shuffle_train=True, seed=7)
    ds_b = load_splits(splits_root=splits_root, size=8, grayscale=True, shuffle_train=True, seed=7)

    assert np.array_equal(ds_a.X_train, ds_b.X_train)
    assert np.array_equal(ds_a.y_train, ds_b.y_train)
