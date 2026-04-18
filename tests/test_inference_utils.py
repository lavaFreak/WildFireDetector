from pathlib import Path

import cv2
import numpy as np

from src.inference import collect_image_paths, preprocess_grayscale_image


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((20, 20, 3), value, dtype=np.uint8)
    ok = cv2.imwrite(str(path), image)
    assert ok


def test_collect_image_paths_accepts_files_and_directories(tmp_path: Path) -> None:
    first = tmp_path / "one.png"
    second = tmp_path / "nested" / "two.jpg"
    _write_image(first, 20)
    _write_image(second, 220)

    paths = collect_image_paths([first, tmp_path / "nested"])

    assert paths == [first.resolve(), second.resolve()]


def test_preprocess_grayscale_image_returns_normalized_tensor_shape(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    _write_image(image_path, 128)

    arr = preprocess_grayscale_image(image_path, size=16)

    assert arr.shape == (1, 1, 16, 16)
    assert arr.dtype == np.float32
    assert 0.0 <= float(arr.min()) <= 1.0
    assert 0.0 <= float(arr.max()) <= 1.0
