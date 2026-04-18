from pathlib import Path

import cv2
import numpy as np

from src.inference import collect_image_paths, preprocess_grayscale_image, preprocess_image


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


def test_preprocess_image_uses_channel_first_layout_for_rgb(tmp_path: Path) -> None:
    image_path = tmp_path / "rgb.png"
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[:, :, 0] = 10
    image[:, :, 1] = 20
    image[:, :, 2] = 30
    ok = cv2.imwrite(str(image_path), image)
    assert ok

    arr = preprocess_image(image_path, size=2, grayscale=False, rgb_tensor_layout="chw")

    assert arr.shape == (1, 3, 2, 2)
    assert np.allclose(arr[0, 0], 30 / 255.0)
    assert np.allclose(arr[0, 1], 20 / 255.0)
    assert np.allclose(arr[0, 2], 10 / 255.0)


def test_preprocess_image_supports_legacy_rgb_layout(tmp_path: Path) -> None:
    image_path = tmp_path / "legacy.png"
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[:, :, 0] = 10
    image[:, :, 1] = 20
    image[:, :, 2] = 30
    ok = cv2.imwrite(str(image_path), image)
    assert ok

    arr = preprocess_image(image_path, size=2, grayscale=False, rgb_tensor_layout="legacy_hwc_reshaped")

    assert arr.shape == (1, 3, 2, 2)
    assert arr[0, 0, 0, 0] == 30 / 255.0
    assert arr[0, 0, 0, 1] == 20 / 255.0
