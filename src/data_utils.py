from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


@dataclass
class DatasetSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def _list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def _load_and_preprocess(
    img_path: Path,
    size: int,
    grayscale: bool = True
) -> np.ndarray:
    """
    Returns a float32 array normalized to [0,1].
    - grayscale=True -> shape (size, size)
    - grayscale=False -> shape (size, size, 3) in RGB order
    """
    if grayscale:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        x = img.astype(np.float32) / 255.0
        return x
    else:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        x = img.astype(np.float32) / 255.0
        return x


def _build_xy(
    split_dir: Path,
    size: int,
    grayscale: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    split_dir points to e.g. data/splits/train
    expects:
      split_dir/fire/
      split_dir/no_fire/
    """
    class_map = {"no_fire": 0, "fire": 1}

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for cls_name, y in class_map.items():
        cls_dir = split_dir / cls_name
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {cls_dir}")
        paths = _list_images(cls_dir)
        for p in paths:
            arr = _load_and_preprocess(p, size=size, grayscale=grayscale)
            X_list.append(arr.reshape(-1))  # flatten
            y_list.append(y)

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def load_splits(
    splits_root: str | Path = "data/splits",
    extra_train_roots: Iterable[str | Path] | None = None,
    size: int = 16,
    grayscale: bool = True,
    shuffle_train: bool = True,
    seed: int = 42
) -> DatasetSplits:
    """
    Loads train/val/test arrays from the canonical folder structure.

    Returns:
      X_* shape: (N, d) where d = size*size (gray) or size*size*3 (RGB)
      y_* shape: (N,)
    """
    splits_root = Path(splits_root)
    train_dir = splits_root / "train"
    val_dir = splits_root / "val"
    test_dir = splits_root / "test"

    X_train, y_train = _build_xy(train_dir, size=size, grayscale=grayscale)
    X_val, y_val = _build_xy(val_dir, size=size, grayscale=grayscale)
    X_test, y_test = _build_xy(test_dir, size=size, grayscale=grayscale)

    if extra_train_roots:
        extra_x_train: list[np.ndarray] = [X_train]
        extra_y_train: list[np.ndarray] = [y_train]
        for extra_root in extra_train_roots:
            extra_train_dir = Path(extra_root) / "train"
            X_extra, y_extra = _build_xy(extra_train_dir, size=size, grayscale=grayscale)
            extra_x_train.append(X_extra)
            extra_y_train.append(y_extra)

        X_train = np.concatenate(extra_x_train, axis=0)
        y_train = np.concatenate(extra_y_train, axis=0)

    if shuffle_train:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(y_train))
        X_train, y_train = X_train[idx], y_train[idx]

    return DatasetSplits(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test
    )
# data loading + feature extraction
