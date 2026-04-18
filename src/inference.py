from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINTS = {
    "cnn_16x16": REPO_ROOT / "figures" / "cnn_16x16" / "best_model.pt",
    "cnn_64x64": REPO_ROOT / "figures" / "cnn_64x64" / "best_model.pt",
}
DEFAULT_THRESHOLD = 0.5


@dataclass
class LoadedClassifier:
    name: str
    size: int
    checkpoint_path: Path
    device: str
    model: Any


def is_image_file(path: str | Path) -> bool:
    return Path(path).is_file() and Path(path).suffix.lower() in IMAGE_EXTS


def collect_image_paths(inputs: Iterable[str | Path]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()

    for raw in inputs:
        path = Path(raw).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")

        if path.is_file():
            if not is_image_file(path):
                raise ValueError(f"Expected an image file, got: {path}")
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                paths.append(resolved)
            continue

        for candidate in sorted(path.rglob("*")):
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTS:
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    paths.append(resolved)

    if not paths:
        raise ValueError("No image files found in the provided inputs.")

    return paths


def preprocess_grayscale_image(image_path: str | Path, size: int) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    arr = img.astype(np.float32) / 255.0
    return arr.reshape(1, 1, size, size)


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for CNN inference. Install torch in your environment first."
        ) from exc

    return torch, nn


def _resolve_device(preferred: str, torch: Any) -> str:
    if preferred != "auto":
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _build_legacy_cnn(size: int) -> Any:
    _, nn = _require_torch()
    feature_side = size // 4
    flattened_dim = 32 * feature_side * feature_side

    class LegacyWildfireCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.fc = nn.Sequential(
                nn.Linear(flattened_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 1),
            )

        def forward(self, x: Any) -> Any:
            z = self.conv(x)
            z = z.view(z.size(0), -1)
            return self.fc(z).squeeze(1)

    return LegacyWildfireCNN()


def load_classifier(
    model_name: str = "cnn_64x64",
    *,
    checkpoint_path: str | Path | None = None,
    device: str = "auto",
) -> LoadedClassifier:
    torch, _ = _require_torch()
    if model_name not in DEFAULT_CHECKPOINTS:
        supported = ", ".join(sorted(DEFAULT_CHECKPOINTS))
        raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}")

    size = 64 if model_name == "cnn_64x64" else 16
    ckpt_path = Path(checkpoint_path) if checkpoint_path is not None else DEFAULT_CHECKPOINTS[model_name]
    ckpt_path = ckpt_path.expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    resolved_device = _resolve_device(device, torch)
    model = _build_legacy_cnn(size)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    model.to(torch.device(resolved_device))

    return LoadedClassifier(
        name=model_name,
        size=size,
        checkpoint_path=ckpt_path,
        device=resolved_device,
        model=model,
    )


def predict_with_classifier(
    classifier: LoadedClassifier,
    image_paths: Iterable[str | Path],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    tta: bool = False,
) -> list[dict[str, Any]]:
    torch, _ = _require_torch()

    resolved_paths = [Path(path).expanduser().resolve() for path in image_paths]
    batch_np = np.concatenate(
        [preprocess_grayscale_image(path, classifier.size) for path in resolved_paths],
        axis=0,
    )
    batch = torch.from_numpy(batch_np).to(torch.device(classifier.device))

    with torch.no_grad():
        logits = classifier.model(batch)
        probs = torch.sigmoid(logits)

        if tta:
            flipped_np = np.flip(batch_np, axis=3).copy()
            flipped = torch.from_numpy(flipped_np).to(torch.device(classifier.device))
            flipped_logits = classifier.model(flipped)
            flipped_probs = torch.sigmoid(flipped_logits)
            probs = (probs + flipped_probs) / 2.0

    probs_np = probs.detach().cpu().numpy()
    results: list[dict[str, Any]] = []
    for path, prob in zip(resolved_paths, probs_np, strict=True):
        prob_value = float(prob)
        label = "fire" if prob_value >= threshold else "no_fire"
        confidence = prob_value if label == "fire" else 1.0 - prob_value
        results.append(
            {
                "path": str(path),
                "model": classifier.name,
                "fire_probability": prob_value,
                "predicted_label": label,
                "confidence": float(confidence),
            }
        )

    return results


def ensemble_predict(
    image_paths: Iterable[str | Path],
    *,
    model_names: Iterable[str] = ("cnn_16x16", "cnn_64x64"),
    threshold: float = DEFAULT_THRESHOLD,
    device: str = "auto",
    tta: bool = False,
) -> list[dict[str, Any]]:
    paths = [Path(path).expanduser().resolve() for path in image_paths]
    model_names = list(model_names)
    if not model_names:
        raise ValueError("Ensemble prediction requires at least one model.")

    per_model_results = [
        predict_with_classifier(
            load_classifier(model_name, device=device),
            paths,
            threshold=threshold,
            tta=tta,
        )
        for model_name in model_names
    ]

    combined: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        probs = [results[index]["fire_probability"] for results in per_model_results]
        avg_prob = float(sum(probs) / len(probs))
        label = "fire" if avg_prob >= threshold else "no_fire"
        confidence = avg_prob if label == "fire" else 1.0 - avg_prob
        combined.append(
            {
                "path": str(path),
                "model": "+".join(model_names),
                "fire_probability": avg_prob,
                "predicted_label": label,
                "confidence": float(confidence),
            }
        )

    return combined
