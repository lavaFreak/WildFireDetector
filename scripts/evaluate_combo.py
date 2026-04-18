from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.eval_utils import average_probability_sets, compute_probability_metrics, find_best_accuracy_threshold
from src.inference import load_classifier, predict_with_classifier


def load_split_records(split_dir: Path) -> tuple[list[Path], np.ndarray]:
    records: list[tuple[Path, int]] = []
    for class_name, label in (("no_fire", 0), ("fire", 1)):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
                records.append((image_path.resolve(), label))

    paths = [path for path, _ in records]
    labels = np.asarray([label for _, label in records], dtype=np.int64)
    return paths, labels


def evaluate_paths(paths: list[Path], labels: np.ndarray, *, checkpoints: list[Path], device: str, tta: bool) -> np.ndarray:
    probability_sets = []
    for checkpoint in checkpoints:
        classifier = load_classifier(
            model_name=checkpoint.parent.name,
            checkpoint_path=checkpoint,
            device=device,
        )
        predictions = predict_with_classifier(classifier, paths, tta=tta)
        probability_sets.append([row["fire_probability"] for row in predictions])

    return average_probability_sets(probability_sets)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Short name for the evaluation artifact")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        type=Path,
        help="Checkpoint(s) to evaluate; specify multiple times to average them as an ensemble",
    )
    parser.add_argument("--splits-root", type=Path, default=Path("data/splits"))
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--tta", action="store_true", help="Use horizontal-flip test-time augmentation")
    args = parser.parse_args()

    checkpoints = [path.expanduser().resolve() for path in args.checkpoint]
    for checkpoint in checkpoints:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    val_paths, y_val = load_split_records(args.splits_root / "val")
    test_paths, y_test = load_split_records(args.splits_root / "test")

    val_probs = evaluate_paths(val_paths, y_val, checkpoints=checkpoints, device=args.device, tta=args.tta)
    test_probs = evaluate_paths(test_paths, y_test, checkpoints=checkpoints, device=args.device, tta=args.tta)

    tuned_threshold = find_best_accuracy_threshold(y_val, val_probs)
    val_metrics_default = compute_probability_metrics(y_val, val_probs, threshold=0.5)
    val_metrics_tuned = compute_probability_metrics(y_val, val_probs, threshold=tuned_threshold)
    test_metrics_default = compute_probability_metrics(y_test, test_probs, threshold=0.5)
    test_metrics_tuned = compute_probability_metrics(y_test, test_probs, threshold=tuned_threshold)

    output = {
        "name": args.name,
        "checkpoints": [str(path) for path in checkpoints],
        "tta": bool(args.tta),
        "splits_root": str(args.splits_root),
        "val": {
            "default_threshold": val_metrics_default.__dict__,
            "tuned_threshold": val_metrics_tuned.__dict__,
        },
        "test": {
            "default_threshold": test_metrics_default.__dict__,
            "tuned_threshold": test_metrics_tuned.__dict__,
        },
    }

    out_dir = Path("results") / "evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.name}.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n")

    print(json.dumps(output, indent=2))
    print(f"\nSaved evaluation to: {out_path}")


if __name__ == "__main__":
    main()
