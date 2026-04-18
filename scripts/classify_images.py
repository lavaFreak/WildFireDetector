from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.inference import (
    BEST_MODEL_NAME,
    DEFAULT_THRESHOLD,
    collect_image_paths,
    ensemble_predict,
    load_classifier,
    predict_with_classifier,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify one image or a batch of images as fire vs no_fire.")
    parser.add_argument("inputs", nargs="+", help="Image files and/or directories to classify")
    parser.add_argument(
        "--model",
        choices=[BEST_MODEL_NAME, "cnn_64x64", "cnn_16x16", "ensemble"],
        default=BEST_MODEL_NAME,
        help="Classifier to run. 'ensemble' averages the legacy CNN checkpoints.",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Fire classification threshold")
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument("--tta", action="store_true", help="Average prediction with a horizontally flipped pass")
    parser.add_argument("--csv-out", type=Path, help="Optional path to save predictions as CSV")
    return parser


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "model", "predicted_label", "confidence", "fire_probability"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    image_paths = collect_image_paths(args.inputs)

    if args.model == "ensemble":
        rows = ensemble_predict(image_paths, threshold=args.threshold, device=args.device, tta=args.tta)
    else:
        classifier = load_classifier(args.model, device=args.device)
        rows = predict_with_classifier(classifier, image_paths, threshold=args.threshold, tta=args.tta)

    print(f"Classified {len(rows)} image(s)")
    print()
    print(f"{'label':<10} {'confidence':>10} {'fire_prob':>10}  path")
    print("-" * 80)
    for row in rows:
        print(
            f"{row['predicted_label']:<10} "
            f"{row['confidence']:>10.3f} "
            f"{row['fire_probability']:>10.3f}  "
            f"{row['path']}"
        )

    if args.csv_out is not None:
        write_csv(args.csv_out, rows)
        print()
        print(f"Saved CSV predictions to: {args.csv_out}")


if __name__ == "__main__":
    main()
