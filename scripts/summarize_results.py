from __future__ import annotations

import json
from pathlib import Path


RUNS_DIR = Path("results")
OUT_TABLE_BASIC = RUNS_DIR / "summary_table.md"
OUT_TABLE_METRICS = RUNS_DIR / "summary_metrics.md"


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def load_runs():
    rows = []

    if not RUNS_DIR.exists():
        print("No results/ directory found.")
        return rows

    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue

        result_file = run_dir / "results.json"
        if not result_file.exists():
            continue

        with open(result_file, "r") as f:
            res = json.load(f)

        cfg = res.get("config", {})
        model = cfg.get("model", "UNKNOWN")
        size = cfg.get("size", None)
        variant = cfg.get("variant", run_dir.name)

        test = res.get("test", {})
        cm = test.get("confusion_matrix", None)
        if cm is None:
            continue

        tn, fp = cm[0]
        fn, tp = cm[1]

        # Convert to strings like "64x64" for display
        if isinstance(size, int):
            resolution = f"{size}x{size}"
        else:
            # fallback from folder name, e.g. cnn_64x64
            resolution = run_dir.name.split("_")[-1]

        accuracy = float(test.get("accuracy", 0.0))
        auc = float(test.get("auc", 0.0))

        # Metrics (positive class = fire = label 1)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)         # TPR
        f1 = safe_div(2 * precision * recall, precision + recall)

        rows.append(
            {
                "run": run_dir.name,
                "model": model,
                "variant": variant,
                "resolution": resolution,
                "accuracy": accuracy,
                "auc": auc,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return rows


def make_basic_table(rows):
    lines = []
    lines.append("| Model | Variant | Resolution | Accuracy | AUC | FP | FN |")
    lines.append("|-------|---------|------------|----------|-----|----|----|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['variant']} | {r['resolution']} | "
            f"{r['accuracy']:.3f} | {r['auc']:.3f} | "
            f"{r['fp']} | {r['fn']} |"
        )
    return "\n".join(lines)


def make_metrics_table(rows):
    lines = []
    lines.append("| Model | Variant | Resolution | Precision | Recall | F1 | TP | TN | FP | FN |")
    lines.append("|-------|---------|------------|-----------|--------|----|----|----|----|----|")
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['variant']} | {r['resolution']} | "
            f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
            f"{r['tp']} | {r['tn']} | {r['fp']} | {r['fn']} |"
        )
    return "\n".join(lines)


def main():
    rows = load_runs()

    if not rows:
        print("No results.json files found under results/")
        return

    # Optional: sort by model then resolution size
    def size_key(res_str: str) -> int:
        # "64x64" -> 64
        try:
            return int(res_str.split("x")[0])
        except Exception:
            return 0

    rows = sorted(rows, key=lambda r: (r["model"], size_key(r["resolution"]), r["variant"]))

    basic = make_basic_table(rows)
    metrics = make_metrics_table(rows)

    print(basic)
    print("\n" + metrics)

    RUNS_DIR.mkdir(exist_ok=True)

    OUT_TABLE_BASIC.write_text(basic + "\n")
    OUT_TABLE_METRICS.write_text(metrics + "\n")

    print(f"\nSaved basic table to: {OUT_TABLE_BASIC}")
    print(f"Saved metrics table to: {OUT_TABLE_METRICS}")


if __name__ == "__main__":
    main()
