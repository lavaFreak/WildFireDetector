from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

from src.data_utils import load_splits


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_confusion_matrix(cm: np.ndarray, out_path: Path, title: str) -> None:
    fig = plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["no_fire", "fire"])
    plt.yticks([0, 1], ["no_fire", "fire"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str) -> float:
    auc = float(roc_auc_score(y_true, y_prob))
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"{title} (AUC={auc:.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return auc


def eval_split(model: LogisticRegression, X: np.ndarray, y: np.ndarray, fig_dir: Path, name: str) -> dict:
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    acc = float(accuracy_score(y, preds))
    auc = float(roc_auc_score(y, probs))
    cm = confusion_matrix(y, preds)

    save_confusion_matrix(cm, fig_dir / f"{name.lower()}_confusion_matrix.png", f"{name} Confusion Matrix")
    save_roc_curve(y, probs, fig_dir / f"{name.lower()}_roc.png", f"{name} ROC Curve")

    return {"accuracy": acc, "auc": auc, "confusion_matrix": cm.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8, help="Downsample size (e.g., 8 or 16)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rgb", action="store_true", help="Use RGB features instead of grayscale")
    parser.add_argument(
        "--C_grid",
        nargs="+",
        type=float,
        default=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        help="Grid of inverse-regularization strengths to try",
    )
    args = parser.parse_args()

    size = args.size
    seed = args.seed
    grayscale = not args.rgb

    fig_dir = Path("figures") / f"logreg_{size}x{size}"
    res_dir = Path("results") / f"logreg_{size}x{size}"
    ensure_dir(fig_dir)
    ensure_dir(res_dir)

    ds = load_splits(size=size, grayscale=grayscale)

    X_train = np.asarray(ds.X_train, dtype=np.float64)
    y_train = np.asarray(ds.y_train, dtype=np.int64)
    X_val = np.asarray(ds.X_val, dtype=np.float64)
    y_val = np.asarray(ds.y_val, dtype=np.int64)
    X_test = np.asarray(ds.X_test, dtype=np.float64)
    y_test = np.asarray(ds.y_test, dtype=np.int64)

    for name, X in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        if not np.isfinite(X).all():
            bad = np.argwhere(~np.isfinite(X))
            raise ValueError(f"{name} has non-finite values. Example index: {bad[0]}")

    best_model: LogisticRegression | None = None
    best_C: float | None = None
    best_val_auc = -1.0

    for C in args.C_grid:
        model = LogisticRegression(
            C=C,
            solver="lbfgs",
            max_iter=5000,
            random_state=seed,
        )
        model.fit(X_train, y_train)

        val_probs = model.predict_proba(X_val)[:, 1]
        val_auc = float(roc_auc_score(y_val, val_probs))

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = model
            best_C = C

    assert best_model is not None and best_C is not None

    results = {
        "config": {
            "model": "LogReg",
            "size": size,
            "grayscale": grayscale,
            "C_grid": args.C_grid,
            "best_C": float(best_C),
            "seed": seed,
        },
        "val": eval_split(best_model, X_val, y_val, fig_dir, "VAL"),
        "test": eval_split(best_model, X_test, y_test, fig_dir, "TEST"),
    }

    if grayscale:
        w = best_model.coef_.reshape(size, size)
        fig = plt.figure()
        plt.imshow(w)
        plt.title(f"LogReg weights ({size}x{size})")
        plt.colorbar()
        fig.savefig(fig_dir / "weights_heatmap.png", bbox_inches="tight")
        plt.close(fig)

    # Save JSON to BOTH locations
    json_text = json.dumps(results, indent=2)
    (fig_dir / "results.json").write_text(json_text)
    (res_dir / "results.json").write_text(json_text)

    print("Saved figures to:", fig_dir)
    print("Saved results to:", res_dir)
    print(json_text)


if __name__ == "__main__":
    main()

