from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, accuracy_score

from src.data_utils import load_splits


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def get_device(preferred: str) -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SmallCNN(nn.Module):
    def __init__(self, in_ch: int, size: int):
        super().__init__()
        # simple, stable baseline CNN
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        z = z.view(z.size(0), -1)
        return self.head(z).squeeze(1)  # logits


@torch.no_grad()
def eval_auc_and_cm(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_probs = []
    all_y = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.detach().cpu().numpy())
        all_y.append(yb.detach().cpu().numpy())

    y = np.concatenate(all_y).astype(int)
    p = np.concatenate(all_probs)

    auc = float(roc_auc_score(y, p))
    preds = (p >= 0.5).astype(int)
    acc = float(accuracy_score(y, preds))
    cm = confusion_matrix(y, preds)
    return auc, acc, cm, y, p


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


def save_history_plots(history: list[dict], fig_dir: Path) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_auc = [h["val_auc"] for h in history]

    fig1 = plt.figure()
    plt.plot(epochs, train_loss)
    plt.title("Train loss vs epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    fig1.savefig(fig_dir / "train_loss.png", bbox_inches="tight")
    plt.close(fig1)

    fig2 = plt.figure()
    plt.plot(epochs, val_auc)
    plt.title("Validation AUC vs epoch")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    fig2.savefig(fig_dir / "val_auc.png", bbox_inches="tight")
    plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=16, help="Image downsample size for CNN (e.g., 16, 64)")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs)")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rgb", action="store_true", help="Use RGB images (3-channel) instead of grayscale")
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | mps | cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    size = args.size
    grayscale = not args.rgb
    in_ch = 1 if grayscale else 3

    fig_dir = Path("figures") / f"cnn_{size}x{size}"
    res_dir = Path("results") / f"cnn_{size}x{size}"
    ensure_dir(fig_dir)
    ensure_dir(res_dir)

    device = get_device(args.device)

    ds = load_splits(size=size, grayscale=grayscale)

    # ds.X_* is (N, size*size) if grayscale; if rgb, typically (N, size*size*3)
    def to_image_tensor(X: np.ndarray) -> torch.Tensor:
        X = np.asarray(X, dtype=np.float32)
        if grayscale:
            X = X.reshape(-1, 1, size, size)
        else:
            X = X.reshape(-1, 3, size, size)
        return torch.from_numpy(X)

    X_train = to_image_tensor(ds.X_train)
    y_train = torch.from_numpy(np.asarray(ds.y_train, dtype=np.float32))

    X_val = to_image_tensor(ds.X_val)
    y_val = torch.from_numpy(np.asarray(ds.y_val, dtype=np.float32))

    X_test = to_image_tensor(ds.X_test)
    y_test = torch.from_numpy(np.asarray(ds.y_test, dtype=np.float32))

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size, shuffle=False)

    model = SmallCNN(in_ch=in_ch, size=size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_state = None
    bad_epochs = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(losses))

        val_auc, _, _, _, _ = eval_auc_and_cm(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_auc": float(val_auc)})

        print(f"Epoch {epoch:02d} | loss={train_loss:.4f} | val AUC={val_auc:.4f}")

        if val_auc > best_val_auc + 1e-6:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch} (best val AUC={best_val_auc:.4f})")
                break

    assert best_state is not None
    model.load_state_dict(best_state)

    # Final eval
    test_auc, test_acc, test_cm, y_true, y_prob = eval_auc_and_cm(model, test_loader, device)

    # Save figures
    save_history_plots(history, fig_dir)
    save_confusion_matrix(test_cm, fig_dir / "test_confusion_matrix.png", "TEST Confusion Matrix")
    save_roc_curve(y_true, y_prob, fig_dir / "test_roc.png", "TEST ROC Curve")

    results = {
        "config": {
            "model": "CNN",
            "size": size,
            "epochs_max": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "patience": args.patience,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "device": str(device),
            "grayscale": grayscale,
        },
        "best_val_auc": float(best_val_auc),
        "test": {
            "accuracy": float(test_acc),
            "auc": float(test_auc),
            "confusion_matrix": test_cm.tolist(),
        },
        "history": history,
    }

    json_text = json.dumps(results, indent=2)

    # Save JSON to BOTH locations
    (fig_dir / "results.json").write_text(json_text)
    (res_dir / "results.json").write_text(json_text)

    print("\nSaved figures to:", fig_dir)
    print("Saved results to:", res_dir)
    print(json_text)


if __name__ == "__main__":
    main()

