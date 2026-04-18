from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class WildfireCNNv2(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        return self.classifier(z).squeeze(1)


def build_model(arch: str, in_ch: int, size: int) -> nn.Module:
    if arch == "legacy":
        return SmallCNN(in_ch=in_ch, size=size)
    if arch == "wildfire_cnn_v2":
        return WildfireCNNv2(in_ch=in_ch)
    raise ValueError(f"Unsupported architecture: {arch}")


def apply_train_augmentations(
    xb: torch.Tensor,
    *,
    hflip_prob: float,
    brightness_jitter: float,
    contrast_jitter: float,
    noise_std: float,
) -> torch.Tensor:
    x = xb.clone()

    if hflip_prob > 0:
        flip_mask = torch.rand(x.size(0), device=x.device) < hflip_prob
        if bool(flip_mask.any()):
            x[flip_mask] = torch.flip(x[flip_mask], dims=(3,))

    if brightness_jitter > 0:
        brightness = 1.0 + (torch.rand(x.size(0), 1, 1, 1, device=x.device) * 2 - 1) * brightness_jitter
        x = x * brightness

    if contrast_jitter > 0:
        means = x.mean(dim=(2, 3), keepdim=True)
        contrast = 1.0 + (torch.rand(x.size(0), 1, 1, 1, device=x.device) * 2 - 1) * contrast_jitter
        x = (x - means) * contrast + means

    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std

    return x.clamp(0.0, 1.0)


def save_checkpoint(
    checkpoint_path: Path,
    *,
    model: nn.Module,
    config: dict,
    best_val_auc: float,
    best_epoch: int,
) -> None:
    payload = {
        "format_version": 2,
        "architecture": config["architecture"],
        "config": config,
        "best_val_auc": float(best_val_auc),
        "best_epoch": int(best_epoch),
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, checkpoint_path)


def load_checkpoint_for_training(checkpoint_path: Path, model: nn.Module) -> dict | None:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        model.load_state_dict(payload["model_state_dict"])
        return payload

    model.load_state_dict(payload)
    return None


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
    parser.add_argument(
        "--run-name",
        type=str,
        default="",
        help="Optional output directory suffix for preserving multiple experiments at the same resolution",
    )
    parser.add_argument(
        "--splits-root",
        type=Path,
        default=Path("data/splits"),
        help="Path to the canonical train/val/test split directory",
    )
    parser.add_argument(
        "--extra-train-root",
        action="append",
        default=[],
        type=Path,
        help="Additional canonical split roots whose train split should be merged into training only",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint to load before training (legacy or metadata format)",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs)")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--arch",
        choices=["legacy", "wildfire_cnn_v2"],
        default="wildfire_cnn_v2",
        help="CNN architecture to train",
    )
    parser.add_argument("--rgb", action="store_true", help="Use RGB images (3-channel) instead of grayscale")
    parser.add_argument("--device", type=str, default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument("--skip-figures", action="store_true", help="Skip saving plots and only write JSON results")
    parser.add_argument("--augment", action="store_true", help="Enable lightweight training-time augmentation")
    parser.add_argument("--hflip-prob", type=float, default=0.5, help="Horizontal flip probability when augmenting")
    parser.add_argument(
        "--brightness-jitter",
        type=float,
        default=0.15,
        help="Brightness jitter strength when augmenting",
    )
    parser.add_argument(
        "--contrast-jitter",
        type=float,
        default=0.15,
        help="Contrast jitter strength when augmenting",
    )
    parser.add_argument("--noise-std", type=float, default=0.02, help="Gaussian noise std when augmenting")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    size = args.size
    grayscale = not args.rgb
    in_ch = 1 if grayscale else 3
    base_run_name = f"cnn_{size}x{size}"
    run_name = f"{base_run_name}_{args.run_name}" if args.run_name else base_run_name

    fig_dir = Path("figures") / run_name
    res_dir = Path("results") / run_name
    ensure_dir(fig_dir)
    ensure_dir(res_dir)

    device = get_device(args.device)

    ds = load_splits(
        splits_root=args.splits_root,
        extra_train_roots=args.extra_train_root,
        size=size,
        grayscale=grayscale,
    )

    # ds.X_* is (N, size*size) if grayscale; if rgb, typically (N, size*size*3)
    def to_image_tensor(X: np.ndarray) -> torch.Tensor:
        X = np.asarray(X, dtype=np.float32)
        if grayscale:
            X = X.reshape(-1, 1, size, size)
        else:
            X = X.reshape(-1, size, size, 3)
            X = np.transpose(X, (0, 3, 1, 2))
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

    model = build_model(args.arch, in_ch=in_ch, size=size).to(device)
    if args.init_checkpoint is not None:
        load_checkpoint_for_training(args.init_checkpoint, model)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            if args.augment:
                xb = apply_train_augmentations(
                    xb,
                    hflip_prob=args.hflip_prob,
                    brightness_jitter=args.brightness_jitter,
                    contrast_jitter=args.contrast_jitter,
                    noise_std=args.noise_std,
                )

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
            best_epoch = epoch
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

    results = {
        "config": {
            "model": "CNN",
            "architecture": args.arch,
            "run_name": run_name,
            "size": size,
            "splits_root": str(args.splits_root),
            "extra_train_roots": [str(path) for path in args.extra_train_root],
            "epochs_max": args.epochs,
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "patience": args.patience,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "device": str(device),
            "grayscale": grayscale,
            "rgb_tensor_layout": "chw",
            "augment": bool(args.augment),
            "hflip_prob": args.hflip_prob,
            "brightness_jitter": args.brightness_jitter,
            "contrast_jitter": args.contrast_jitter,
            "noise_std": args.noise_std,
            "variant": (
                f"{'rgb' if args.rgb else 'gray'}-"
                f"{args.arch}-"
                f"{'aug' if args.augment else 'plain'}"
            ),
        },
        "best_val_auc": float(best_val_auc),
        "best_epoch": int(best_epoch),
        "test": {
            "accuracy": float(test_acc),
            "auc": float(test_auc),
            "confusion_matrix": test_cm.tolist(),
        },
        "history": history,
    }

    save_checkpoint(
        fig_dir / "best_model.pt",
        model=model,
        config=results["config"],
        best_val_auc=best_val_auc,
        best_epoch=best_epoch,
    )

    json_text = json.dumps(results, indent=2)

    # Save JSON to BOTH locations
    (fig_dir / "results.json").write_text(json_text)
    (res_dir / "results.json").write_text(json_text)

    if not args.skip_figures:
        save_history_plots(history, fig_dir)
        save_confusion_matrix(test_cm, fig_dir / "test_confusion_matrix.png", "TEST Confusion Matrix")
        save_roc_curve(y_true, y_prob, fig_dir / "test_roc.png", "TEST ROC Curve")

    print("\nSaved figures to:", fig_dir)
    print("Saved results to:", res_dir)
    print(json_text)


if __name__ == "__main__":
    main()
