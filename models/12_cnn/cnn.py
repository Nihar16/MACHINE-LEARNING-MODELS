"""Shared CNN code: the network, the training loop and evaluation.

Used by model.py (MNIST benchmark), train.py (your own images) and predict.py.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def conv_block(c_in: int, c_out: int, dropout: float) -> nn.Sequential:
    """Two 3x3 convolutions with batch norm, then 2x2 max pooling."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2), nn.Dropout(dropout),
    )


class SimpleCNN(nn.Module):
    """A compact VGG-style CNN.

    conv(32) x2 -> pool -> conv(64) x2 -> pool [-> conv(128) x2 -> pool] -> global pool -> dense -> classes

    Global average pooling makes it work for any input size (28x28 digits, 64x64 photos, ...).
    """

    def __init__(self, n_classes: int, in_channels: int = 1, widths=(32, 64, 128), dropout: float = 0.25):
        super().__init__()
        blocks, c = [], in_channels
        for w in widths:
            blocks.append(conv_block(c, w, dropout))
            c = w
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(c, 128), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_model(arch: str, n_classes: int, in_channels: int, pretrained: bool = False) -> nn.Module:
    if arch == "simple":
        return SimpleCNN(n_classes, in_channels)
    if arch == "resnet18":  # transfer learning: usually the best choice for small photo datasets
        from torchvision.models import ResNet18_Weights, resnet18

        m = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        if in_channels != 3:
            m.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        return m
    raise ValueError(f"unknown arch {arch!r}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), 1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def fit(model, train_loader: DataLoader, val_loader: DataLoader, epochs: int, lr: float, device,
        label_smoothing: float = 0.05, verbose: bool = True) -> dict:
    """Train with AdamW + one-cycle LR; keep the weights with the best validation accuracy."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    best_acc, best_state, best_epoch = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total, correct, loss_sum = time.perf_counter(), 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        val_probs, val_y = predict_loader(model, val_loader, device)
        val_acc = float((val_probs.argmax(1) == val_y).mean())
        history["train_loss"].append(loss_sum / total)
        history["train_acc"].append(correct / total)
        history["val_acc"].append(val_acc)
        if val_acc > best_acc:
            best_acc, best_epoch, best_state = val_acc, epoch, copy.deepcopy(model.state_dict())
        if verbose:
            print(f"  epoch {epoch:>2}/{epochs}  loss {loss_sum / total:.4f}  train acc {correct / total:.4f}  "
                  f"val acc {val_acc:.4f}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    history["best_epoch"], history["best_val_acc"] = best_epoch, best_acc
    return history


def save_plots(history: dict, cm: np.ndarray, class_names, title: str, out_dir: Path,
               wrong_images=None, wrong_info=None) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
    except ImportError:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    ep = range(1, len(history["val_acc"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(ep, history["train_loss"], "o-")
    ax[0].set(title="Training loss", xlabel="Epoch", ylabel="Loss")
    ax[1].plot(ep, history["train_acc"], "o-", label="train")
    ax[1].plot(ep, history["val_acc"], "o-", label="validation")
    ax[1].axhline(0.99, ls="--", color="gray", lw=0.8, label="99%")
    ax[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(max(0.0, min(history["val_acc"]) - 0.02), 1.0))
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=110)
    plt.close(fig)

    size = 4 + 0.4 * len(cm)
    fig, ax = plt.subplots(figsize=(size, size))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=110)
    plt.close(fig)

    if wrong_images is not None and len(wrong_images):
        n = min(len(wrong_images), 32)
        cols = 8
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.6))
        for i, a in enumerate(np.atleast_1d(axes).ravel()):
            a.axis("off")
            if i < n:
                img = wrong_images[i]
                a.imshow(img.squeeze(), cmap="gray" if img.shape[0] == 1 else None)
                a.set_title(wrong_info[i], fontsize=7)
        fig.suptitle("Misclassified test images (true -> predicted)")
        fig.tight_layout()
        fig.savefig(out_dir / "misclassified.png", dpi=110)
        plt.close(fig)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")
