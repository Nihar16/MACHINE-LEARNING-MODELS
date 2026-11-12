"""Convolutional Neural Network on MNIST (70,000 handwritten digits, 28x28).

Run:  python models/12_cnn/model.py            (about 10 epochs; minutes on CPU, seconds on GPU)
      python models/12_cnn/model.py --epochs 3 (quick check)

Protocol: the official 60,000 training images are split into 55,000 for training
and 5,000 for validation (used only to pick the best epoch). The official 10,000
test images are scored once at the end.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnn import build_model, count_params, fit, get_device, predict_loader, save_plots, seed_everything, write_json  # noqa: E402

HERE = Path(__file__).resolve().parent
MEAN, STD = 0.1307, 0.3081  # MNIST pixel statistics


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--data-dir", default=str(HERE.parents[1] / "data"))
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    seed_everything(42)
    device = get_device()

    # Light augmentation (small shifts, rotations, zooms) teaches the network that a
    # slightly moved or tilted digit is still the same digit.
    train_tf = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,)),
    ])
    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
    train_aug = datasets.MNIST(a.data_dir, train=True, download=True, transform=train_tf)
    train_plain = datasets.MNIST(a.data_dir, train=True, download=False, transform=eval_tf)
    test = datasets.MNIST(a.data_dir, train=False, download=True, transform=eval_tf)

    idx = np.random.default_rng(42).permutation(len(train_aug))
    train_idx, val_idx = idx[5_000:], idx[:5_000]
    loader = lambda ds, shuffle: DataLoader(ds, batch_size=a.batch_size, shuffle=shuffle, num_workers=a.workers)
    train_loader = loader(Subset(train_aug, train_idx), True)
    val_loader = loader(Subset(train_plain, val_idx), False)
    test_loader = loader(test, False)

    model = build_model("simple", n_classes=10, in_channels=1)
    print(f"MNIST: {len(train_idx)} train / {len(val_idx)} validation / {len(test)} test images")
    print(f"SimpleCNN with {count_params(model):,} parameters on {device}, {a.epochs} epochs")
    t0 = time.perf_counter()
    history = fit(model, train_loader, val_loader, a.epochs, a.lr, device)
    train_secs = time.perf_counter() - t0

    probs, y = predict_loader(model, test_loader, device)
    pred = probs.argmax(1)
    acc = float((pred == y).mean())
    wrong = np.flatnonzero(pred != y)
    print(f"\nBest epoch {history['best_epoch']} (validation accuracy {history['best_val_acc']:.4f})")
    print(f"Test accuracy: {acc:.4f}  ({len(wrong)} wrong out of {len(y)})")
    print(f"≥99% on the held-out test set: {'YES' if acc >= 0.99 else 'NO'}")

    out = HERE / "results"
    metrics = {
        "model": "CNN (SimpleCNN)", "dataset": "MNIST", "n_train": len(train_idx), "n_val": len(val_idx),
        "n_test": len(y), "parameters": count_params(model), "epochs": a.epochs, "best_epoch": history["best_epoch"],
        "val_accuracy": history["best_val_acc"], "test_accuracy": acc, "test_errors": int(len(wrong)),
        "device": str(device), "train_seconds": round(train_secs),
    }
    write_json(out / "metrics.json", metrics)
    raw = datasets.MNIST(a.data_dir, train=False, transform=transforms.ToTensor())
    save_plots(history, confusion_matrix(y, pred), list(range(10)), f"CNN on MNIST: test acc {acc:.4f}", out,
               wrong_images=[raw[i][0].numpy() for i in wrong[:32]],
               wrong_info=[f"{y[i]} -> {pred[i]}" for i in wrong[:32]])
    torch.save({"state_dict": model.state_dict(), "arch": "simple", "classes": [str(i) for i in range(10)],
                "in_channels": 1, "image_size": 28, "mean": [MEAN], "std": [STD]}, out / "cnn_mnist.pt")
    print(f"Saved metrics, plots and weights to {out}")


if __name__ == "__main__":
    main()
