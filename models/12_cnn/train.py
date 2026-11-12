"""Train a CNN on your own images.

Put one folder per class:
    my_images/
        cats/  img001.jpg  img002.png ...
        dogs/  ...
(or my_images/train/<class>/... and my_images/test/<class>/... to use your own split)

    python models/12_cnn/train.py --data my_images
    python models/12_cnn/train.py --data my_images --arch resnet18 --pretrained   # transfer learning, best for photos
    python models/12_cnn/predict.py --model ml_output/cnn/model.pt --images new_photos/

Protocol: 70% train / 15% validation (picks the best epoch) / 15% test (scored once),
stratified by class. The test score is the honest estimate on unseen images.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnn import build_model, count_params, fit, get_device, predict_loader, save_plots, seed_everything, write_json  # noqa: E402

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def make_transforms(size: int, channels: int, mean, std, flip: bool):
    base = [transforms.Grayscale(1)] if channels == 1 else [transforms.Lambda(lambda im: im.convert("RGB"))]
    aug = [
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        *([transforms.RandomHorizontalFlip()] if flip else []),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2) if channels == 3 else transforms.Lambda(lambda im: im),
    ]
    tail = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    return (transforms.Compose(base + aug + tail),
            transforms.Compose(base + [transforms.Resize((size, size))] + tail))


def split_indices(targets, seed=42, val=0.15, test=0.15):
    idx = np.arange(len(targets))
    counts = np.bincount(targets)
    if counts.min() < 4:
        raise SystemExit(f"Every class needs at least 4 images; smallest has {counts.min()}.")
    tr, rest = train_test_split(idx, test_size=val + test, stratify=targets, random_state=seed)
    va, te = train_test_split(rest, test_size=test / (val + test), stratify=np.asarray(targets)[rest], random_state=seed)
    return tr, va, te


def run(argv=None):
    p = argparse.ArgumentParser(description="Train a CNN on your own image folders.")
    p.add_argument("--data", required=True, help="folder with one sub-folder per class (or train/ and test/)")
    p.add_argument("--arch", choices=["simple", "resnet18"], default="simple")
    p.add_argument("--pretrained", action="store_true", help="start from ImageNet weights (resnet18 only)")
    p.add_argument("--size", type=int, help="image side in pixels (default 64; 224 for pretrained resnet18)")
    p.add_argument("--channels", type=int, choices=[1, 3], default=3, help="1 = grayscale, 3 = colour")
    p.add_argument("--no-flip", action="store_true", help="disable horizontal flips (use for digits/text)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, help="peak learning rate (default 3e-3; 1e-3 for pretrained)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--target-score", type=float, default=0.99)
    p.add_argument("--out", default="ml_output/cnn")
    a = p.parse_args(argv)
    if a.pretrained and a.arch != "resnet18":
        raise SystemExit("--pretrained needs --arch resnet18")
    seed_everything(42)
    size = a.size or (224 if a.pretrained else 64)
    channels = 3 if a.pretrained else a.channels
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if a.pretrained else ([0.5] * channels, [0.5] * channels)
    lr = a.lr or (1e-3 if a.pretrained else 3e-3)
    train_tf, eval_tf = make_transforms(size, channels, mean, std, flip=not a.no_flip)

    root = Path(a.data)
    if (root / "train").is_dir() and (root / "test").is_dir():
        full_aug = datasets.ImageFolder(root / "train", transform=train_tf)
        full_plain = datasets.ImageFolder(root / "train", transform=eval_tf)
        test_ds = datasets.ImageFolder(root / "test", transform=eval_tf)
        if test_ds.classes != full_aug.classes:
            raise SystemExit("train/ and test/ must contain the same class folders.")
        tr, va = train_test_split(np.arange(len(full_aug)), test_size=0.15, stratify=full_aug.targets, random_state=42)
        test_set = test_ds
    else:
        full_aug = datasets.ImageFolder(root, transform=train_tf)
        full_plain = datasets.ImageFolder(root, transform=eval_tf)
        tr, va, te = split_indices(np.array(full_aug.targets))
        test_set = Subset(full_plain, te)
    classes = full_aug.classes
    if len(classes) < 2:
        raise SystemExit(f"Found {len(classes)} class folder(s) in {root}; need at least 2.")

    loader = lambda ds, shuffle: DataLoader(ds, batch_size=a.batch_size, shuffle=shuffle, num_workers=a.workers)
    train_loader = loader(Subset(full_aug, tr), True)
    val_loader = loader(Subset(full_plain, va), False)
    test_loader = loader(test_set, False)

    device = get_device()
    model = build_model(a.arch, len(classes), channels, pretrained=a.pretrained)
    counts = np.bincount(np.array(full_aug.targets)[tr], minlength=len(classes))
    print(f"{len(classes)} classes: {dict(zip(classes, counts.tolist()))} (training images per class)")
    print(f"Split: {len(tr)} train / {len(va)} validation / {len(test_loader.dataset)} test")
    print(f"{a.arch}{' (pretrained)' if a.pretrained else ''}, {count_params(model):,} parameters, "
          f"{size}x{size}x{channels}, on {device}, {a.epochs} epochs")
    t0 = time.perf_counter()
    history = fit(model, train_loader, val_loader, a.epochs, lr, device)
    secs = time.perf_counter() - t0

    probs, y = predict_loader(model, test_loader, device)
    pred = probs.argmax(1)
    acc = float((pred == y).mean())
    baseline = float(np.bincount(y).max() / len(y))
    print(f"\nBest epoch {history['best_epoch']} (validation accuracy {history['best_val_acc']:.4f})")
    print(f"Test accuracy: {acc:.4f} ({int((pred != y).sum())} wrong out of {len(y)}); "
          f"baseline (always the most common class): {baseline:.4f}")
    print(classification_report(y, pred, labels=range(len(classes)), target_names=classes, digits=3, zero_division=0))
    if len(y) < 200:
        print(f"Note: only {len(y)} test images, so one mistake moves accuracy by {1 / len(y):.1%}.")
    if acc >= a.target_score:
        print(f"Target {a.target_score:.0%}: REACHED on the held-out test images.")
    else:
        tips = "Try --arch resnet18 --pretrained, more epochs, or more images per class." if not a.pretrained \
            else "Try more epochs or more (and more varied) images per class."
        print(f"Target {a.target_score:.0%}: NOT reached. {tips}")

    out = Path(a.out)
    metrics = {"model": f"CNN ({a.arch}{', pretrained' if a.pretrained else ''})", "data": str(root),
               "classes": classes, "n_train": len(tr), "n_val": len(va), "n_test": len(y), "image_size": size,
               "channels": channels, "epochs": a.epochs, "best_epoch": history["best_epoch"],
               "val_accuracy": history["best_val_acc"], "test_accuracy": acc, "baseline_accuracy": baseline,
               "target_reached": acc >= a.target_score, "train_seconds": round(secs)}
    write_json(out / "metrics.json", metrics)
    save_plots(history, confusion_matrix(y, pred, labels=range(len(classes))), classes,
               f"test acc {acc:.4f}", out)
    torch.save({"state_dict": model.state_dict(), "arch": a.arch, "classes": classes, "in_channels": channels,
                "image_size": size, "mean": mean, "std": std}, out / "model.pt")
    print(f"Saved {out / 'model.pt'}, metrics.json and plots")
    return metrics


if __name__ == "__main__":
    run()
