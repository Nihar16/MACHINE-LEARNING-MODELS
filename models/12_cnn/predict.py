"""Classify images with a CNN saved by train.py or model.py.

    python models/12_cnn/predict.py --model ml_output/cnn/model.pt --images photo.jpg
    python models/12_cnn/predict.py --model ml_output/cnn/model.pt --images new_photos/ --out predictions.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnn import build_model, get_device  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def load(model_path):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], len(ckpt["classes"]), ckpt["in_channels"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mode = transforms.Grayscale(1) if ckpt["in_channels"] == 1 else transforms.Lambda(lambda im: im.convert("RGB"))
    tf = transforms.Compose([mode, transforms.Resize((ckpt["image_size"], ckpt["image_size"])),
                             transforms.ToTensor(), transforms.Normalize(ckpt["mean"], ckpt["std"])])
    return model, tf, ckpt["classes"]


def main(argv=None):
    p = argparse.ArgumentParser(description="Classify images with a saved CNN.")
    p.add_argument("--model", required=True)
    p.add_argument("--images", required=True, help="an image file or a folder of images")
    p.add_argument("--out", help="optional CSV path")
    a = p.parse_args(argv)
    model, tf, classes = load(a.model)
    device = get_device()
    model.to(device)
    src = Path(a.images)
    files = sorted(f for f in (src.rglob("*") if src.is_dir() else [src]) if f.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"No images found in {src}")
    rows = []
    with torch.no_grad():
        for i in range(0, len(files), 64):
            batch = files[i:i + 64]
            x = torch.stack([tf(Image.open(f)) for f in batch]).to(device)
            probs = torch.softmax(model(x), 1).cpu()
            for f, pr in zip(batch, probs):
                k = int(pr.argmax())
                rows.append({"file": str(f), "prediction": classes[k], "confidence": round(float(pr[k]), 4)})
    for r in rows[:20]:
        print(f"{r['prediction']:<15}{r['confidence']:.3f}  {r['file']}")
    if len(rows) > 20:
        print(f"... {len(rows) - 20} more")
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {len(rows)} predictions to {a.out}")
    return rows


if __name__ == "__main__":
    main()
