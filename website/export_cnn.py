"""Export the trained MNIST CNN to a compact binary the website runs in the browser.

Batch-norm layers are folded into the preceding convolutions, so the browser only
needs conv / relu / max-pool / average-pool / dense. Run after models/12_cnn/model.py:

    python website/export_cnn.py
"""

import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "models" / "12_cnn" / "results" / "cnn_mnist.pt"
OUT = ROOT / "docs" / "assets" / "cnn"


def main():
    ckpt = torch.load(SRC, map_location="cpu", weights_only=False)
    sd = {k: v.numpy().astype(np.float32) for k, v in ckpt["state_dict"].items()}
    layers, blobs, offset = [], [], 0

    def add(arr):
        nonlocal offset
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        blobs.append(arr.tobytes())
        start, offset = offset, offset + arr.size
        return start

    for b in range(3):
        for conv_i, bn_i in ((0, 1), (3, 4)):
            w = sd[f"features.{b}.{conv_i}.weight"]
            g, beta = sd[f"features.{b}.{bn_i}.weight"], sd[f"features.{b}.{bn_i}.bias"]
            mean, var = sd[f"features.{b}.{bn_i}.running_mean"], sd[f"features.{b}.{bn_i}.running_var"]
            scale = g / np.sqrt(var + 1e-5)
            layers.append({"type": "conv", "out": int(w.shape[0]), "in": int(w.shape[1]),
                           "w": add(w * scale[:, None, None, None]), "b": add(beta - mean * scale)})
        layers.append({"type": "pool"})
    for name in ("head.2", "head.5"):
        w = sd[f"{name}.weight"]
        layers.append({"type": "dense", "out": int(w.shape[0]), "in": int(w.shape[1]),
                       "w": add(w), "b": add(sd[f"{name}.bias"])})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "weights.bin").write_bytes(b"".join(blobs))
    (OUT / "model.json").write_text(json.dumps({
        "mean": ckpt["mean"][0], "std": ckpt["std"][0], "classes": ckpt["classes"], "floats": offset,
        "layers": layers}, indent=1) + "\n")
    print(f"wrote {offset:,} floats ({offset * 4 / 1e6:.2f} MB) to {OUT}")


if __name__ == "__main__":
    main()
