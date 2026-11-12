"""Tests for the CNN scripts (skipped if PyTorch is not installed). Run: python -m pytest tests -q"""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from PIL import Image  # noqa: E402

CNN_DIR = Path(__file__).resolve().parents[1] / "models" / "12_cnn"
sys.path.insert(0, str(CNN_DIR))

import predict  # noqa: E402
import train  # noqa: E402
from cnn import SimpleCNN, build_model  # noqa: E402


def make_shapes(root: Path, n_per_class: int = 40, size: int = 32, seed: int = 0) -> Path:
    """Three easy classes: horizontal bar, vertical bar, filled square, with jitter."""
    rng = np.random.default_rng(seed)
    for cls in ("hbar", "vbar", "square"):
        (root / cls).mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            img = rng.integers(0, 40, (size, size), dtype=np.uint8)
            o = rng.integers(-3, 4)
            c = size // 2 + o
            if cls == "hbar":
                img[c - 2:c + 2, 4:-4] = 255
            elif cls == "vbar":
                img[4:-4, c - 2:c + 2] = 255
            else:
                img[c - 7:c + 7, c - 7:c + 7] = 255
            Image.fromarray(img).convert("RGB").save(root / cls / f"{i}.png")
    return root


@pytest.mark.parametrize("shape", [(2, 1, 28, 28), (2, 3, 64, 64), (2, 3, 100, 80)])
def test_simple_cnn_any_size(shape):
    out = SimpleCNN(n_classes=5, in_channels=shape[1])(torch.zeros(shape))
    assert out.shape == (2, 5)


def test_resnet18_builds_without_download():
    out = build_model("resnet18", n_classes=4, in_channels=1)(torch.zeros(2, 1, 64, 64))
    assert out.shape == (2, 4)


def test_train_and_predict_on_image_folder(tmp_path):
    data = make_shapes(tmp_path / "shapes")
    out = tmp_path / "out"
    m = train.run(["--data", str(data), "--size", "32", "--channels", "1", "--no-flip",
                   "--epochs", "10", "--batch-size", "8", "--workers", "0", "--out", str(out)])
    assert m["classes"] == ["hbar", "square", "vbar"]
    assert m["test_accuracy"] >= 0.9
    assert (out / "model.pt").exists() and (out / "confusion_matrix.png").exists()

    rows = predict.main(["--model", str(out / "model.pt"), "--images", str(data / "square"),
                         "--out", str(tmp_path / "p.csv")])
    assert len(rows) == 40
    assert np.mean([r["prediction"] == "square" for r in rows]) >= 0.9


def test_train_rejects_single_class(tmp_path):
    data = make_shapes(tmp_path / "s")
    for d in ("vbar", "square"):
        for f in (data / d).iterdir():
            f.unlink()
        (data / d).rmdir()
    with pytest.raises(SystemExit):
        train.run(["--data", str(data), "--epochs", "1", "--workers", "0", "--out", str(tmp_path / "o")])
