"""Train K-Means clustering on your own data (no target needed).

    python models/09_kmeans/train.py --data your_data.csv [--k 4] [--target optional_label_for_scoring]

Run with --help for all options. The original benchmark is in model.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from utils.generalized import kmeans_cli  # noqa: E402

if __name__ == "__main__":
    kmeans_cli()
