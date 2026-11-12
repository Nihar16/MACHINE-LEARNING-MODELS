"""Train Support Vector Machine on your own data (classification or regression).

    python models/05_svm/train.py --data your_data.csv --target column_to_predict

Run with --help for all options. The original benchmark is in model.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from utils.generalized import train_cli  # noqa: E402

if __name__ == "__main__":
    train_cli("svm")
