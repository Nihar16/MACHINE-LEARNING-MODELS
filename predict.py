"""Predict with a saved model.

    python predict.py --model ml_output/best/model.joblib --data new_rows.csv --out predictions.csv

Run with --help for all options.
"""

from utils.generalized import predict_cli

if __name__ == "__main__":
    predict_cli()
