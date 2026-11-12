"""Try every model in this repo on your data and keep the best one.

    python automl.py --data your_data.csv --target column_to_predict

Run with --help for all options.
"""

from utils.generalized import automl_cli

if __name__ == "__main__":
    automl_cli()
