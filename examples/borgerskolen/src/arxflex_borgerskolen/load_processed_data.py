from pathlib import Path

import pandas as pd

DATA_FILE = Path(__file__).resolve().parent / "borgerskolen_experiment.csv"


def load_processed_data(path=None):
    """Load borgerskolen_experiment.csv (the notebook's `proccessed_data`).
    30-min data, 2021-03-01 00:30 to 2021-03-27 23:30 UTC, 1295 rows.

    path : the CSV to read; by default the copy installed with this package.
    """
    data = pd.read_csv(DATA_FILE if path is None else path, index_col=0)
    data["time"] = pd.to_datetime(data["time"])
    return data
