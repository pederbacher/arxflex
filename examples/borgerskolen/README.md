# arxflex-borgerskolen

Example data for [arxflex](../../README.md): 30-min room temperatures and
thermostat set-points for 16 rooms in Borgerskolen, 1-27 March 2021, with
outdoor temperature and global radiation from DMI station 06188 (Sjaelsmark).

```python
from arxflex_borgerskolen import load_processed_data, available_rooms, temperature_column

data = load_processed_data()
available_rooms(data)             # ['0.06', '0.07', ...]
```

## Vignette

`vignette.py` is a worked example of arxflex on this data. It fits a model
to one room, analyses the simulation, residuals and estimated functions,
compares model structures, validates on unseen data, and fits all rooms.
It is a Python script with `# %%` cells. Run it cell by cell, or render it:

```sh
quarto render vignette.py                    # HTML page with the figures
jupytext --to ipynb --execute vignette.py    # or an executed notebook
```

Quarto needs `nbformat`, `nbclient` and `ipykernel` in its Python.

## The data

`borgerskolen_experiment.csv` was made from the raw 15-min building data of
Borgerskolen and the DMI observations, resampled to 30 min. The raw data and
the script that builds the file are not part of this repository. The DMI data
were downloaded in 2026 from DMI's open data API (v2); the original analysis
used the older v1 API, which no longer exists.

The data are licensed under CC BY 4.0; the weather data are from the Danish
Meteorological Institute (DMI). See [`DATA_LICENSE.md`](DATA_LICENSE.md).

## Folders (repository only, not installed)

- `legacy/`: the original notebook `ARXmodelling_experiment.ipynb` and the
  hand-written models ported from it. `check_fitmodel.py` shows that
  `arxflex.fitmodel()` reproduces them.

Run the scripts from a clone with both packages installed
(`pip install -e . -e examples/borgerskolen` in the repository root).
