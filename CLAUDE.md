# arxflex

ARX models with nonlinear input blocks (B-splines over time of day, sigmoid and
PCHIP valve curves), specified as formulas. The nonlinear parameters are
optimised (SLSQP); for each trial the linear ARX parameters are found by least
squares. Fitted models are simulated multi-step.

A public repository on GitHub (GPL-3.0-or-later; the example data CC BY 4.0),
so nothing private (keys, paths to personal data, unpublished data) may be
committed.

## Layout

```
src/arxflex/                  the package: formula core only
examples/borgerskolen/        a separate package, arxflex-borgerskolen
  src/arxflex_borgerskolen/   installed: processed CSV + load_processed_data,
                              temperature_column, available_rooms
  legacy/                     repo only: original notebook + hand-written models
  check_fitmodel.py           fitmodel() vs the legacy models
```

- `src/arxflex` must stay site independent: no room names, no Borgerskolen or
  ENEA column names. Site-specific helpers belong in an example package or in
  the user's own project.
- The example is optional: `pip install arxflex` gives the core only;
  `arxflex-borgerskolen` (or `arxflex[examples]`) adds the example data.
  Legacy code is never installed. Check this by building the wheels
  (`pip wheel --no-deps --no-build-isolation -w <tmpdir> . ./examples/borgerskolen`)
  and listing their contents.
- Core dependencies are numpy and scipy; pandas (data-frame input) and
  matplotlib (the `plot_*` functions) are the optional extras `[pandas]` and `[plot]`.

## Code conventions

- Each file is named after its main function; a helper used by only one
  function lives in that function's file.
- Functions take their settings as arguments; the calling script owns the
  values. The only value kept in the package is `BAD_FIT` (objective value for
  invalid parameters), which `fitmodel()` and `compile_model()` accept as
  `bad_fit=`.
- Model specification: a dict of formula terms, e.g.
  `{"AR": 2, "ambient": "Ta - Ti", "solar": "s(timeofday, k=5) * Ps",
  "heating": "v(sigmoid, Tset - Ti) * (Tfor - Ti)"}`. `AR` may be 0.

## Checking changes

With both packages installed (`pip install -e . -e examples/borgerskolen`),
or with `PYTHONPATH=src:examples/borgerskolen/src`:

```sh
python examples/borgerskolen/check_fitmodel.py
```

For each of the sigmoid, free-knots, valve and no-solar models it should give
an objective difference at the old optimum of about 1e-14, a simulation
difference of about 1e-13, and a refitted S within about 1e-5 of the legacy fit
(S = 8.7469, 8.7469, 6.7317 and 9.5928). Small differences after
refitting come from the SLSQP path, as the parameter vector is ordered by term.

The user runs Python as a long-lived process in Emacs: after changing package
code, the process has to be restarted (or the modules reloaded).

## The Borgerskolen data

`src/arxflex_borgerskolen/borgerskolen_experiment.csv` (1295 rows, 30 min,
2021-03-01 00:30 to 2021-03-27 23:30 UTC) is built by `build_data.py`, which
lives with the raw data in a separate, non-public data repository, not here.
The original processed file was never available; this is a rebuild:

- `inchannels15min.csv`: raw 15-min building data, taken from the Borgerskolen
  data dump `2021-04-13_data-csv.zip`.
- `dmi_06188.csv`: DMI station 06188 (Sjaelsmark), `temp_mean_past1h` and
  `radia_glob`, downloaded from DMI's open metObs v2 API (no key). The notebook
  used the v1 API, which no longer exists. `load_DMI_data(cache_file=...)`
  reads this file and downloads it only if it is missing.
- The processing deliberately reproduces the notebook, including its quirks:
  Danish local time (Europe/Copenhagen), a 20-min radiation gap filled with
  two copies instead of one, and gap filling of temperature with values from
  the wrong end of the series.
- Assumption: v1 excluded the end of the time interval. v2 includes both ends,
  and with both kept the processing fails (Ta becomes one value too long).

The fit values printed in `legacy/ARXmodelling_experiment.ipynb` are not an
exact reference: the notebook was run across several kernel sessions, out of
order. The valve fit for room C1.09 is close (S 6.732 here vs 6.738 in the
notebook); the sigmoid fit for C0.07 is not (8.747 vs 6.719). Don't chase an
exact match; `check_fitmodel.py` compares legacy and new code on the same data.

## Users

The ENEA grid project uses arxflex for its own data; its scripts and data
stay in that project's own repository.
