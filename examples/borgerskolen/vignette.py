# %% [markdown]
# ---
# title: "arxflex: modelling room temperatures in Borgerskolen"
# format:
#   html:
#     toc: true
#     embed-resources: true
# jupyter: python3
# ---
#
# This vignette shows how to use **arxflex** to fit ARX models with nonlinear
# input blocks to room temperatures, simulate them, and compare models. It
# uses the example data in **arxflex-borgerskolen**: 30-min measurements from
# a school in March 2021.
#
# You can run it cell by cell (`# %%` marks a cell) in Emacs, VS Code or
# Jupyter, or render it to HTML from `examples/borgerskolen`:
#
# ```sh
# quarto render vignette.py                    # HTML page with the figures
# jupytext --to ipynb --execute vignette.py    # or an executed notebook
# ```
#
# It needs both packages and the optional extras:
#
# ```sh
# pip install ".[pandas,plot]" ./examples/borgerskolen
# ```

# %% [markdown]
# ## Setup
#
# `arxflex` holds the modelling functions. `arxflex_borgerskolen` holds only
# the example data and a few helpers to find the columns of a room.
#

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.close("all")
plt.ion()

from arxflex import fitmodel, print_fit, fit_row, plot_fit, plot_daily_residuals
from arxflex_borgerskolen import load_processed_data, available_rooms, temperature_column

# %% [markdown]
# ## The data
#
# `load_processed_data()` returns one data frame: 1295 samples every 30 min,
# from 1 to 27 March 2021 (UTC). It holds the temperature and thermostat
# set-point of each room, plus the outdoor temperature `Ta` and the global
# radiation `Ps` from the DMI weather station in Sjaelsmark.

# %%
data = load_processed_data()
print(data.shape)
print(data["time"].iloc[[0, -1]].tolist())
available_rooms(data)

# %% [markdown]
# ### One room as a model data frame
#
# `fitmodel()` knows nothing about rooms or column names: every variable in a
# formula is simply a column of the data frame it gets. So the first step is
# always to build a small data frame with short, convenient names. Here:
#
# | column      | meaning                                              |
# |-------------|------------------------------------------------------|
# | `Ti`        | room temperature [°C], the output                    |
# | `Ta`        | outdoor temperature [°C]                             |
# | `Ps`        | global radiation [kW/m²]                              |
# | `Tset`      | thermostat set-point [°C]                            |
# | `Tfor`      | forward temperature of the heating, assumed 60 °C    |
# | `timeofday` | hour of the day (UTC), for the solar gain spline     |

# %%
TFOR = 60.0          # assumed forward temperature [°C]
PER_DAY = 48         # samples per day (30 min)


def room_frame(data, room):
    """The columns one room's models use, with short names."""
    time = data["time"]
    return pd.DataFrame({
        "time": time,
        "Ti": data[temperature_column(data, room)].to_numpy(float),
        "Ta": data["Ta"].to_numpy(float),
        "Ps": data["Ps"].to_numpy(float),
        "Tset": data["Tset_C" + room].to_numpy(float),
        "Tfor": TFOR,
        "timeofday": (time.dt.hour + time.dt.minute / 60).to_numpy(float),
    })


room = "0.07"
df = room_frame(data, room)
df.describe().round(2)

# %%
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
axes[0].plot(df["time"], df["Ti"], label="Ti", color="orangered")
axes[0].plot(df["time"], df["Tset"], label="Tset", color="grey", lw=1)
axes[0].set_ylabel("°C")
axes[0].legend()
axes[1].plot(df["time"], df["Ta"], color="steelblue")
axes[1].set_ylabel("Ta [°C]")
axes[2].plot(df["time"], df["Ps"], color="goldenrod")
axes[2].set_ylabel("Ps [kW/m²]")
for ax in axes:
    ax.grid(alpha=0.4)
fig.suptitle(f"Room C{room}")
fig.tight_layout()
################################################################


################################################################
model_sigmoid = {
    "AR": 1,
    "ambient": "Ta - Ti",
    "heating": "v(sigmoid, Tset - Ti) * (Tfor - Ti)",
}

fit_sigmoid1 = fitmodel(df, "Ti", model_sigmoid)
print_fit(fit_sigmoid1)

plot_fit(fit_sigmoid1, name=f"C{room} sigmoid", plots="simulation",
                per_day=PER_DAY, inputs=["Ta", "Ps"],
                regcontribution=[["ambient", "heating"]])

plot_fit(fit_sigmoid1, name=f"C{room} sigmoid", plots="residuals", per_day=PER_DAY)

plot_fit(fit_sigmoid1, name=f"C{room} sigmoid", plots="residual_series",
                per_day=PER_DAY, inputs=["Ta", "Ps"],
                regcontribution=[["ambient", "heating"]])

# Sunny against cloudy days, side by side on the same scales: a morning bump in
# the residuals that is there on sunny days only points to direct solar gain.
plot_daily_residuals(fit_sigmoid1, df, "Ps", 3.0, ["above", "below"],
                     name=f"Room C{room}", unit="kW/m²", daily_unit="kWh/m²")
################################################################


################################################################
# %% [markdown]
# ## Specifying a model
#
# A model is a dict of formula terms:
#
# - `"AR": 2` uses two lags of the output, Ti(k-1) and Ti(k-2).
# - `"ambient": "Ta - Ti"` is the heat loss to the outside, linear in the data.
# - `"solar": "s(timeofday, k=5) * Ps"` is the solar radiation multiplied by
#   a **B-spline over the time of day**. The spline describes how much of
#   the radiation reaches the room at each hour (window orientation, shading).
#   `k` is the number of knots.
# - `"heating": "v(sigmoid, Tset - Ti) * (Tfor - Ti)"` is the radiator. A
#   **sigmoid valve curve** in the distance to the set-point gives the valve
#   opening, which multiplies the temperature difference that drives the heat
#   flow.
#
# Each term gets one linear parameter per lag, found by least squares. The
# parameters inside `s()` and `v()` are nonlinear, and SLSQP optimises them.
# For every trial value the linear parameters are solved again.

# %%
model_sigmoid = {
    "AR": 2,
    "ambient": "Ta - Ti",
    "solar": "s(timeofday, k=5) * Ps",
    "heating": "v(sigmoid, Tset - Ti) * (Tfor - Ti)",
}

# %% [markdown]
# ## Fitting
#
# `fitmodel(data, y, model)` returns a dict with everything about the fit.
# `print_fit()` prints a summary: the optimiser result, the nonlinear
# parameters, the linear parameters, and the one-step and simulation errors.

# %%
fit_sigmoid = fitmodel(df, "Ti", model_sigmoid)
print_fit(fit_sigmoid)

# %% [markdown]
# Some of the entries in the result:

# %%
print("nonlinear parameters per term:", {k: np.round(v, 3) for k, v in fit_sigmoid["params_of"].items()})
print("one-step RMSE:  ", round(fit_sigmoid["rmse"], 4))
print("simulation RMSE:", round(fit_sigmoid["rmse_sim"], 4))

# %% [markdown]
# ## Analysing the fit
#
# ### Simulation
#
# The one-step prediction always starts from the measured temperature, so it
# looks good almost regardless of the model. A much stricter test is a
# **simulation**, where the model runs from the inputs only and feeds its own
# output back. `fitmodel()` does this by default and stores it in
# `fit["sim"]`.
#
# `plot_fit()` draws the measured and simulated output. Panels can be added
# below it: raw inputs (`inputs=`), and what each term contributes to the
# prediction (`regcontribution=`), in °C.

# %%
plot_fit(fit_sigmoid, name=f"C{room} sigmoid", plots="simulation",
                per_day=PER_DAY, inputs=["Ta", "Ps"],
                regcontribution=[["ambient", "solar", "heating"]])

# %% [markdown]
# ### Residuals
#
# The one-step residuals should look like white noise. Autocorrelation left
# at small lags means there is dynamics the model does not capture.

# %%
plot_fit(fit_sigmoid, name=f"C{room} sigmoid", plots="residuals", per_day=PER_DAY)

# %% [markdown]
# `plots="residual_series"` puts the residuals above the same panels as
# `"simulation"`, so large residuals can be matched with what the inputs and
# terms did at that time.

# %%
plot_fit(fit_sigmoid, name=f"C{room} sigmoid", plots="residual_series",
                per_day=PER_DAY, inputs=["Ta", "Ps"],
                regcontribution=[["ambient", "solar", "heating"]])

# %% [markdown]
# ### The estimated nonlinear functions
#
# `plots="blocks"` draws each `s()` and `v()` at the fitted parameters, with
# the knots marked. For the solar term this is the gain over the day. For the
# heating term it is the valve opening against `Tset - Ti`.

# %%
plot_fit(fit_sigmoid, name=f"C{room} sigmoid", plots="blocks")
plt.show()

# %% [markdown]
# The fitted functions are also available as callables, so they can be
# evaluated directly:

# %%
valve = fit_sigmoid["functions"]["heating"]
print("valve opening at Tset - Ti = -1, 0, 1:", np.round(valve([-1.0, 0.0, 1.0]), 3))

# %% [markdown]
# ## Other model structures
#
# Changing the model is a matter of changing a formula. Here are three
# alternatives to the sigmoid model:
#
# - **free knots:** the solar spline also estimates where its interior knots
#   are, with `knots="free"`.
# - **PCHIP valve:** a monotone spline replaces the sigmoid valve, which
#   allows a more flexible shape (`v(pchip, ..., n=5)`).
# - **no solar:** the solar term is left out, to see how much it explains.

# %%
models = {
    "sigmoid": model_sigmoid,
    "free knots": {
        "AR": 2,
        "ambient": "Ta - Ti",
        "solar": 's(timeofday, k=5, knots="free") * Ps',
        "heating": "v(sigmoid, Tset - Ti) * (Tfor - Ti)",
    },
    "pchip valve": {
        "AR": 2,
        "ambient": "Ta - Ti",
        "solar": "s(timeofday, k=4) * Ps",
        "heating": "v(pchip, Tset - Ti, n=5) * (Tfor - Ti)",
    },
    "no solar": {
        "AR": 2,
        "ambient": "Ta - Ti",
        "heating": "v(sigmoid, Tset - Ti) * (Tfor - Ti)",
    },
}

fits = {"sigmoid": fit_sigmoid}
for label, model in models.items():
    if label not in fits:
        fits[label] = fitmodel(df, "Ti", model)

# %% [markdown]
# `fit_row()` turns a fit into one row of a comparison table. `S` is the
# residual sum of squares of the one-step fit, and `rmse` is the simulation
# RMSE.

# %%
summary = pd.DataFrame([fit_row(f, label) for label, f in fits.items()]).set_index("model")
summary[["n_nonlinear", "success", "S", "sigma", "rmse_onestep", "rmse"]].round(4)

# %%
fig, ax = plt.subplots(figsize=(10, 3.5))
t = np.arange(len(df)) / PER_DAY
ax.plot(t, df["Ti"], color="black", lw=1, label="measured")
for label, f in fits.items():
    ax.plot(t, f["sim"], lw=1, label=f"{label} ({f['rmse_sim']:.3f})")
ax.set_xlabel("Time [days]")
ax.set_ylabel("Ti [°C]")
ax.legend(ncol=3, fontsize=9)
ax.grid(alpha=0.4)
ax.set_title(f"Room C{room}: simulations (RMSE in brackets)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Comparing the valve curves
#
# The fitted valve functions are callables, so the two curves can be drawn
# together. (`plot_fit(..., plots="blocks", block_xlim=(-4, 4))` gives the
# same x-range in separate figures.)

# %%
fig, ax = plt.subplots(figsize=(5, 3.6))
x = np.linspace(-4, 4, 300)
for label in ("sigmoid", "pchip valve"):
    ax.plot(x, fits[label]["functions"]["heating"](x), label=label)
ax.set_xlabel("Tset - Ti [°C]")
ax.set_ylabel("valve opening")
ax.legend()
ax.grid(alpha=0.4)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Validation on unseen data
#
# With `n_train` the nonlinear parameters are fitted on the first `n_train`
# samples only. With `refit_on_all_data=False` the linear parameters are
# fitted on them too, so the rest of the period is a clean test set.
# `rmse_train` and `rmse_test` give the simulation error on each part.

# %%
n_train = 20 * PER_DAY       # the first 20 days
fit_val = fitmodel(df, "Ti", model_sigmoid, n_train=n_train, refit_on_all_data=False)
print(f"simulation RMSE: training {fit_val['rmse_train']:.3f}, test {fit_val['rmse_test']:.3f}")
plot_fit(fit_val, name=f"C{room} sigmoid, validation", plots="simulation",
                per_day=PER_DAY)
plt.show()

# %% [markdown]
# ## All rooms
#
# Because the model is just a dict, fitting the same model to every room is
# a loop. Some rooms fit much better than others: the model assumes one
# radiator controlled by the room's own thermostat, which may not hold
# everywhere.

# %%
rows = []
for r in available_rooms(data):
    try:
        f = fitmodel(room_frame(data, r), "Ti", model_sigmoid)
        rows.append(fit_row(f, f"C{r}"))
    except Exception as err:          # e.g. too many missing values
        print(f"C{r}: {err}")
rooms_table = pd.DataFrame(rows).set_index("model")
rooms_table[["success", "S", "rmse_onestep", "rmse"]].sort_values("rmse").round(4)

# %% [markdown]
# ## Using your own data
#
# Nothing above is specific to Borgerskolen except `room_frame()`. For your
# own data:
#
# 1. Make a data frame with one column per variable, using the names you want
#    in the formulas.
# 2. Write the model as a dict of formulas. `s(x, ...)` gives a B-spline in
#    any variable, `v(sigmoid, x)` a sigmoid, `v(pchip, x, n=...)` a monotone
#    spline. A term may contain at most one of these, multiplied or divided
#    by data expressions.
# 3. Call `fitmodel(df, "y", model)` and look at the result with `print_fit`
#    and `plot_fit`.
#
# The options of each block are documented in its docstring, e.g.
# `help(arxflex.BSplineBlock)` and `help(arxflex.PchipBlock)`. New blocks can
# be added with `arxflex.register_block()`.
