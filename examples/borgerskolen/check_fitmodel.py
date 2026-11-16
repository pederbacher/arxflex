"""Check that arxflex.fitmodel() reproduces the older, hand-written models
(legacy/, ported from legacy/ARXmodelling_experiment.ipynb).

For each of run_sigmoid_model, run_free_knots_model and run_valve_spline_model
this checks three things:

  1. the objective is the same function - the old optimum, written in the new
     parameter order, gives the same residual sum of squares,
  2. fitting the formula model from scratch ends in the same place (small
     differences are the SLSQP path, since the parameter vector is ordered by
     term instead of valve-parameters-first),
  3. the generic simulation gives the same trajectory as the hard-coded
     predict_multistep_ARX_model for the same parameters.

Run:  python check_fitmodel.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
sys.path.insert(0, str(HERE))

from arxflex import fitmodel, compile_model, simulate_model
from arxflex_borgerskolen import load_processed_data, temperature_column
from legacy import run_sigmoid_model, run_free_knots_model, run_valve_spline_model

# =============================================================================
# SETTINGS
# =============================================================================

# The modelling assumptions about the data set. The functions take these as
# arguments, so they are set here rather than inside the package.
TFOR = 60.0                  # assumed forward temperature [degC]
SAMPLES_PER_DAY = 48         # Ts = 30 min

Nd = 2                       # number of output lags
room = "0.07"                # sigmoid, free-knots and no-solar checks
room_valve = "1.09"          # valve check (the notebook used this room)
bspline_order = 5            # solar spline, sigmoid and free-knots models
bspline_order_valve = 4      # solar spline, valve model
sigmoid_bound = 10           # box bound on the sigmoid valve parameters
N_valve = 5                  # PCHIP valve knots
n_train = 850                # samples used for the nonlinear parameters, valve model

# Each check fits the same model twice, once with an old run_* function and once
# as a formula, so these two must match what the old functions do internally.
# Both are still hardcoded in legacy/, so change them here only together
# with the legacy function named in the comment.
knot_sum = (5, 22)           # free knots: last interior knot in [5, 22] h
                             # (construct_linear_inequality_constraints)
valve_start = 0.6            # starting value of the spline coefficients
                             # (run_valve_spline_model)

# =============================================================================
# The models, written as formulas
# =============================================================================

AMBIENT = "Ta - Ti"
HEAT_SIG = "v(sigmoid, Tset - Ti) * (Tfor - Ti)"
HEAT_PCHIP = f"v(pchip, Tset - Ti, n={N_valve}) * (Tfor - Ti)"
SOLAR = f"s(timeofday, k={bspline_order}) * Ps"
SOLAR_FREE = (f's(timeofday, k={bspline_order}, knots="free", '
              f"knot_sum={knot_sum}) * Ps")
SOLAR_VALVE = f"s(timeofday, k={bspline_order_valve}, start={valve_start}) * Ps"


def prepare_room(data, room):
    """A plain data frame for one room: the columns the formulas refer to.

    This is the step fitmodel() does not do: pick the columns of one room and
    give them names the formulas can use.
    """
    n = len(data)
    # NOTE: as in the older functions the time of day comes from the row number,
    # on a grid of 48 points spanning [0, 24] (spacing 24/47 h, not 30 min)
    day = np.linspace(0.0, 24.0, SAMPLES_PER_DAY)
    return pd.DataFrame({
        "Ti": data[temperature_column(data, room)].to_numpy(float),
        "Ta": data["Ta"].to_numpy(float),
        "Ps": data["Ps"].to_numpy(float),
        "Tset": data["Tset_C" + room].to_numpy(float),
        "Tfor": np.full(n, TFOR),
        "timeofday": day[np.arange(n) % SAMPLES_PER_DAY],
    })


def check(label, old, df, model, reorder, n_train=None):
    """Compare one old model with the same model written as formulas.

    reorder : the old nonlinear parameters in the new (term by term) order
    """
    cm = compile_model(df, "Ti", model)
    fit_cm = cm.restrict(n_train) if n_train else cm
    x_old = reorder(old["res"].x)

    same_objective = abs(fit_cm.objective(x_old) - old["res"].fun)
    sim_old_params = simulate_model({"compiled": cm, "theta": old["theta"], "params": x_old})
    same_sim = np.max(np.abs(sim_old_params - old["sim"]))

    new = fitmodel(df, "Ti", model, n_train=n_train)
    dS = abs(old["res"].fun - new["res"].fun)
    dtheta = np.max(np.abs(np.asarray(old["theta"]) - new["theta"]))

    print(f"{label:<11} objective at old optimum: dS={same_objective:.2e}   "
          f"simulation with old parameters: max|dy|={same_sim:.2e}")
    print(f"{'':<11} refitted: S_old={old['res'].fun:.8f} S_new={new['res'].fun:.8f} "
          f"dS={dS:.2e} max|dtheta|={dtheta:.2e} "
          f"rmse_sim {old_rmse(old):.4f} / {new['rmse_sim']:.4f}")
    return new


def old_rmse(old):
    err = old["sim"] - old["Ti"]
    return float(np.sqrt(np.mean(err ** 2)))


data = load_processed_data()
df = prepare_room(data, room)
print(f"room C{room}, Nd = {Nd}\n")

# --- sigmoid: old parameters are [alpha, offset, coef...] --------------------
check("sigmoid",
      run_sigmoid_model(data, SAMPLES_PER_DAY, TFOR, room, Nd,
                        bspline_order=bspline_order, bound=sigmoid_bound),
      df,
      {"AR": Nd, "ambient": AMBIENT, "solar": SOLAR, "heating": HEAT_SIG},
      lambda x: np.concatenate((x[2:], x[:2])))

# --- free knots: old parameters are [alpha, offset, coef(6), knots(3)] -------
check("freeknots",
      run_free_knots_model(data, SAMPLES_PER_DAY, TFOR, room, Nd,
                           bspline_order=bspline_order),
      df,
      {"AR": Nd,
       "ambient": AMBIENT,
       "solar": SOLAR_FREE,
       "heating": HEAT_SIG},
      lambda x: np.concatenate((x[2:], x[:2])))

# --- valve: old parameters are already in term order ------------------------
df_v = prepare_room(data, room_valve)
print(f"\n(valve model on room C{room_valve})")
check("valve",
      run_valve_spline_model(data, SAMPLES_PER_DAY, TFOR, room_valve, Nd,
                             bspline_order=bspline_order_valve, N_valve=N_valve,
                             n_train=n_train),
      df_v,
      {"AR": Nd,
       "ambient": AMBIENT,
       "solar": SOLAR_VALVE,
       "heating": HEAT_PCHIP},
      lambda x: x, n_train=n_train)

# --- no solar input ---------------------------------------------------------
print()
check("no solar",
      run_sigmoid_model(data, SAMPLES_PER_DAY, TFOR, room, Nd, use_solar=False),
      df,
      {"AR": Nd, "ambient": AMBIENT, "heating": HEAT_SIG},
      lambda x: x)
