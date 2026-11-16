import numpy as np


def fit_summary(r, model=None):
    """Key numbers of a fitted model as a flat dict (one row of a summary table).

    r     : result dict from run_valve_spline_model, run_sigmoid_model,
            run_free_knots_model or fit_ARX_model_order
    model : model name to put in the row, e.g. "valve"

    Columns:
      room, model, Nd, use_solar
      success, n_nonlinear    optimiser converged, number of optimised parameters
      S                       residual sum of squares of the one-step predictions
      rmse                    simulation RMSE over the whole period [degC]
      rmse_train, rmse_test   simulation RMSE before/after n_train
                              (NaN for models fitted on all data)
    """
    err = r["sim"] - r["Ti"]
    n = r["n_train"]
    return {
        "room": r["roomname"],
        "model": model,
        "Nd": r["Nd"],
        "use_solar": r.get("use_solar", True),
        "success": bool(r["res"].success),
        "n_nonlinear": len(r["res"].x),
        "S": r["S"],
        "rmse": np.sqrt(np.mean(err ** 2)),
        "rmse_train": np.sqrt(np.mean(err[:n] ** 2)) if n else np.nan,
        "rmse_test": np.sqrt(np.mean(err[n:] ** 2)) if n else np.nan,
    }
