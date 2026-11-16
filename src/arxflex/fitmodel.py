"""Fit an ARX model written as a dict of formulas.

    r = fitmodel(data, y="Ti", model={
            "AR":      2,
            "ambient": "Ta - Ti",
            "solar":   "s(timeofday, k=5) * Ps",
            "heating": "v(sigmoid, Tset - Ti) * (Tfor - Ti)",
        })

Every variable of the formulas, and the output variable y, is a column of the
data frame, so the data frame is prepared before the call and the function does
not know anything about rooms, buildings or naming conventions.

The model is

    y_k = sum over terms sum over lags i  theta * term_{k-i}  + e_k

with the output lags Ti(k-1)..Ti(k-Nd) as the first term ("AR": 0 leaves
them out, so the model is a regression on the lagged terms). For given nonlinear
parameters (spline coefficients, spline knots, valve parameters) the model is
linear in theta, so theta is solved by least squares inside the objective and
the optimiser only sees the nonlinear parameters. All nonlinear parameters of
all terms form one vector and are optimised together.

See nonlinear_blocks.py for the available functions s() and v() and their
options, and parse_formula.py for what a formula may contain.
"""

import numpy as np
import scipy.optimize as optim

from .compile_model import compile_model, AR_KEY, BAD_FIT
from .least_squares import least_squares
from .nonlinear_blocks import BlockError
from .simulate_model import simulate_model


def fitmodel(data, y, model, lags=None, n_train=None, refit_on_all_data=True,
             method="SLSQP", options=None, x0=None, callback=None, simulate=True,
             bad_fit=BAD_FIT):
    """Fit the model to the data.

    data   : data frame holding the output and all input variables
    y      : name of the output column
    model  : {"AR": number of output lags, term name: formula, ...}; "AR": 0
             gives a model without output lags
    lags   : number of lags of the input terms (default: the AR order, or 1
             when it is 0); a
             single term can override it with {"formula": ..., "lags": ...}
    n_train: fit the nonlinear parameters on the first n_train samples only
             (None: use all samples)
    refit_on_all_data : with n_train, estimate the linear parameters on all
             samples (True, as in run_valve_spline_model) or on the training
             samples only (False, a clean train/test split)
    method, options, x0 : passed to scipy.optimize.minimize; x0 defaults to the
             starting values of the blocks
    simulate: also simulate the output from the inputs only (default True)
    bad_fit: objective value returned for invalid nonlinear parameters

    Missing values: a NaN in the output or in an input enters the design matrix
    through every lag it appears in, and those rows are dropped from the least
    squares. One missing sample therefore costs L rows, L being the largest lag.
    The returned residuals and simulation keep the full length, with NaN where
    nothing could be computed, so they stay aligned with the data.

    Returns a dict with
      y, Nd, terms, model          what was fitted
      theta, names, theta_of       linear parameters, their names, per-term slices
      params, params_of, functions nonlinear parameters, per term, as callables
      S, sigma, resid, fitted      one-step fit: RSS, residual sd, residuals
      rmse                         one-step prediction RMSE
      n_used, rows_used            rows the least squares used, and their mask
      sim, sim_err                 simulation from the inputs only, and its error
      rmse_sim                     simulation RMSE over the whole period
      rmse_train, rmse_test        simulation RMSE before / after n_train
      res                          the optimiser result (None if nothing to optimise)
      compiled, fit_compiled       the CompiledModel on all / on the training data
    """
    full = compile_model(data, y, model, lags=lags, bad_fit=bad_fit)
    fit = full.restrict(n_train) if n_train else full

    res = None
    if fit.nparam:
        x0 = fit.init() if x0 is None else np.asarray(x0, float)
        if len(x0) != fit.nparam:
            raise ValueError(f"x0 has {len(x0)} values, the model has {fit.nparam} "
                             f"nonlinear parameters")
        cons = fit.constraints()
        kwargs = dict(bounds=fit.bounds(), method=method,
                      options={"maxiter": 500} if options is None else options)
        if cons is not None:
            A, lb, ub = cons
            kwargs["constraints"] = optim.LinearConstraint(A, lb, ub)
        res = optim.minimize(fit.objective, x0, callback=callback, **kwargs)
        params = res.x
    else:
        params = np.zeros(0)

    theta_model = full if (refit_on_all_data or not n_train) else fit
    X, yv, used = theta_model.design_used(params)
    if len(yv) <= theta_model.ntheta:
        raise ValueError(
            f"only {len(yv)} usable samples for {theta_model.ntheta} linear "
            "parameters; too many missing values")
    theta, S = least_squares(X, yv)

    # One-step predictions over the whole period. Rows that were dropped keep a
    # NaN here, so fitted and resid stay aligned with the data.
    Xall, yall = full.design(params)
    fitted = Xall @ theta
    resid = yall - fitted
    ok = np.isfinite(resid)
    dof = max(int(ok.sum()) - len(theta) - full.nparam, 1)

    r = dict(
        y=y, Nd=full.Nd, terms=full.terms, model=model, y_data=full.y_all,
        theta=theta, names=full.parameter_names(), theta_of=full.theta_slices(),
        params=params, params_of=full.unpack(params),
        functions={t.name: t.block.function(params[sl])
                   for t, sl in zip(full.terms, full.slices) if t.block is not None},
        blocks={t.name: t.block.describe(params[sl])
                for t, sl in zip(full.terms, full.slices) if t.block is not None},
        S=S, sigma=float(np.sqrt(resid[ok] @ resid[ok] / dof)), resid=resid, fitted=fitted,
        rmse=float(np.sqrt(np.mean(resid[ok] ** 2))),
        n_used=int(len(yv)), rows_used=used,
        res=res, n_train=n_train, compiled=full, fit_compiled=fit,
        sim=None, sim_err=None, rmse_sim=np.nan, rmse_train=np.nan, rmse_test=np.nan,
    )
    if simulate:
        sim = simulate_model(r)
        err = sim - full.y_all
        n = n_train

        def _rmse(e):
            e = e[np.isfinite(e)]
            return float(np.sqrt(np.mean(e ** 2))) if len(e) else np.nan

        r.update(sim=sim, sim_err=err, rmse_sim=_rmse(err),
                 rmse_train=_rmse(err[:n]) if n else np.nan,
                 rmse_test=_rmse(err[n:]) if n else np.nan)
    return r


def print_fit(r):
    """Print the parameters of a fit from fitmodel()."""
    c = r["compiled"]
    print(f"\noutput {r['y']}, Nd = {r['Nd']}, {c.n} samples"
          + (f", nonlinear parameters fitted on the first {r['n_train']}" if r["n_train"] else ""))
    for t in r["terms"]:
        print(f"  {t.name:<12} {t.text}" + (f"   [{t.lags} lags]" if t.lags != max(r["Nd"], 1) else ""))
    if r["res"] is not None:
        res = r["res"]
        print(f"\noptimiser: success={res.success}, S={res.fun:.4f}, nfev={res.nfev}, "
              f"message: {res.message}")
        for t in r["terms"]:
            if t.block is None:
                continue
            p = r["params_of"][t.name]
            print(f"  {t.name:<12} "
                  + "  ".join(f"{n}={v:.4g}" for n, v in zip(t.block.param_names(), p)))

    print(f"\n{'parameter':<22}{'estimate':>12}")
    for name, value in zip(r["names"], r["theta"]):
        print(f"{name:<22}{value:>12.5g}")
    sums = [f"{name} {r['theta'][sl].sum():.4g}" for name, sl in r["theta_of"].items()
            if sl.stop > sl.start]
    print("\nsum over lags:  " + "  ".join(sums))
    print(f"one-step RMSE: {r['rmse']:.4f}, residual sd: {r['sigma']:.4f}")
    if r["sim"] is not None:
        if r["n_train"]:
            print(f"simulation RMSE: training {r['rmse_train']:.3f}, "
                  f"after training period {r['rmse_test']:.3f}, all {r['rmse_sim']:.3f}")
        else:
            print(f"simulation RMSE: {r['rmse_sim']:.3f}")


def fit_row(r, label=None):
    """Key numbers of a fit as a flat dict, one row of a summary table."""
    return {
        "model": label,
        "y": r["y"],
        "Nd": r["Nd"],
        "terms": " + ".join(t.name for t in r["terms"]),
        "n_nonlinear": len(r["params"]),
        "success": bool(r["res"].success) if r["res"] is not None else True,
        "S": r["S"],
        "sigma": r["sigma"],
        "rmse_onestep": r["rmse"],
        "rmse": r["rmse_sim"],
        "rmse_train": r["rmse_train"],
        "rmse_test": r["rmse_test"],
    }
