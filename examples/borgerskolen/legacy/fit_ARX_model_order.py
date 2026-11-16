import numpy as np
import scipy.optimize as optim

from .construct_bspline_basisfunctions import construct_bspline_basisfunctions
from .solar_gain import solar_gain
from .sigmoid_valve import sigmoid_valve
from .construct_design_matrix import construct_design_matrix
from arxflex.least_squares import least_squares
from .fit_splines_and_thermostat_parameters import fit_splines_and_thermostat_parameters
from ._finish import _finish


def fit_ARX_model_order(data, roomname, bspline_order, samples_per_day, Tfor, criterion="bic",
                        Nend=None, spline_bounds=10, max_order=10, use_solar=True):
    """Choose the lag order Nd of the sigmoid model by AIC/BIC (notebook cells 40-41).

    samples_per_day : samples per day of the data (48 for the 30-min notebook data)
    Tfor            : assumed forward temperature [degC]
    Orders 1, 2, ... are fitted until the criterion gets worse; the previous
    order is returned. Estimation uses data[:Nend].
    use_solar=False : sigmoid model without solar radiation input.

    NOTE, fixes relative to the notebook:
      * it called construct_bspline_basisfunctions(n_bsplines=...), which fails
      * it mixed the thermostat parameters of the chosen order with the spline
        coefficients of the next order; here all come from the chosen order
      * it optimised on all data even when Nend was given
      * the criterion uses log(S/N) instead of log(S) (standard form; the
        number of observations changes with the order)
    """
    if criterion == "bic":
        def crit(S, p, N): return N * np.log(S / N) + p * np.log(N)
    elif criterion == "aic":
        def crit(S, p, N): return N * np.log(S / N) + 2 * p
    else:
        raise ValueError("criterion must be 'aic' or 'bic'")

    train = data.iloc[:Nend] if Nend is not None else data
    n_coef = bspline_order + 1 if use_solar else 0
    basis = (construct_bspline_basisfunctions(samples_per_day, bspline_order)
             if use_solar else None)
    best = None
    for order in range(1, max_order + 1):
        p0 = np.concatenate(([1, 0], np.ones(n_coef)))
        bounds = [(0.2, 5), (-spline_bounds, spline_bounds)] + [(-spline_bounds, spline_bounds)] * n_coef
        res = optim.minimize(fit_splines_and_thermostat_parameters, p0, bounds=bounds,
                             args=(train, roomname, order, basis, use_solar), method="SLSQP")
        valve = sigmoid_valve(*res.x[:2])
        X, y = construct_design_matrix(order, train, roomname, valve,
                                       solar_gain(len(train), basis, res.x[2:]) if use_solar else None,
                                       use_solar=use_solar)
        theta, S = least_squares(X, y)
        c = crit(S, len(theta), len(y))
        print(f"order {order}: {criterion.upper()} = {c:.2f}")
        if best is not None and c > best["criterion"]:
            break
        best = dict(order=order, res=res, theta=theta, S=S, valve=valve, criterion=c)

    Nd = best["order"]
    bspline_coef = best["res"].x[2:] if use_solar else None
    solar = solar_gain(len(data), basis, bspline_coef) if use_solar else None
    return _finish(data, roomname, Nd, best["theta"], best["S"], best["valve"], solar, best["res"],
                   Tfor=Tfor, n_train=Nend, use_solar=use_solar, basis=basis,
                   bspline_coef=bspline_coef,
                   criterion=best["criterion"])
