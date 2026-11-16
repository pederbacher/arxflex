"""Simulate a fitted model from the inputs only.

A one-step prediction uses the measured output at k-1, k-2, ...; a simulation
uses the values it produced itself, so a term like (Ta - Ti) or
v(Tset - Ti)*(Tfor - Ti), which depends on the output, has to be recomputed at
every step from the simulated output. Term.bind() does that: the formula is
parsed once and only the input values change, and the block keeps its fitted
parameters.

This replaces the hard-coded predict_multistep_ARX_model() for models built
with fitmodel(); the terms come from the formulas instead of being fixed.

Terms must be pointwise in time - a formula is a function of the values of its
variables at the same sample - which is what makes this possible.

Missing values: a step whose history or inputs are not fully finite cannot be
computed, and the simulation restarts from the measured output instead of
carrying the NaN forward for the rest of the period. A gap therefore costs L
simulated samples and the trajectory resumes after it, which is what makes a
frame stitched from several periods usable.
"""

import numpy as np


def simulate_model(r, y0=None, length=None):
    """Simulate the output of a model fitted with fitmodel().

    r      : the result dict from fitmodel()
    y0     : the first L output values to start from (default: the measured ones,
             L = the largest number of lags in the model)
    length : number of samples to simulate (default: all of them)

    Returns the simulated output, an array of that length.
    """
    cm = r["compiled"]
    theta, params = np.asarray(r["theta"], float), np.asarray(r["params"], float)
    L, Nd = cm.L, cm.Nd
    n = cm.n if length is None else int(length)
    if n > cm.n:
        raise ValueError(f"cannot simulate {n} samples, the data has {cm.n}")

    y = np.zeros(n)
    y[:L] = cm.y_all[:L] if y0 is None else np.asarray(y0, float)[:L]

    theta_of = cm.theta_slices()
    theta_ar = theta[:Nd]

    # Terms that do not depend on the output are the same as when fitting
    static, dynamic = [], []
    for t, sl in zip(cm.terms, cm.slices):
        entry = (t, theta[theta_of[t.name]], t.lags)
        if t.uses_y:
            dynamic.append(entry + (params[sl],))
        else:
            static.append((t.column(params[sl]),) + entry[1:])

    # The columns the dynamic terms read, as views over the whole period
    needed = sorted({v for t, _, _, _ in dynamic for v in t.variables} - {cm.y_name})
    cols = {v: cm.columns[v] for v in needed}

    for i in range(n - L):
        k = i + L
        window = y[i:k]                      # the values of the last L samples
        value = theta_ar @ window[::-1][:Nd]
        for col, th, q in static:
            value += th @ col[k - q:k]
        if dynamic:
            env = {v: a[i:k] for v, a in cols.items()}
            env[cm.y_name] = window
            for t, th, q, p in dynamic:
                value += th @ t.bind(env).column(p)[L - q:]
        # Across a gap the value is NaN; restart from the measurement so the
        # simulation recovers instead of staying NaN to the end.
        y[k] = value if np.isfinite(value) else cm.y_all[k]
    return y
