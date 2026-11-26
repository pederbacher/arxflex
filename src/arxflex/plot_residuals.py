"""One-step residuals of a fit and their autocorrelation."""

import numpy as np

from .plot_helpers import TimeAxis


def plot_residuals(r, name="model", maxlag=40, per_day=None, time=None,
                   data=None, xlim=None):
    """One-step residuals over time (left) and their autocorrelation (right).

    The dashed lines are the 95% band ±1.96/√N for the autocorrelation of
    white noise, as R's acf() draws it (N residuals; qnorm(0.975) = 1.959964):
    residuals that are white stay inside it at about 19 of 20 lags.

    r       : the result dict from fitmodel()
    name    : name used in the title
    maxlag  : largest lag of the autocorrelation, in samples
    per_day, time, data, xlim : the time axis, as in plot_simulation()

    Returns the matplotlib figure.
    """
    import matplotlib.pyplot as plt

    resid = r["resid"]
    axis = TimeAxis(r, per_day, time, data, xlim)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2),
                                   gridspec_kw={"width_ratios": [2.2, 1]})
    ax1.plot(axis.t[r["compiled"].L:], resid, color="grey", lw=0.8)
    ax1.set_xlabel(axis.label)
    ax1.set_ylabel("one-step residual")
    ax1.grid(alpha=0.4)
    axis.apply([ax1])
    maxlag = min(maxlag, len(resid) - 1)
    ax2.acorr(resid - resid.mean(), maxlags=maxlag, usevlines=True, normed=True, lw=1)
    bound = 1.959964 / np.sqrt(int(np.isfinite(resid).sum()))
    for b in (bound, -bound):
        ax2.axhline(b, color="blue", ls="--", lw=0.8)
    ax2.set_xlim(0, maxlag)
    ax2.set_xlabel("lag")
    ax2.set_ylabel("autocorrelation")
    ax2.set_title("dashed: ±1.96/√N", fontsize=9)
    ax2.grid(alpha=0.4)
    fig.suptitle(f"{name}: one-step residuals, RMSE {r['rmse']:.4f}")
    fig.tight_layout()
    return fig
