"""One-step residuals of a fit and their autocorrelation."""

from .plot_helpers import TimeAxis


def plot_residuals(r, name="model", maxlag=40, per_day=None, time=None,
                   data=None, xlim=None):
    """One-step residuals over time (left) and their autocorrelation (right).

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
    ax2.set_xlim(0, maxlag)
    ax2.set_xlabel("lag")
    ax2.set_ylabel("autocorrelation")
    ax2.grid(alpha=0.4)
    fig.suptitle(f"{name}: one-step residuals, RMSE {r['rmse']:.4f}")
    fig.tight_layout()
    return fig
