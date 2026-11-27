"""Autocorrelation of the one-step residuals of a fit."""

import numpy as np


def plot_acf(r, name="model", maxlag=40):
    """Autocorrelation of the one-step residuals, with the 95% band
    ±1.96/√N for white noise as R's acf() draws it (N residuals;
    qnorm(0.975) = 1.959964): residuals that are white stay inside it at about
    19 of 20 lags.

    r      : the result dict from fitmodel()
    name   : name used in the title
    maxlag : largest lag, in samples

    Returns the matplotlib figure.
    """
    import matplotlib.pyplot as plt

    resid = r["resid"]
    maxlag = min(maxlag, len(resid) - 1)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.acorr(resid - resid.mean(), maxlags=maxlag, usevlines=True, normed=True, lw=1)
    bound = 1.959964 / np.sqrt(int(np.isfinite(resid).sum()))
    for b in (bound, -bound):
        ax.axhline(b, color="blue", ls="--", lw=0.8)
    ax.set_xlim(0, maxlag)
    ax.set_xlabel("lag")
    ax.set_ylabel("autocorrelation")
    ax.grid(alpha=0.4)
    ax.set_title(f"{name}: ACF of the one-step residuals (dashed: ±1.96/√N)", fontsize=10)
    fig.tight_layout()
    return fig
