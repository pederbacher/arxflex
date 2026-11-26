"""Cross-correlation of the one-step residuals of a fit with its inputs."""

import numpy as np

from .plot_helpers import (INPUT_COLOURS, as_panels, data_column, reg_column,
                           term_panels)


def plot_ccf(r, name="model", inputs=(), reginput=(), data=None, maxlag=40):
    """Cross-correlation of the one-step residuals with inputs and terms.

    A significant correlation at lag > 0 means the input still explains part of
    the residual, e.g. a missing lag or a wrong shape of a nonlinear function.

    r       : the result dict from fitmodel()
    name    : name used in the title
    inputs  : one panel per entry; an entry of several columns draws them side
              by side in one panel, e.g. ["Ps", ["Ta", "Tset"]]
    reginput: term names, given as for `inputs`; the term's regression input,
              i.e. the formula evaluated at the fitted nonlinear parameters.
              With neither `inputs` nor `reginput`: one panel per term.
    data    : the data frame, needed only for `inputs` that no formula uses
    maxlag  : largest lag, in samples, e.g. the samples of one day

    Returns the matplotlib figure.
    """
    import matplotlib.pyplot as plt

    y, resid = r["y_data"], r["resid"]
    # resid is aligned with the rows L.. of the data; pad it to full length
    e = np.full(len(y), np.nan)
    e[len(y) - len(resid):] = resid
    lags = np.arange(-maxlag, maxlag + 1)
    bound = 1.959964 / np.sqrt(int(np.isfinite(e).sum()))   # 95% if e is white, as in R

    if not inputs and not reginput:
        reginput = "all"
    panels = [[(col, data_column(r, col, data)) for col in panel]
              for panel in as_panels(inputs)]
    panels += [[(f"{nm}\n(regression input)" if len(panel) == 1 else nm,
                 reg_column(r, nm)) for nm in panel]
               for panel in term_panels(r, reginput, "reginput")]

    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 1.2 + 1.9 * len(panels)),
                             sharex=True, squeeze=False)
    for ax, panel in zip(axes[:, 0], panels):
        width = 0.8 / len(panel)            # series of one panel side by side
        for j, (nm, x) in enumerate(panel):
            c = _ccf(e, np.asarray(x, float), lags)
            pos = lags + (j - (len(panel) - 1) / 2) * width
            ax.vlines(pos, 0, c, color=INPUT_COLOURS[j % len(INPUT_COLOURS)],
                      lw=1.5, label=nm.split("\n")[0])
        ax.axhline(0, color="black", lw=0.6)
        for b in (bound, -bound):
            ax.axhline(b, color="blue", ls="--", lw=0.8)
        ax.axvline(0, color="grey", lw=0.6, ls=":")
        ax.set_ylabel(" / ".join(nm for nm, _ in panel))
        if len(panel) > 1:
            ax.legend(fontsize=8, loc="upper right", ncol=min(len(panel), 3))
        ax.grid(alpha=0.4)
    axes[-1, 0].set_xlabel("lag [samples]: corr(residual(k), input(k - lag)); "
                           "lag > 0: the input leads")
    fig.suptitle(f"{name}: cross-correlation of one-step residuals and inputs "
                 f"(dashed: ±1.96/√N)")
    fig.tight_layout()
    return fig


def _ccf(e, x, lags):
    """corr(e(k), x(k - lag)) for each lag, over the pairs where both are
    finite; e and x are standardised with all their finite values."""
    ok_e, ok_x = np.isfinite(e), np.isfinite(x)
    e = (e - e[ok_e].mean()) / e[ok_e].std()
    x = (x - x[ok_x].mean()) / (x[ok_x].std() or 1.0)
    n, out = len(e), np.full(len(lags), np.nan)
    for i, lag in enumerate(lags):
        if lag >= 0:
            a, b = e[lag:], x[:n - lag]
        else:
            a, b = e[:n + lag], x[-lag:]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.any():
            out[i] = np.mean(a[ok] * b[ok])
    return out
