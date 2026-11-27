"""One-step residuals of a fit as a time series, with inputs and terms below."""

from .plot_helpers import TimeAxis, stacked_figure


def plot_residual_series(r, name="model", inputs=(), reginput=(),
                         regcontribution=(), data=None, per_day=None, time=None,
                         xlim=None):
    """One-step residuals as a time series, with panels of inputs and terms
    below, to see which input a pattern in the residuals follows.

    The arguments are those of plot_simulation(), except `top`. Returns the
    matplotlib figure.
    """
    L = r["compiled"].L

    def draw(ax, t):
        ax.plot(t[L:], r["resid"], color="grey", lw=0.8, label="one-step residual")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel(f"{r['y']} residual")

    title = f"{name}: one-step residuals of {r['y']}, RMSE {r['rmse']:.4f}"
    axis = TimeAxis(r, per_day, time, data, xlim)
    return stacked_figure(r, draw, title, axis, inputs, reginput,
                          regcontribution, data)
