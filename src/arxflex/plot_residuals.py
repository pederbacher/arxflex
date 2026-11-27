"""One-step residuals and prediction of a fit as time series, with inputs and
terms below."""

from .plot_helpers import TimeAxis, data_column, stacked_figure

TOP_COLOURS = ("grey", "purple", "teal")


def plot_residuals(r, name="model", inputs=(), reginput=(), regcontribution=(),
                   data=None, per_day=None, time=None, xlim=None, top=()):
    """The one-step residuals (top), the one-step prediction with the
    observations (below it), and then panels of inputs and terms, all on one
    time axis: to find what the inputs and terms did where the residuals are
    large, zoomed in with `xlim`.

    The arguments are those of plot_simulation(); `top` adds columns to the
    panel of the prediction and observations, e.g. ["Tset"].

    Returns the matplotlib figure.
    """
    L = r["compiled"].L
    y_name = r["y"]

    def draw_residuals(ax, t):
        ax.plot(t[L:], r["resid"], color="grey", lw=0.8, label="one-step residual")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel(f"{y_name} residual")

    def draw_prediction(ax, t):
        ax.plot(t, r["y_data"], color="orangered", lw=1, label=f"{y_name} measured")
        ax.plot(t[L:], r["fitted"], color="deepskyblue", lw=1, label="one-step prediction")
        for j, col in enumerate([top] if isinstance(top, str) else top):
            ax.plot(t, data_column(r, col, data), color=TOP_COLOURS[j % len(TOP_COLOURS)],
                    lw=1, label=col, zorder=1)          # behind the output
        ax.set_ylabel(y_name)

    title = f"{name}: one-step residuals and prediction of {y_name}, RMSE {r['rmse']:.4f}"
    axis = TimeAxis(r, per_day, time, data, xlim)
    return stacked_figure(r, [draw_residuals, draw_prediction], title, axis,
                          inputs, reginput, regcontribution, data)
