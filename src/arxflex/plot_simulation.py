"""Measured and simulated output of a fit, with inputs and terms below."""

from .plot_helpers import TimeAxis, data_column, stacked_figure

TOP_COLOURS = ("grey", "purple", "teal")


def plot_simulation(r, name="model", inputs=(), reginput=(), regcontribution=(),
                    data=None, per_day=None, time=None, xlim=None, top=()):
    """Measured and simulated output, with panels of inputs and terms below.

    r       : the result dict from fitmodel()
    name    : name used in the title
    inputs  : one panel per entry, e.g. ["Ta", ["Ta", "Ti"]] draws Ta alone and
              then Ta and Ti together in one panel on a shared y-axis, so their
              levels can be compared; taken from `data`, or from the columns the
              model already read
    reginput: term names, e.g. "heating", ["solar", "heating"] for two panels,
              or [["solar", "heating"]] for both in one panel; plots the column
              that term contributes to the design matrix, i.e. the formula
              evaluated at the fitted nonlinear parameters, which is how the
              nonlinear function actually enters the regression; "all" gives
              one panel per term of the model
    regcontribution: term names, given as for `reginput`; plots what the term
              adds to the one-step prediction, i.e. its regression input
              multiplied by the fitted theta of each lag and summed over the
              lags, sum_i theta_i * term(k-i); "all" as for `reginput`
    data    : the data frame, needed only for `inputs` that no formula uses, or
              when `time` is a column name
    per_day : samples per day; given, the time axis is in days instead of samples
    time    : time stamps of the samples, one per row of the data the model was
              fitted to, or the name of that column in `data`; given, the time
              axis shows them (and `per_day` is not needed)
    xlim    : (lo, hi) range of the time axis. With `time`, give time stamps,
              e.g. ["2021-03-01 10:00", "2021-03-01 12:00"]; a string without a
              time zone is read in the time zone of `time`. Without `time`, give
              days (with `per_day`) or sample numbers. The y-axes are scaled to
              the data inside the range.
    top     : columns drawn in the upper panel with the measured and simulated
              output, on the same axis, e.g. ["Tset"] for the set-point; taken
              from `data`, or from the columns the model already read

    Returns the matplotlib figure.
    """
    y, sim = r["y_data"], r["sim"]

    def draw(ax, t):
        ax.plot(t, y, color="orangered", lw=1, label=f"{r['y']} measured")
        if sim is not None:
            ax.plot(t, sim, color="deepskyblue", lw=1.5, label="simulation")
        for j, col in enumerate([top] if isinstance(top, str) else top):
            ax.plot(t, data_column(r, col, data), color=TOP_COLOURS[j % len(TOP_COLOURS)],
                    lw=1, label=col, zorder=1)          # behind the output
        ax.set_ylabel(r["y"])

    title = f"{name}: {r['y']}, Nd = {r['Nd']}"
    if sim is not None:
        title += f", simulation RMSE {r['rmse_sim']:.3f}"
    axis = TimeAxis(r, per_day, time, data, xlim)
    return stacked_figure(r, draw, title, axis, inputs, reginput,
                          regcontribution, data)
