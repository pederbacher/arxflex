"""Helpers shared by the plot functions: the time axis, the panels of inputs
and regression terms, and the stacked figure of plot_simulation() and
plot_residuals()."""

import numpy as np

# Series colours within one panel; all series of a panel share one y-axis.
INPUT_COLOURS = ("steelblue", "goldenrod", "purple", "teal")
REG_COLOURS = ("seagreen", "firebrick", "purple", "goldenrod")


def as_panels(spec):
    """"a" or ["a", ["b", "c"]] -> [["a"]] / [["a"], ["b", "c"]]: one entry per
    panel, each holding the names drawn in it."""
    spec = [spec] if isinstance(spec, str) else list(spec)
    return [[p] if isinstance(p, str) else list(p) for p in spec]


def term_panels(r, spec, arg):
    """Panels of term names, as as_panels(); "all" gives one panel per term.
    `arg` is the argument name, for the error message."""
    names = [t.name for t in r["compiled"].terms]
    panels = [[nm] for nm in names] if spec == "all" else as_panels(spec)
    unknown = [t for panel in panels for t in panel if t not in names]
    if unknown:
        raise ValueError(f"unknown term(s) {unknown} in {arg}; the model has {names}")
    return panels


def term_text(r, term_name):
    return next(t.text for t in r["compiled"].terms if t.name == term_name)


def data_column(r, col, data):
    """An input column: from `data`, or from the columns the model read."""
    if data is not None and col in data:
        return np.asarray(data[col], float)
    if col in r["compiled"].columns:
        return r["compiled"].columns[col]
    raise KeyError(f"{col!r} is not a column of the model; pass data=...")


def reg_column(r, term_name):
    """The term's column in the design matrix, before lagging and theta: the
    formula evaluated at the fitted nonlinear parameters."""
    cm = r["compiled"]
    for term, sl in zip(cm.terms, cm.slices):
        if term.name == term_name:
            return np.asarray(term.column(r["params"][sl]), float)
    raise KeyError(term_name)


def contribution(r, term_name):
    """sum_i theta_i * term(k-i): the term's part of the one-step prediction.
    theta of a term is ordered oldest lag first, as in CompiledModel._lags();
    the first `lags` samples have no full lag window and are NaN."""
    x = reg_column(r, term_name)
    th = np.asarray(r["theta"], float)[r["theta_of"][term_name]]
    q, out = len(th), np.full(len(x), np.nan)
    out[q:] = sum(th[i] * x[i:len(x) - q + i] for i in range(q))
    return out


class TimeAxis:
    """The time axis of a fit: samples, days (per_day) or time stamps (time).

    t       : x values, one per row of the data
    label   : x-axis label
    xlim    : the range to show, converted to the units of t (or None)
    t_train : where the training period ends, on the axis (or None)
    """

    def __init__(self, r, per_day=None, time=None, data=None, xlim=None):
        n = len(r["y_data"])
        self.dates = time is not None
        if self.dates:
            t = _time_stamps(data[time] if isinstance(time, str) else time)
            if len(t) != n:
                raise ValueError(f"time has {len(t)} values, but the data has {n} rows")
            self.label = "Time"
            if xlim is not None:
                xlim = [_time_stamp(v, t) for v in xlim]
        else:
            t = np.arange(n) / (per_day if per_day else 1)
            self.label = "Time [days]" if per_day else "Sample"
        self.t, self.xlim = t, xlim
        n_train = r["n_train"]
        self.t_train = t[n_train] if n_train and n_train < n else None

    def apply(self, axes):
        """Readable date labels with time stamps, and the range xlim with each
        y-axis scaled to the data inside it."""
        if self.dates:
            import matplotlib.dates as mdates
            loc = mdates.AutoDateLocator()
            for ax in axes:
                ax.xaxis.set_major_locator(loc)
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        if self.xlim is not None:
            for ax in axes:
                ax.set_xlim(*self.xlim)          # also for axes that don't share x
                _fit_ylim(ax)


def stacked_figure(r, draw_top, title, axis, inputs=(), reginput=(),
                   regcontribution=(), data=None):
    """A figure with the top panel drawn by draw_top(ax, t), and one panel per
    entry of `inputs`, `reginput` and `regcontribution` stacked under it.
    draw_top may also be a list of such functions, one top panel each."""
    import matplotlib.pyplot as plt

    inputs = as_panels(inputs)
    reginput = term_panels(r, reginput, "reginput")
    regcontribution = term_panels(r, regcontribution, "regcontribution")
    t = axis.t

    def draw_panel(ax, series, colours):
        """series: [(ylabel, legend label, values), ...] drawn in one panel.
        All series of a panel share one y-axis, so their levels are directly
        comparable; put quantities of different units in separate panels."""
        for j, (_, lab, vals) in enumerate(series):
            ax.plot(t, vals, color=colours[j % len(colours)], lw=1, label=lab)
        ax.grid(alpha=0.4)
        ax.set_ylabel(" / ".join(nm for nm, _, _ in series))
        if len(series) > 1:
            ax.legend(fontsize=8, loc="upper right", ncol=min(len(series), 3))
        if axis.t_train is not None:
            ax.axvline(axis.t_train, color="grey", ls=":")

    def draw_terms(axes, panels, values, what):
        for ax, panel in zip(axes, panels):
            series = []
            for nm in panel:
                text = term_text(r, nm)
                series.append((nm if len(panel) > 1 else f"{nm}\n({what})",
                               f"{nm}: {text}" if len(panel) > 1 else text,
                               values(r, nm)))
            draw_panel(ax, series, REG_COLOURS)

    tops = list(draw_top) if isinstance(draw_top, (list, tuple)) else [draw_top]
    ntop = len(tops)
    nrows = ntop + len(inputs) + len(reginput) + len(regcontribution)
    fig, axes = plt.subplots(nrows, 1,
                             figsize=(11, 2.8 + 2.3 * (ntop - 1) + 1.5 * (nrows - ntop)),
                             sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [1.8] * ntop + [1] * (nrows - ntop)})
    axes = axes[:, 0]
    for i, (ax, draw) in enumerate(zip(axes, tops)):
        draw(ax, t)
        if axis.t_train is not None:
            ax.axvline(axis.t_train, color="grey", ls=":",
                       label="end of training" if i == 0 else None)
        ax.legend(ncol=3, fontsize=9)
        ax.grid(alpha=0.4)

    rows = iter(axes[ntop:])
    for panel in inputs:
        draw_panel(next(rows), [(c, c, data_column(r, c, data)) for c in panel],
                   INPUT_COLOURS)
    # The terms as the regression sees them: the formula evaluated at the
    # fitted nonlinear parameters, before lagging and before theta.
    draw_terms([next(rows) for _ in reginput], reginput, reg_column,
               "regression input")
    # The same multiplied by the fitted theta and summed over the lags: what
    # the term adds to the one-step prediction, in units of the output.
    draw_terms([next(rows) for _ in regcontribution], regcontribution,
               contribution, "contribution")
    axes[-1].set_xlabel(axis.label)
    axis.apply(axes)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _time_stamps(time):
    """Time stamps as a pandas DatetimeIndex (keeping the time zone), or as
    numpy datetime64 when pandas isn't installed."""
    try:
        import pandas as pd
    except ImportError:
        return np.asarray(time, dtype="datetime64[ns]")
    return pd.DatetimeIndex(time)


def _time_stamp(value, t):
    """One end of xlim as a time stamp comparable with `t`: a string or a
    naive time stamp is taken to be in the time zone of `t`."""
    if not hasattr(t, "tz"):                 # numpy datetime64
        return np.datetime64(value, "ns")
    import pandas as pd
    v = pd.Timestamp(value)
    if t.tz is not None and v.tz is None:
        v = v.tz_localize(t.tz)
    elif t.tz is None and v.tz is not None:
        v = v.tz_convert(None)
    return v


def _fit_ylim(ax, margin=0.05):
    """Scale the y-axis to the plotted data that lies inside the x-range.
    Only lines drawn in data coordinates count, not axvline/axhline."""
    lo, hi = sorted(ax.get_xlim())
    ys = []
    for line in ax.get_lines():
        if line.get_transform() != ax.transData:
            continue
        xy = np.asarray(line.get_xydata(), float)   # x already as numbers
        inside = (xy[:, 0] >= lo) & (xy[:, 0] <= hi) & np.isfinite(xy[:, 1])
        ys.append(xy[inside, 1])
    ys = np.concatenate(ys) if ys else np.zeros(0)
    if ys.size:
        y0, y1 = ys.min(), ys.max()
        pad = (y1 - y0) * margin or abs(y0) * margin or 1.0
        ax.set_ylim(y0 - pad, y1 + pad)
