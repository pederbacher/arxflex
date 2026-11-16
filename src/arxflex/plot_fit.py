"""Figures for a model fitted with fitmodel().

Unlike plot_results(), which knows the Borgerskolen columns, this works for any
model: it plots the measured and simulated output, and one figure per nonlinear
function in the model, whatever the formulas were.
"""

import numpy as np

from .nonlinear_blocks import BSplineBlock, PchipBlock

PLOT_TYPES = ("simulation", "residual_series", "residuals", "blocks")
# Series colours within one panel; all series of a panel share one y-axis.
INPUT_COLOURS = ("steelblue", "goldenrod", "purple", "teal")
REG_COLOURS = ("seagreen", "firebrick", "purple", "goldenrod")


def plot_fit(r, outdir=None, name="model", plots=PLOT_TYPES, inputs=(), data=None,
             per_day=None, show_knots=True, reginput=(), regcontribution=(),
             block_xlim=None, time=None, xlim=None):
    """Make (and optionally save) figures of a fit from fitmodel().

    r       : the result dict from fitmodel()
    outdir  : folder to save the figures in as <name>_<plot>.png (None: don't save)
    name    : name used in the titles and file names
    plots   : any of
              "simulation"      measured and simulated output, with the panels
                                of `inputs`, `reginput`, `regcontribution` below
              "residual_series" the same, with the one-step residuals on top
              "residuals"       one-step residuals and their autocorrelation
              "blocks"          one figure per nonlinear function
    inputs  : one panel per entry, e.g. ["Ta", ["Ta", "Ti"]] draws Ta alone and
              then Ta and Ti together in one panel on a shared y-axis, so their
              levels can be compared; taken from `data`, or from the columns the
              model already read
    reginput: term names, e.g. "heating", ["solar", "heating"] for two panels,
              or [["solar", "heating"]] for both in one panel; plots the
              column that term contributes to the design matrix, i.e. the
              formula evaluated at the fitted nonlinear parameters, which is
              how the nonlinear function actually enters the regression;
              "all" gives one panel per term of the model
    regcontribution: term names, given as for `reginput`; plots what the term
              adds to the one-step prediction, i.e. its regression input
              multiplied by the fitted theta of each lag and summed over the
              lags, sum_i theta_i * term(k-i); "all" as for `reginput`
    data    : the data frame, needed only for `inputs` that no formula uses
    per_day : samples per day; given, the time axis is in days instead of samples
    time    : time stamps of the samples, one per row of the data the model was
              fitted to, or the name of that column in `data`; given, the time
              axis shows them (and `per_day` is not needed)
    xlim    : (lo, hi) range of the time axis of the "simulation",
              "residual_series" and "residuals" figures. With `time`, give time
              stamps, e.g. ["2021-03-01 10:00", "2021-03-01 12:00"]; a string
              without a time zone is read in the time zone of `time`. Without
              `time`, give days (with `per_day`) or sample numbers. The y-axes
              are scaled to the data inside the range.
    show_knots : mark the spline knots in the block figures
    block_xlim : (lo, hi) x-axis range of all block figures, so they can be
              compared (None: the range of each block's input)

    Returns {plot name: matplotlib figure}; block figures are keyed "block:<term>".
    """
    import matplotlib.pyplot as plt

    if isinstance(plots, str):
        plots = [plots]
    unknown = [p for p in plots if p not in PLOT_TYPES]
    if unknown:
        raise ValueError(f"unknown plot(s) {unknown}; choose from {list(PLOT_TYPES)}")

    cm = r["compiled"]
    y, sim, n_train = r["y_data"], r["sim"], r["n_train"]
    term_of = {t.name: (t, sl) for t, sl in zip(cm.terms, cm.slices)}

    def as_panels(spec):
        """"a" or ["a", ["b", "c"]] -> [["a"]] / [["a"], ["b", "c"]]: one entry
        per panel, each holding the names drawn in it."""
        spec = [spec] if isinstance(spec, str) else list(spec)
        return [[p] if isinstance(p, str) else list(p) for p in spec]

    inputs = as_panels(inputs)
    def reg_panels_of(spec):
        return [[nm] for nm in term_of] if spec == "all" else as_panels(spec)

    reginput = reg_panels_of(reginput)
    regcontribution = reg_panels_of(regcontribution)
    for arg, spec in (("reginput", reginput), ("regcontribution", regcontribution)):
        unknown = [t for panel in spec for t in panel if t not in term_of]
        if unknown:
            raise ValueError(f"unknown term(s) {unknown} in {arg}; "
                             f"the model has {list(term_of)}")
    if time is not None:
        t = _time_stamps(data[time] if isinstance(time, str) else time)
        if len(t) != len(y):
            raise ValueError(f"time has {len(t)} values, but the data has {len(y)} rows")
        xlabel = "Time"
        if xlim is not None:
            xlim = [_time_stamp(v, t) for v in xlim]
    else:
        t = np.arange(len(y)) / (per_day if per_day else 1)
        xlabel = "Time [days]" if per_day else "Sample"
    # where the training period ends, on the time axis
    t_train = t[n_train] if n_train and n_train < len(t) else None
    figures = {}

    def column(col):
        if data is not None and col in data:
            return np.asarray(data[col], float)
        if col in cm.columns:
            return cm.columns[col]
        raise KeyError(f"{col!r} is not a column of the model; pass data=...")

    def reg_column(term_name):
        """The term's column in the design matrix, before lagging and theta."""
        term, sl = term_of[term_name]
        return np.asarray(term.column(r["params"][sl]), float)

    def contribution(term_name):
        """sum_i theta_i * term(k-i): the term's part of the one-step prediction.
        theta of a term is ordered oldest lag first, as in CompiledModel._lags();
        the first `lags` samples have no full lag window and are NaN."""
        x = reg_column(term_name)
        th = np.asarray(r["theta"], float)[r["theta_of"][term_name]]
        q, out = len(th), np.full(len(x), np.nan)
        out[q:] = sum(th[i] * x[i:len(x) - q + i] for i in range(q))
        return out

    def set_time_axis(axes):
        """Readable date labels with `time`, and the range `xlim` with each
        y-axis scaled to the data inside it."""
        if time is not None:
            import matplotlib.dates as mdates
            loc = mdates.AutoDateLocator()
            for axi in axes:
                axi.xaxis.set_major_locator(loc)
                axi.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        if xlim is not None:
            axes[0].set_xlim(*xlim)
            for axi in axes:
                axi.set_xlim(*xlim)          # also for axes that don't share x
                _fit_ylim(axi)

    # ---- stacked time series: a top panel with inputs etc. below ---------
    def draw_panel(axi, series, colours):
        """series: [(ylabel, legend label, values), ...] drawn in one panel.

        All series of a panel share one y-axis, so their levels are directly
        comparable; put quantities of different units in separate panels.
        """
        for j, (nm, lab, vals) in enumerate(series):
            axi.plot(t, vals, color=colours[j % len(colours)], lw=1, label=lab)
        axi.grid(alpha=0.4)
        axi.set_ylabel(" / ".join(nm for nm, _, _ in series))
        if len(series) > 1:
            axi.legend(fontsize=8, loc="upper right", ncol=min(len(series), 3))
        if t_train is not None:
            axi.axvline(t_train, color="grey", ls=":")

    def reg_panels(axes, spec, start, values, what):
        for row, panel in enumerate(spec, start=start):
            series = []
            for term_name in panel:
                text = term_of[term_name][0].text
                label = f"{term_name}: {text}" if len(panel) > 1 else text
                series.append((term_name if len(panel) > 1
                               else f"{term_name}\n({what})",
                               label, values(term_name)))
            draw_panel(axes[row, 0], series, REG_COLOURS)

    def stacked_figure(draw_top, title):
        """A figure with the top panel drawn by draw_top(ax), and one panel per
        entry of `inputs` / `reginput` / `regcontribution` stacked under it."""
        nrows = 1 + len(inputs) + len(reginput) + len(regcontribution)
        heights = [1.8] + [1] * (nrows - 1)
        fig, axes = plt.subplots(nrows, 1, figsize=(11, 2.8 + 1.5 * (nrows - 1)),
                                 sharex=True, squeeze=False,
                                 gridspec_kw={"height_ratios": heights})
        ax = axes[0, 0]
        draw_top(ax)
        if t_train is not None:
            ax.axvline(t_train, color="grey", ls=":", label="end of training")
        ax.legend(ncol=3, fontsize=9)
        ax.grid(alpha=0.4)

        for row, panel in enumerate(inputs, start=1):
            draw_panel(axes[row, 0],
                       [(col, col, column(col)) for col in panel], INPUT_COLOURS)
        # The terms as the regression sees them: the formula evaluated at the
        # fitted nonlinear parameters, before lagging and before theta.
        reg_panels(axes, reginput, 1 + len(inputs), reg_column, "regression input")
        # The same multiplied by the fitted theta and summed over the lags: what
        # the term adds to the one-step prediction, in units of the output.
        reg_panels(axes, regcontribution, 1 + len(inputs) + len(reginput),
                   contribution, "contribution")
        axes[-1, 0].set_xlabel(xlabel)
        set_time_axis(axes[:, 0])
        fig.suptitle(title)
        fig.tight_layout()
        return fig

    # ---- measured and simulated output ----------------------------------
    if "simulation" in plots:
        def draw_simulation(ax):
            ax.plot(t, y, color="orangered", lw=1, label=f"{r['y']} measured")
            if sim is not None:
                ax.plot(t, sim, color="deepskyblue", lw=1.5, label="simulation")
            ax.set_ylabel(r["y"])

        title = f"{name}: {r['y']}, Nd = {r['Nd']}"
        if sim is not None:
            title += f", simulation RMSE {r['rmse_sim']:.3f}"
        figures["simulation"] = stacked_figure(draw_simulation, title)

    # ---- one-step residuals as a time series, with inputs below -----------
    if "residual_series" in plots:
        def draw_residuals(ax):
            ax.plot(t[cm.L:], r["resid"], color="grey", lw=0.8,
                    label="one-step residual")
            ax.axhline(0, color="black", lw=0.6)
            ax.set_ylabel(f"{r['y']} residual")

        figures["residual_series"] = stacked_figure(
            draw_residuals,
            f"{name}: one-step residuals of {r['y']}, RMSE {r['rmse']:.4f}")

    # ---- one-step residuals ---------------------------------------------
    if "residuals" in plots:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.2),
                                       gridspec_kw={"width_ratios": [2.2, 1]})
        ax1.plot(t[cm.L:], r["resid"], color="grey", lw=0.8)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel("one-step residual")
        ax1.grid(alpha=0.4)
        set_time_axis([ax1])
        ax2.acorr(r["resid"] - r["resid"].mean(), maxlags=min(40, len(r["resid"]) - 1),
                  usevlines=True, normed=True, lw=1)
        ax2.set_xlim(0, min(40, len(r["resid"]) - 1))
        ax2.set_xlabel("lag")
        ax2.set_ylabel("autocorrelation")
        ax2.grid(alpha=0.4)
        fig.suptitle(f"{name}: one-step residuals, RMSE {r['rmse']:.4f}")
        fig.tight_layout()
        figures["residuals"] = fig

    # ---- the nonlinear functions ----------------------------------------
    if "blocks" in plots:
        for term, sl in zip(cm.terms, cm.slices):
            if term.block is None:
                continue
            block, p = term.block, r["params"][sl]
            # nanmin/nanmax: with a missing sample in the input, min/max give NaN
            # and the curve disappears, leaving only the knots
            lo, hi = float(np.nanmin(block.x)), float(np.nanmax(block.x))
            if isinstance(block, BSplineBlock):
                lo, hi = block.lo, block.hi
            elif isinstance(block, PchipBlock):
                # cover all knots, also those outside the range of the data
                lo, hi = min(lo, block.lower), max(hi, block.upper)
            if block_xlim is not None and not isinstance(block, BSplineBlock):
                # the B-spline is only defined on [block.lo, block.hi]
                lo, hi = block_xlim
            xx = np.linspace(lo, hi, 300)
            fig, ax = plt.subplots(figsize=(4.5, 3.6))
            ax.plot(xx, r["functions"][term.name](xx), color="black")
            if show_knots and isinstance(block, PchipBlock):
                ax.scatter(*block.knots_and_values(p), color="firebrick", s=18, zorder=3)
            elif show_knots and isinstance(block, BSplineBlock):
                for k in block.knots(p):
                    ax.axvline(k, color="firebrick", ls=":", lw=1)
            ax.set_xlabel(f"input of {block.kind}() in term {term.name!r}")
            ax.set_ylabel(f"{block.kind}({block.name})")
            ax.set_title(term.text, fontsize=9)
            if block_xlim is not None:
                ax.set_xlim(block_xlim)
            ax.grid(alpha=0.4, ls="--")
            fig.tight_layout()
            figures[f"block:{term.name}"] = fig

    if outdir is not None:
        from pathlib import Path
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        for key, fig in figures.items():
            safe = key.replace("block:", "block_").replace(" ", "_").replace("/", "_")
            fig.savefig(outdir / f"{name}_{safe}.png", dpi=120)
    return figures


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
