"""The nonlinear functions of a fit, one figure each."""

import ast

import numpy as np

from .nonlinear_blocks import BSplineBlock, PchipBlock
from .plot_helpers import _time_stamps


def plot_blocks(r, show_knots=True, xlim=None, max_day=(), per_day=None,
                time=None, data=None):
    """One figure per nonlinear function (s() or v()) of the model, drawn
    against its input at the fitted parameters.

    r          : the result dict from fitmodel()
    show_knots : mark the spline knots
    xlim       : (lo, hi) x-axis range of all figures, so they can be compared
                 (None: the range of each block's input). A B-spline is only
                 defined on its domain and is always drawn there.
    max_day    : term names, e.g. ["solar"], whose figure gets a lower panel
                 with what the term adds to the output on the day with the
                 largest sum of the data its function multiplies; for
                 "s(timeofday) * Ps" that is the sunniest day. Drawn per sample
                 of that day against the function's input, as
                 sum(theta) * f(x) * Ps, in the units of the output. The
                 upper panel also shows Ps on that day, on a right-hand axis.
                 Terms
                 the model doesn't have are skipped, so one list can be used
                 for several models.
    per_day, time, data : how samples are grouped into days for `max_day`:
                 calendar days of the time stamps `time` (or of the column
                 `time` of `data`), or else blocks of `per_day` samples
                 counted from the first sample

    Returns {term name: matplotlib figure}.
    """
    import matplotlib.pyplot as plt

    cm = r["compiled"]
    max_day = [max_day] if isinstance(max_day, str) else list(max_day)
    max_day = [nm for nm in max_day
               if any(t.name == nm and t.block is not None for t in cm.terms)]
    days = _days(len(r["y_data"]), per_day, time, data) if max_day else None

    figures = {}
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
        if xlim is not None and not isinstance(block, BSplineBlock):
            lo, hi = xlim
        xx = np.linspace(lo, hi, 300)

        if term.name in max_day:
            fig, (ax, ax2) = plt.subplots(2, 1, figsize=(4.5, 5.6), sharex=True)
            axes = (ax, ax2)
        else:
            fig, ax = plt.subplots(figsize=(4.5, 3.6))
            axes = (ax,)
        ax.plot(xx, r["functions"][term.name](xx), color="black")
        if show_knots and isinstance(block, PchipBlock):
            ax.scatter(*block.knots_and_values(p), color="firebrick", s=18, zorder=3)
        elif show_knots and isinstance(block, BSplineBlock):
            for k in block.knots(p):
                for a in axes:
                    a.axvline(k, color="firebrick", ls=":", lw=1)
        ax.set_ylabel(f"{block.kind}({block.name})")
        ax.set_title(term.text, fontsize=9)
        if term.name in max_day:
            _draw_max_day(ax, ax2, r, term, block, p, days)
        axes[-1].set_xlabel(f"input of {block.kind}() in term {term.name!r}")
        if xlim is not None:
            ax.set_xlim(xlim)
        for a in axes:
            a.grid(alpha=0.4, ls="--")
        fig.tight_layout()
        figures[term.name] = fig
    return figures


def _days(n, per_day, time, data):
    """(day of each sample, label of a day): calendar days of the time
    stamps, or blocks of per_day samples."""
    if time is not None:
        t = _time_stamps(data[time] if isinstance(time, str) else time)
        if len(t) != n:
            raise ValueError(f"time has {len(t)} values, but the data has {n} rows")
        day = np.asarray(t.date) if hasattr(t, "date") else t.astype("datetime64[D]")
        return day, str
    if per_day:
        return np.arange(n) // int(round(per_day)), lambda d: f"day {d + 1}"
    raise ValueError("max_day needs per_day or time to find the days")


def _draw_max_day(ax_top, ax, r, term, block, p, days):
    """On the day with the largest daily sum of the multiplier: the multiplier
    on a right-hand axis of ax_top, and sum(theta) * f(x) * multiplier in ax,
    per sample against x."""
    if term.const is None or np.ndim(term.const) == 0:
        raise ValueError(f"max_day: the function in term {term.name!r} is not "
                         "multiplied by a data column")
    day, label = days
    mult = np.asarray(term.const, float)
    keys = np.unique(day)
    sums = [np.nansum(mult[day == k]) for k in keys]
    best = keys[int(np.argmax(sums))]
    on = day == best
    theta = float(np.sum(r["theta"][r["theta_of"][term.name]]))
    x = np.asarray(block.x, float)[on]
    y = theta * np.asarray(block.evaluate(p), float)[on] * mult[on]
    order = np.argsort(x)
    name = _multiplier_text(term.text)

    right = ax_top.twinx()
    right.plot(x[order], mult[on][order], color="goldenrod", lw=1.2)
    right.set_ylabel(f"{name} on {label(best)}", color="goldenrod")
    right.tick_params(axis="y", colors="goldenrod")

    ax.plot(x[order], y[order], color="seagreen", marker="o", ms=2.5, lw=1)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel(f"{r['y']} contribution")
    ax.set_title(f"× {name} × Σθ on {label(best)}, the day with the largest "
                 f"sum of {name}", fontsize=9)


def _multiplier_text(text):
    """The data part of "f(...) * data" as written in the formula, e.g. "Ps"
    in "s(timeofday, k=5) * Ps"; "data" when the formula has another form."""
    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError:
        return "data"

    def has_block(n):
        return any(isinstance(c, ast.Call) and getattr(c.func, "id", None) in ("s", "v")
                   for c in ast.walk(n))

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        if has_block(node.left) and not has_block(node.right):
            return ast.unparse(node.right)
        if has_block(node.right) and not has_block(node.left):
            return ast.unparse(node.left)
    return "data"
