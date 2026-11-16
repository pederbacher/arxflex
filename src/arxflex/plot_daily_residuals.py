"""One-step residuals against the time of day, for days selected by the daily
sum of an input, e.g. only clear-sky days by the solar radiation."""

import numpy as np

WHICH = ("above", "below")


def plot_daily_residuals(r, data, column, threshold, which="above", time="time",
                         name="model", ylim=None, unit="", daily_unit=""):
    """Residuals (upper) and an input (lower) against the time of day, one line
    per day, for the days whose daily sum of the input is above or below a
    threshold.

    Comparing the days above and below shows whether a pattern in the residuals
    follows the input: e.g. a morning bump in the residuals on sunny days only
    points to a missing or wrong solar gain.

    r         : the result dict from fitmodel()
    data      : the data frame the model was fitted to (needs pandas)
    column    : the input that selects the days and is drawn in the lower
                panel, e.g. "Ps"
    threshold : limit on the daily sum of `column` integrated over time in
                hours, i.e. sum(column * sampling interval [h]); for a solar
                radiation in kW/m² that is kWh/m²
    which     : "above" (daily sum > threshold), "below" (<= threshold), or
                ["above", "below"] for both side by side in one figure, with
                shared y-axes
    time      : time stamps of the rows of `data`, or the name of that column;
                days and time of day are taken in their time zone
    name      : name used in the title
    ylim      : (lo, hi) of the residual axis, to compare figures on one scale
    unit, daily_unit : units of `column` and of its daily sum, for the labels

    Returns the matplotlib figure.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    which = [which] if isinstance(which, str) else list(which)
    unknown = [w for w in which if w not in WHICH]
    if unknown:
        raise ValueError(f"unknown which {unknown}; choose from {list(WHICH)}")

    t = pd.DatetimeIndex(data[time] if isinstance(time, str) else time)
    x = np.asarray(data[column], float)
    n = len(r["y_data"])
    if len(t) != n or len(x) != n:
        raise ValueError(f"time and {column!r} must have one value per row of the "
                         f"data the model was fitted to ({n} rows)")
    resid = np.full(n, np.nan)              # one-step residuals, aligned with data
    resid[n - len(r["resid"]):] = r["resid"]  # (the first L samples have none)

    day = t.floor("D")
    hour = np.asarray(t.hour + t.minute / 60 + t.second / 3600, float)
    dt_hours = np.median(np.diff(t.asi8)) / 3.6e12          # ns -> h
    daily = pd.Series(x * dt_hours).groupby(day).sum(min_count=1)

    ncol = len(which)
    fig, axes = plt.subplots(2, ncol, figsize=(8 if ncol == 1 else 13, 6),
                             sharex=True, sharey="row", squeeze=False)
    for col, w in enumerate(which):
        keep = daily > threshold if w == "above" else daily <= threshold
        days = daily.index[keep]
        sign = ">" if w == "above" else "≤"
        what = f"daily sum of {column} {sign} {threshold}{' ' + daily_unit if daily_unit else ''}"
        print(f"{len(days)} of {len(daily)} days with {what}")
        ax_res, ax_x = axes[0, col], axes[1, col]
        for i, d in enumerate(days):
            rows = np.asarray(day == d)
            colour = f"C{i % 10}"           # the same colour for a day in both panels
            ax_res.plot(hour[rows], resid[rows], color=colour, lw=0.8, alpha=0.7)
            ax_x.plot(hour[rows], x[rows], color=colour, lw=0.8, alpha=0.7,
                      label=d.strftime("%d %b"))
        # mean residual at each time of day over the selected days
        sel = np.asarray(day.isin(days))
        mean_resid = pd.Series(resid[sel]).groupby(hour[sel]).mean()
        ax_res.plot(mean_resid.index, mean_resid.to_numpy(), color="black", lw=2,
                    label="mean")
        ax_res.axhline(0, color="grey", lw=0.6)
        ax_res.legend(fontsize=8)
        ax_res.set_title(f"{len(days)} days with {what}", fontsize=10)
        tz = f", {t.tz}" if t.tz is not None else ""
        ax_x.set_xlabel(f"Time of day [h{tz}]")
        if len(days):
            ax_x.legend(fontsize=7, ncol=4)
    axes[0, 0].set_ylabel(f"one-step residual of {r['y']}")
    axes[1, 0].set_ylabel(f"{column} [{unit}]" if unit else column)
    if ylim is not None:
        axes[0, 0].set_ylim(ylim)
    for ax in axes.flat:
        ax.grid(alpha=0.4)
        ax.set_xlim(0, 24)
    fig.suptitle(f"{name}: one-step residuals and {column} by time of day")
    fig.tight_layout()
    return fig
