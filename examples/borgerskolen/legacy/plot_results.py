import numpy as np


PLOT_TYPES = ("simulation", "valve", "solar")


def plot_results(data, r, outdir, model, samples_per_day, plots=PLOT_TYPES):
    """Make and save figures of a fitted model.

    plots : list of the figures to make, any of
        "simulation"  measured vs. simulated temperature, with Ta and Ps
        "valve"       the valve function v(Tset - Ti)
        "solar"       the daily solar gain s(h) (skipped for models without solar input)
      e.g. plots=["simulation"]. Default: all.

    The figures are saved in outdir as C<room>_<model>_<plot>.png.
    Returns a dict {plot name: matplotlib figure} of the figures made.
    """
    import matplotlib.pyplot as plt

    if isinstance(plots, str):
        plots = [plots]
    unknown = [p for p in plots if p not in PLOT_TYPES]
    if unknown:
        raise ValueError(f"unknown plot(s) {unknown}; choose from {list(PLOT_TYPES)}")

    room = r["roomname"]
    use_solar = r.get("use_solar", True)
    figures = {}

    # Simulation (notebook cell 94)
    if "simulation" in plots:
        days = np.arange(len(data)) / samples_per_day
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5), sharex=True,
                                       gridspec_kw={"height_ratios": [1.8, 1]})
        ax1.plot(days, r["Ti"], color="orangered", lw=1, label="Temperature measurement")
        ax1.plot(days, r["sim"], color="deepskyblue", lw=1.5, label="Prediction")
        ax1.plot(days, data["Tset_C" + room], color="black", lw=0.8, label="Set-point")
        if r["n_train"]:
            ax1.axvline(r["n_train"] / samples_per_day, color="grey", ls=":", label="end of training")
        ax1.set_ylabel("Temperature [°C]")
        ax1.legend(ncol=4, fontsize=9)
        ax1.grid(alpha=0.4)
        ax2.plot(days, data["Ta"], color="steelblue")
        ax2.set_ylabel("Ta [°C]")
        ax2.set_xlabel("Time [days]")
        ax2.grid(alpha=0.4)
        ax22 = ax2.twinx()
        ax22.plot(days, data["Ps"], color="goldenrod", ls="--")
        ax22.set_ylabel("Ps [kW/m²]", color="goldenrod")
        fig.suptitle(f"Room C{room}, {model} model, Nd = {r['Nd']}" + ("" if use_solar else ", no solar input"))
        fig.tight_layout()
        fig.savefig(outdir / f"C{room}_{model}_simulation.png", dpi=120)
        figures["simulation"] = fig

    # Valve function (notebook cell 87)
    if "valve" in plots:
        fig, ax = plt.subplots(figsize=(4, 4))
        dT = np.linspace(-10, 10, 200)
        ax.plot(dT, r["valve"](dT))
        if "valve_knots" in r:
            ax.scatter(*r["valve_knots"], color="black", s=15)
        ax.set_xlabel("Tset - Ti [°C]")  # NOTE: notebook label said Ti - Tset
        ax.set_ylabel("Valve state")
        ax.grid(alpha=0.4, ls="--")
        fig.tight_layout()
        fig.savefig(outdir / f"C{room}_{model}_valve.png", dpi=120)
        figures["valve"] = fig

    # Solar gain over the day (notebook cell 90); not for models without solar input
    if "solar" in plots and use_solar:
        avg_solar = np.array([data["Ps"].iloc[i::samples_per_day].mean()
                              for i in range(samples_per_day)])
        profile = r["basis"] @ r["bspline_coef"]
        fig, ax = plt.subplots(figsize=(5, 4))
        hours = np.linspace(0, 24, samples_per_day)
        ax.plot(hours, profile, color="black")
        ax.set_xlabel("Hour of the day")
        ax.set_ylabel("B-spline representation s(h)")
        ax.set_xticks(range(0, 25, 4))
        ax.grid(alpha=0.4)
        ax2 = ax.twinx()
        ax2.plot(hours, profile * avg_solar, color="goldenrod")
        ax2.set_ylabel("s(h) × average solar radiation", color="goldenrod")
        fig.tight_layout()
        fig.savefig(outdir / f"C{room}_{model}_solar.png", dpi=120)
        figures["solar"] = fig

    return figures
