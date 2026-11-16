import numpy as np


def input_names(use_solar=True):
    """Names of the input blocks in the design matrix (after the Ti lags)."""
    return ["Ta-Ti", "s*Ps", "v*(Tfor-Ti)"] if use_solar else ["Ta-Ti", "v*(Tfor-Ti)"]


def parameter_names(order, use_solar=True):
    """Names of the linear parameters, in design matrix order."""
    names = [f"Ti(k-{j + 1})" for j in range(order)]
    for e in input_names(use_solar):
        names += [f"{e}(k-{order - i})" for i in range(order)]
    return names


def print_results(r):
    Nd, theta = r["Nd"], r["theta"]
    use_solar = r.get("use_solar", True)
    print(f"\nRoom C{r['roomname']}, Nd = {Nd}" + ("" if use_solar else ", no solar input"))
    print(f"optimiser: success={r['res'].success}, S={r['res'].fun:.4f}, message: {r['res'].message}")
    print(f"nonlinear parameters: {np.array2string(r['res'].x, precision=4)}")
    print(f"\n{'parameter':<22}{'estimate':>12}")
    for name, value in zip(parameter_names(Nd, use_solar), theta):
        print(f"{name:<22}{value:>12.5g}")
    # Sums over the lags (as printed in notebook cell 34)
    sums = [f"Ti {theta[:Nd].sum():.4f}"]
    sums += [f"{name} {theta[(j + 1) * Nd:(j + 2) * Nd].sum():.4g}"
             for j, name in enumerate(input_names(use_solar))]
    print("\nsum over lags:  " + "  ".join(sums))
    err = r["sim"] - r["Ti"]
    if r["n_train"]:
        n = r["n_train"]
        print(f"simulation RMSE: training {np.sqrt(np.mean(err[:n] ** 2)):.3f} degC, "
              f"after training period {np.sqrt(np.mean(err[n:] ** 2)):.3f} degC")
    else:
        print(f"simulation RMSE: {np.sqrt(np.mean(err ** 2)):.3f} degC")
