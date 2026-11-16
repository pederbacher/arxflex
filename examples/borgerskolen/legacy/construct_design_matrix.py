import numpy as np

from arxflex_borgerskolen import temperature_column


def lagged(x, order):
    """Columns x_{k-order}, ..., x_{k-1} for k = order..n-1 (oldest lag first, as in notebook)."""
    x = np.asarray(x, float)
    n = len(x)
    return np.column_stack([x[i:n - order + i] for i in range(order)])


def construct_design_matrix(order, data, roomname, valve=None, solar=None, use_solar=True):
    """Design matrix X and target y.

    Column order (as in the notebook):
      Ti_{k-1} .. Ti_{k-order}
      (Ta-Ti)_{k-order} .. (Ta-Ti)_{k-1}
      s*Ps_{k-order} .. s*Ps_{k-1}                 (left out if use_solar=False)
      v*(Tfor-Ti)_{k-order} .. v*(Tfor-Ti)_{k-1}

    valve : callable v(Tset - Ti), or None for a plain linear ARX model
            (then neither the valve function nor the solar gain is used)
    solar : array s(h_k) for all samples, or None for s = 1
    use_solar : False -> no solar radiation input (the Ps columns are left out)
    """
    Ti = data[temperature_column(data, roomname)].to_numpy(float)
    n = len(Ti)
    Ta_Ti = data["Ta_minus_Ti_C" + roomname].to_numpy(float)
    Ps = data["Ps"].to_numpy(float)
    Tfor_Ti = data["Tfor_minus_Ti_C" + roomname].to_numpy(float)

    ar = np.column_stack([Ti[order - 1 - j:n - 1 - j] for j in range(order)])
    if valve is None:
        exog = [Ta_Ti, Ps, Tfor_Ti]
    else:
        Tset_Ti = data["Tset_minus_Ti_C" + roomname].to_numpy(float)
        exog = [Ta_Ti,
                Ps * solar if solar is not None else Ps,
                valve(Tset_Ti) * Tfor_Ti]
    if not use_solar:
        del exog[1]
    X = np.hstack([ar] + [lagged(e, order) for e in exog])
    return X, Ti[order:]
