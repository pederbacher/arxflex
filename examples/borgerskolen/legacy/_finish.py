import numpy as np

from arxflex_borgerskolen import temperature_column


def predict_multistep_ARX_model(theta, Y0, length, Ta, Ps, Tset, Tfor, solar=None, valve=None,
                                use_solar=True):
    """Simulate Ti from the inputs only (Ta, Ps, Tset); the inputs that depend on Ti
    are recomputed from the simulated Ti.

    Generalises the notebook's last (3rd) definition to any valve function.
    Y0 = the first Nd measured values; theta in construct_design_matrix order.
    Tfor = assumed forward temperature [degC], a modelling assumption of the caller
    use_solar=False : model without solar input (theta has no Ps parameters; Ps is ignored)
    NOTE: the notebook's 1st definition took the inputs from the design matrix,
    i.e. computed with *measured* Ti, which is not a true simulation.
    """
    Nd = len(Y0)
    Ta, Tset = np.asarray(Ta, float), np.asarray(Tset, float)
    if use_solar:
        Ps = np.asarray(Ps, float)
        gain = np.ones(len(Ps)) if solar is None else np.asarray(solar)
    Y = np.zeros(length)
    Y[:Nd] = Y0
    for i in range(length - Nd):
        y_hist = Y[i:i + Nd]  # oldest first
        v = 1.0 if valve is None else valve(Tset[i:i + Nd] - y_hist)
        parts = [Ta[i:i + Nd] - y_hist]
        if use_solar:
            parts.append(gain[i:i + Nd] * Ps[i:i + Nd])
        parts.append(v * (Tfor - y_hist))
        exog = np.concatenate(parts)
        Y[i + Nd] = y_hist @ theta[:Nd][::-1] + theta[Nd:] @ exog
    return Y


def _finish(data, roomname, Nd, theta, S, valve, solar, res, *, Tfor, n_train=None,
            use_solar=True, **extra):
    """Simulate over the whole period and collect results. Tfor as above."""
    Ti = data[temperature_column(data, roomname)].to_numpy(float)
    sim = predict_multistep_ARX_model(theta, Ti[:Nd], len(data), data["Ta"], data["Ps"],
                                      data["Tset_C" + roomname], Tfor, solar=solar, valve=valve,
                                      use_solar=use_solar)
    return dict(roomname=roomname, Nd=Nd, theta=theta, S=S, res=res, valve=valve,
                solar=solar, sim=sim, Ti=Ti, Tfor=Tfor, n_train=n_train, use_solar=use_solar,
                **extra)
