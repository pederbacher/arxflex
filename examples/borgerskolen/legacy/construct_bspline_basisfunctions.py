import numpy as np
import scipy.interpolate as si


def construct_bspline_basisfunctions(samples_per_day, bspline_order=4, knot_params=None):
    """Quadratic B-spline basis over the day for the solar gain s(h).

    samples_per_day : rows of the returned basis, one per sample of the day
                    (48 for the 30-min data of the notebook)
    bspline_order : number of knots including the end points 0 h and 24 h
                    (the notebook calls it "order"). Gives bspline_order+1
                    basis functions.
    knot_params   : None -> equidistant interior knots; otherwise the interior
                    knots are cumsum(knot_params), len = bspline_order-2
                    (the free-knots variant).
    Returns an array (samples_per_day, bspline_order+1), one row per sample of
    the day.
    """
    x = np.linspace(0, 24, 10)
    if knot_params is None:
        t = np.linspace(x[0], x[-1], bspline_order)[1:-1]
    else:
        t = np.cumsum(knot_params)
    tck = si.splrep(x, np.ones(x.shape), t=t, k=2)
    # NOTE (as in notebook): 48 points over [0, 24] have spacing 24/47 h, not exactly 30 min
    tt = np.linspace(0.0, 24, samples_per_day)
    bsplines = np.zeros((samples_per_day, bspline_order + 1))
    for i in range(bspline_order + 1):
        coeff = np.zeros(len(tck[1]))
        coeff[i] = 1.0
        bsplines[:, i] = si.splev(tt, (tck[0], coeff, tck[2]))
    return bsplines
