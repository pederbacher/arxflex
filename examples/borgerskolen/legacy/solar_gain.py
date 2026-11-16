import numpy as np


def solar_gain(n, basissplines, bsplinecoeff):
    """s(h_k) for samples k = 0..n-1.

    NOTE (as in notebook): the time of day is taken from the row number
    (k mod 48), so the first row (00:30) counts as hour 0.
    """
    daily = basissplines @ bsplinecoeff
    return daily[np.arange(n) % basissplines.shape[0]]
