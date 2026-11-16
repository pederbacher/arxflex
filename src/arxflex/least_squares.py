import numpy as np


def least_squares(X, y):
    """Linear parameters and residual sum of squares.
    NOTE: the notebook used inv(X'X)' X'y; lstsq gives the same result, more stably."""
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ theta
    return theta, float(r @ r)
