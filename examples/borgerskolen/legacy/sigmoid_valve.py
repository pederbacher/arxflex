import numpy as np


def sigmoid_valve(alpha, Toffset):
    """Valve opening v(dT) = 1/(1+exp(-alpha*(dT+Toffset))), dT = Tset - Ti."""
    return lambda dT: 1.0 / (1.0 + np.exp(-alpha * (np.asarray(dT) + Toffset)))
