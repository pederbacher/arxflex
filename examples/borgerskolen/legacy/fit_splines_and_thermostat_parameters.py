from .solar_gain import solar_gain
from .sigmoid_valve import sigmoid_valve
from .construct_design_matrix import construct_design_matrix
from arxflex.least_squares import least_squares


def fit_splines_and_thermostat_parameters(params, data, roomname, order, basissplines=None,
                                          use_solar=True):
    """params = [alpha, Toffset, bspline coefficients...]
    (only [alpha, Toffset] if use_solar=False; basissplines is then not used)"""
    alpha, Toffset = params[:2]
    solar = solar_gain(len(data), basissplines, params[2:]) if use_solar else None
    X, y = construct_design_matrix(order, data, roomname, sigmoid_valve(alpha, Toffset), solar,
                                   use_solar=use_solar)
    return least_squares(X, y)[1]
