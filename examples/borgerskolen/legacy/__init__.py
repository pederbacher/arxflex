"""The hand-written models ported from ARXmodelling_experiment.ipynb.

Not part of the installed packages. They are kept to show that arxflex.fitmodel()
reproduces them (see ../check_fitmodel.py). Each file is named after its main
function; helpers used by one function live in that function's file.
"""

from .construct_bspline_basisfunctions import construct_bspline_basisfunctions
from .solar_gain import solar_gain
from .sigmoid_valve import sigmoid_valve
from .run_valve_spline_model import valve_knots_and_values
from .run_valve_spline_model import pchip_valve
from .construct_design_matrix import lagged
from .construct_design_matrix import construct_design_matrix
from .print_results import parameter_names
from .fit_splines_and_thermostat_parameters import fit_splines_and_thermostat_parameters
from .run_free_knots_model import fit_splines_knots_and_thermostat_parameters
from .run_valve_spline_model import fit_splines_and_thermostat_parameters_valvesplines
from .run_free_knots_model import construct_linear_inequality_constraints
from .run_valve_spline_model import construct_linear_inequality_constraints_valve
from ._finish import predict_multistep_ARX_model
from ._finish import _finish
from .run_sigmoid_model import run_sigmoid_model
from .run_free_knots_model import run_free_knots_model
from .run_valve_spline_model import run_valve_spline_model
from .fit_ARX_model_order import fit_ARX_model_order
from .print_results import print_results
from .fit_summary import fit_summary
from .plot_results import plot_results
