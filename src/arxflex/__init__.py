"""arxflex: ARX models with nonlinear input blocks, specified as formulas.

The methods are described in C. A. Thilker, P. Bacher, D. Cali and H. Madsen
(2022), Identification of non-linear autoregressive models with exogenous
inputs for room air temperature modelling, Energy and AI 9, 100165,
https://doi.org/10.1016/j.egyai.2022.100165.

Each file is named after its main function. The model is given as a dict of
formula terms, fitted with fitmodel() and simulated with simulate_model():

    from arxflex import fitmodel
    fit = fitmodel(df, "Ti", {"AR": 2, "ambient": "Ta - Ti",
                              "heating": "v(sigmoid, Tset - Ti) * (Tfor - Ti)"})

The functions take their settings as arguments. The only value kept in the
package is BAD_FIT, the objective value for invalid parameters, which
fitmodel() and compile_model() accept as bad_fit=.
"""

from .nonlinear_blocks import (Block, BlockError, BSplineBlock, SigmoidBlock,
                               PchipBlock, bspline_basis, register_block, make_block)
from .parse_formula import Term, FormulaError, parse_formula
from .least_squares import least_squares
from .compile_model import CompiledModel, compile_model, BAD_FIT
from .simulate_model import simulate_model
from .fitmodel import fitmodel, print_fit, fit_row
from .plot_fit import plot_fit
from .plot_daily_residuals import plot_daily_residuals

__version__ = "0.1.0"

__all__ = ["Block", "BlockError", "BSplineBlock", "SigmoidBlock", "PchipBlock",
           "bspline_basis", "register_block", "make_block",
           "Term", "FormulaError", "parse_formula", "least_squares",
           "CompiledModel", "compile_model", "BAD_FIT", "simulate_model",
           "fitmodel", "print_fit", "fit_row", "plot_fit",
           "plot_daily_residuals"]
