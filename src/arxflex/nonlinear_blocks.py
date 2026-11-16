"""Nonlinear building blocks ("conditional functions") used in fitmodel formulas.

A block is the one nonlinear function a model term may contain, e.g. the
thermostatic valve opening v(.) or the time-of-day solar gain s(.). A term is

    <block>(x) * <expression of data columns>

and the linear ARX parameters in front of it are found by least squares, so a
block only has to describe its own nonlinear parameters:

    nparam        number of parameters
    init()        starting values
    bounds()      box bounds, one (lo, hi) per parameter
    constraints() linear inequality constraints (A, lb, ub) in LOCAL
                  coordinates, i.e. A has nparam columns; None if there are none
    valid(p)      cheap guard; False -> the objective returns its bad_fit value
    function(p)   the block as a callable f(x), for plotting and simulation
    evaluate(p)   the column f(x_k) for all samples

fitmodel() stacks the blocks of all terms into one parameter vector and
optimises them together, so the spline coefficients, spline knots and valve
parameters of a model are always estimated jointly.

Blocks are looked up by kind and name, e.g. "v" and "sigmoid" for
v(sigmoid, Tset-Ti). Add your own with register_block().
"""

import copy

import numpy as np
import scipy.interpolate as si


class BlockError(Exception):
    """Raised by Block.evaluate() for parameters the block cannot represent."""


# =============================================================================
# Base class
# =============================================================================

class Block:
    """Common behaviour; subclasses set kind, name and the parameter layout."""

    kind = "?"          # "s", "v", ... the function name used in a formula
    name = "?"          # the variant, e.g. "sigmoid"
    nparam = 0

    def __init__(self, x):
        self.x = np.asarray(x, float)

    # ---- parameters ---------------------------------------------------
    def init(self):
        raise NotImplementedError

    def bounds(self):
        raise NotImplementedError

    def constraints(self):
        """(A, lb, ub) with A.shape == (nrows, self.nparam), or None."""
        return None

    def valid(self, p):
        return True

    def param_names(self):
        return [f"{self.kind}{i}" for i in range(self.nparam)]

    # ---- evaluation ----------------------------------------------------
    def function(self, p):
        """The block as a callable f(x)."""
        raise NotImplementedError

    def evaluate(self, p):
        """The column f(x_k), one value per sample."""
        return self.function(p)(self.x)

    def restrict(self, sl):
        """A copy evaluated on data[sl] only (used for n_train)."""
        new = copy.copy(self)
        new.x = self.x[sl]
        return new

    def rebind(self, x):
        """A copy with the same settings, evaluated at new input values.

        Used during simulation, where the input of the block is recomputed from
        the simulated output.
        """
        new = copy.copy(self)
        new.x = np.asarray(x, float)
        return new

    def describe(self, p):
        return {"kind": self.kind, "name": self.name,
                "params": dict(zip(self.param_names(), np.asarray(p, float)))}

    def __repr__(self):
        return f"{self.kind}({self.name}, nparam={self.nparam})"


# =============================================================================
# s(): B-spline in one variable, fixed or free knots
# =============================================================================

def bspline_basis(x, k, lo, hi, knot_params=None):
    """Quadratic B-spline basis evaluated at x; k+1 basis functions.

    k           : number of knots including the end points lo and hi
                  (bspline_order in the older functions)
    knot_params : None -> equidistant interior knots; otherwise the interior
                  knots are lo + cumsum(knot_params), len = k-2 (free knots).
    """
    # The grid only carries the knot vector through splrep; the basis functions
    # depend on the knots alone. 10 points as in the older functions, more when
    # there are many knots (splrep needs enough points between the knots).
    grid = np.linspace(lo, hi, max(10, 2 * k + 2))
    if knot_params is None:
        t = np.linspace(lo, hi, k)[1:-1]
    else:
        t = lo + np.cumsum(np.asarray(knot_params, float))
    try:
        tck = si.splrep(grid, np.ones_like(grid), t=t, k=2)
    except (ValueError, TypeError) as e:
        raise BlockError(f"invalid B-spline knots (k={k}, domain [{lo:.4g}, {hi:.4g}], "
                         f"interior knots {np.array2string(np.atleast_1d(t), precision=4, threshold=8)}): {e}")
    B = np.empty((len(x), k + 1))
    for i in range(k + 1):
        coeff = np.zeros(len(tck[1]))
        coeff[i] = 1.0
        B[:, i] = si.splev(x, (tck[0], coeff, tck[2]))
    return B


class BSplineBlock(Block):
    """s(x, k=5, knots="fixed"|"free"): quadratic B-spline gain in x.

    Parameters: k+1 spline coefficients, and for knots="free" the k-2 interior
    knot increments as well (the free-knots model of the notebook).

    Keyword arguments usable in a formula:
      k          number of knots including the end points      (default 5)
      knots      "fixed" or "free"                             (default "fixed")
      lo, hi     domain of the spline    (default: min and max of x in the data)
      bound      bounds (-bound, bound) on the coefficients     (default 10)
      start      starting value of every coefficient            (default 1.0)
      knot_sum   (lb, ub) on the position of the last interior knot, free knots
                 only (default: one sixteenth of the domain in from each end)
      knot_step  (lo, hi) bounds on one knot increment, free knots only
                 (default (span/16, 0.75*span))
    """

    kind = "s"
    name = "bspline"

    def __init__(self, x, k=5, knots="fixed", lo=None, hi=None, bound=10.0,
                 start=1.0, knot_sum=None, knot_step=None):
        super().__init__(x)
        if knots not in ("fixed", "free"):
            raise ValueError(f's(): knots must be "fixed" or "free", not {knots!r}')
        self.k = int(k)
        self.free = knots == "free"
        if self.k < 3:
            raise ValueError("s(): k must be at least 3")
        # nanmin/nanmax: the column may hold missing samples, whose design
        # rows are dropped later; they must not blank out the knot range.
        self.lo = float(np.nanmin(self.x)) if lo is None else float(lo)
        self.hi = float(np.nanmax(self.x)) if hi is None else float(hi)
        span = self.hi - self.lo
        self.bound = float(bound)
        self.start = float(start)
        self.n_coef = self.k + 1
        self.n_knots = self.k - 2 if self.free else 0
        self.nparam = self.n_coef + self.n_knots
        self.knot_sum = (span / 16, span - span / 16) if knot_sum is None else tuple(knot_sum)
        self.knot_step = (span / 16, 0.75 * span) if knot_step is None else tuple(knot_step)
        # Fixed knots: the basis does not depend on the parameters, build it once
        self.B = None if self.free else bspline_basis(self.x, self.k, self.lo, self.hi)

    def init(self):
        coef = np.full(self.n_coef, self.start)
        if not self.free:
            return coef
        knots0 = np.diff(np.linspace(self.lo, self.hi, self.k)[:-1])
        return np.concatenate((coef, knots0))

    def bounds(self):
        b = [(-self.bound, self.bound)] * self.n_coef
        return b + [self.knot_step] * self.n_knots

    def constraints(self):
        if not self.free:
            return None
        # The last interior knot, lo + sum(increments), must stay inside the domain
        A = np.concatenate((np.zeros((1, self.n_coef)), np.ones((1, self.n_knots))), axis=1)
        return A, np.array([self.knot_sum[0]]), np.array([self.knot_sum[1]])

    def valid(self, p):
        if not self.free:
            return True
        inc = p[self.n_coef:]
        return bool(np.all(inc > 0) and self.lo + inc.sum() < self.hi)

    def param_names(self):
        names = [f"c{i}" for i in range(self.n_coef)]
        return names + [f"dknot{i + 1}" for i in range(self.n_knots)]

    def _basis(self, p):
        if not self.free:
            return self.B
        return bspline_basis(self.x, self.k, self.lo, self.hi, p[self.n_coef:])

    def evaluate(self, p):
        return self._basis(p) @ p[:self.n_coef]

    def function(self, p):
        knot_params = p[self.n_coef:] if self.free else None
        coef = p[:self.n_coef]

        def f(x):
            x = np.atleast_1d(np.asarray(x, float))
            return bspline_basis(x, self.k, self.lo, self.hi, knot_params) @ coef

        return f

    def knots(self, p):
        """Interior knot positions for the fitted parameters."""
        if not self.free:
            return np.linspace(self.lo, self.hi, self.k)[1:-1]
        return self.lo + np.cumsum(p[self.n_coef:])

    def describe(self, p):
        d = super().describe(p)
        d.update(k=self.k, knots="free" if self.free else "fixed",
                 domain=(self.lo, self.hi), interior_knots=self.knots(p),
                 coef=np.asarray(p[:self.n_coef], float))
        return d

    def restrict(self, sl):
        new = super().restrict(sl)
        if self.B is not None:
            new.B = self.B[sl]
        return new

    def rebind(self, x):
        new = super().rebind(x)
        if not self.free:
            new.B = bspline_basis(new.x, self.k, self.lo, self.hi)
        return new


# =============================================================================
# v(sigmoid, .): logistic function
# =============================================================================

class SigmoidBlock(Block):
    """v(sigmoid, x): 1/(1+exp(-alpha*(x+offset))), parameters alpha and offset.

    Keyword arguments usable in a formula:
      bound         bounds (-bound, bound) on the offset      (default 10)
      alpha_bounds  bounds on alpha                           (default (0.2, 5))
      alpha, offset starting values                           (default 1, 0)
    """

    kind = "v"
    name = "sigmoid"
    nparam = 2

    def __init__(self, x, bound=10.0, alpha_bounds=(0.2, 5.0), alpha=1.0, offset=0.0):
        super().__init__(x)
        self.bound = float(bound)
        self.alpha_bounds = tuple(alpha_bounds)
        self.start = (float(alpha), float(offset))

    def init(self):
        return np.array(self.start, float)

    def bounds(self):
        return [self.alpha_bounds, (-self.bound, self.bound)]

    def param_names(self):
        return ["alpha", "offset"]

    def function(self, p):
        alpha, offset = p
        return lambda x: 1.0 / (1.0 + np.exp(-alpha * (np.asarray(x, float) + offset)))


# =============================================================================
# v(pchip, .): monotone spline with estimated knots and values
# =============================================================================

class PchipBlock(Block):
    """v(pchip, x, n=5): monotone PCHIP spline, f(lower)=0 and f(upper)=1.

    Parameters: n-2 knot increments and n-2 function values; the values are
    constrained to be non-decreasing, so f is monotone. This is the valve
    function of run_valve_spline_model.

    Keyword arguments usable in a formula:
      n             number of knots including the end points   (default 5)
      lower, upper  the end points                             (default -10, 10)
      step          (lo, hi) bounds on one knot increment      (default (0.05, span-n))
      knot_sum      (lb, ub) on the last interior knot, measured from lower
                    (default (n, span-1))
    """

    kind = "v"
    name = "pchip"

    def __init__(self, x, n=5, lower=-10.0, upper=10.0, step=None, knot_sum=None):
        super().__init__(x)
        self.n = int(n)
        if self.n < 3:
            raise ValueError("v(pchip): n must be at least 3")
        self.lower, self.upper = float(lower), float(upper)
        span = self.upper - self.lower
        self.span = span
        self.nv = self.n - 2
        self.nparam = 2 * self.nv
        self.step = (0.05, span - self.n) if step is None else tuple(step)
        self.knot_sum = (float(self.n), span - 1.0) if knot_sum is None else tuple(knot_sum)

    # ---- knots and values ---------------------------------------------
    def knots_and_values(self, p):
        inc, values = p[:self.nv], p[self.nv:]
        knots = np.concatenate(([self.lower], self.lower + np.cumsum(inc), [self.upper]))
        values = np.concatenate(([0.0], values, [1.0]))
        return knots, values

    def init(self):
        inc0 = np.diff(np.linspace(self.lower, self.upper, self.n)[:-1])
        values0 = np.array([0, 0.1, 0.9]) if self.n == 5 else np.linspace(0.1, 0.9, self.nv)
        return np.concatenate((inc0, values0))

    def bounds(self):
        return [self.step] * self.nv + [(0.0, 1.0)] * self.nv

    def constraints(self):
        # row 0: the last interior knot stays below the upper end point
        row0 = np.concatenate((np.ones(self.nv), np.zeros(self.nv)))
        # rows 1..: -1 <= f_i - f_{i+1} <= 0, i.e. the values are non-decreasing
        D = np.eye(self.nv - 1, self.nv) - np.eye(self.nv - 1, self.nv, k=1)
        rows = np.hstack((np.zeros((self.nv - 1, self.nv)), D))
        A = np.vstack((row0, rows))
        lb = np.concatenate(([self.knot_sum[0]], -np.ones(self.nv - 1)))
        ub = np.concatenate(([self.knot_sum[1]], np.zeros(self.nv - 1)))
        return A, lb, ub

    def valid(self, p):
        knots, _ = self.knots_and_values(p)
        return bool(np.all(np.diff(knots) > 0))

    def param_names(self):
        return ([f"dx{i + 1}" for i in range(self.nv)]
                + [f"f{i + 1}" for i in range(self.nv)])

    def function(self, p):
        knots, values = self.knots_and_values(p)
        if np.any(np.diff(knots) <= 0):
            raise BlockError(f"v(pchip): knots are not increasing: {knots}")
        return si.PchipInterpolator(knots, values)

    def describe(self, p):
        d = super().describe(p)
        knots, values = self.knots_and_values(p)
        d.update(n=self.n, knots=knots, values=values)
        return d


# =============================================================================
# Registry
# =============================================================================

BLOCKS = {
    "s": {"bspline": BSplineBlock},
    "v": {"sigmoid": SigmoidBlock, "pchip": PchipBlock},
}

DEFAULT_BLOCK = {"s": "bspline", "v": "sigmoid"}


def register_block(kind, name, cls, default=False):
    """Make a Block subclass usable in formulas as kind(name, x, ...)."""
    BLOCKS.setdefault(kind, {})[name] = cls
    if default or kind not in DEFAULT_BLOCK:
        DEFAULT_BLOCK[kind] = name


def make_block(kind, name, x, kwargs):
    """Build the block used by kind(name, x, **kwargs) in a formula."""
    if kind not in BLOCKS:
        raise ValueError(f"unknown function {kind!r} in formula; "
                         f"known: {', '.join(sorted(BLOCKS))}")
    name = DEFAULT_BLOCK[kind] if name is None else name
    if name not in BLOCKS[kind]:
        raise ValueError(f"unknown {kind}() variant {name!r}; "
                         f"known: {', '.join(sorted(BLOCKS[kind]))}")
    return BLOCKS[kind][name](x, **kwargs)
