"""Bind a model specification to data: design matrix, parameter packing, objective.

compile_model() turns

    {"AR": 2,
     "ambient": "Ta - Ti",
     "solar":   "s(timeofday, k=5) * Ps",
     "heating": "v(sigmoid, Tset - Ti) * (Tfor - Ti)"}

into a CompiledModel, which holds one term per entry and knows

  * where each term's nonlinear parameters sit in one global parameter vector,
    so the spline coefficients, spline knots and valve parameters of all terms
    are optimised together in a single call to the optimiser,
  * the box bounds and linear inequality constraints of that vector, with the
    constraints of each block moved into its own columns,
  * how to build the design matrix and the least squares objective.

The column order of the design matrix follows the older
construct_design_matrix(): the output lags newest first, then for each term its
lags oldest first. With "AR": 0 there are no output lags, and the model is a
plain regression on the lagged terms.

Missing values are allowed. A NaN in the output or in an input reaches the
design matrix through every lag it enters, and the rows that are not fully
finite are dropped from the least squares (see used_rows() and design_used()).
A gap of one sample therefore costs L rows, L being the largest lag in the
model. Nothing is interpolated, and no row is dropped from design(), so the
rows stay aligned with the data.
"""

import numpy as np

from .least_squares import least_squares
from .nonlinear_blocks import BlockError
from .parse_formula import parse_formula, FormulaError

AR_KEY = "AR"   # reserved key of the model dict: the number of output lags
BAD_FIT = 1e10  # objective value returned for invalid nonlinear parameters


def _is_numeric(series):
    """Can this column be used as a model variable?

    np.issubdtype() raises TypeError on a pandas extension dtype (string,
    categorical, ...) instead of returning False, so a frame holding a text
    column used to crash here.  Nullable numeric dtypes (Int64, Float64) are
    numeric and are kept.
    """
    try:
        import pandas as pd
        return bool(pd.api.types.is_numeric_dtype(series))
    except ImportError:
        pass
    try:
        return bool(np.issubdtype(series.dtype, np.number))
    except TypeError:
        return False


def _numeric(series):
    """Column as float64, with the missing values of a nullable dtype as NaN."""
    try:
        return series.to_numpy(dtype=float, na_value=np.nan)
    except TypeError:            # plain numpy-backed column: no na_value kwarg
        return series.to_numpy(float)


class CompiledModel:
    """A model specification bound to one data set. Built by compile_model()."""

    def __init__(self, y, y_name, terms, Nd, columns=None, bad_fit=BAD_FIT):
        self.y_all = np.asarray(y, float)
        self.y_name = y_name
        self.columns = {} if columns is None else columns
        self.bad_fit = float(bad_fit)
        self.terms = list(terms)
        self.Nd = int(Nd)
        self.n = len(self.y_all)
        self.L = max([self.Nd] + [t.lags for t in self.terms])
        if self.n <= self.L:
            raise ValueError(f"only {self.n} samples for {self.L} lags")

        # ---- parameter layout ------------------------------------------
        self.slices, start = [], 0
        for t in self.terms:
            self.slices.append(slice(start, start + t.nparam))
            start += t.nparam
        self.nparam = start

        # ---- what does not change during the optimisation --------------
        self.y = self.y_all[self.L:]
        # (n - L) x Nd; with Nd = 0 an empty block, so design() needs no special case
        self.ar = np.column_stack(
            [self.y_all[self.L - 1 - j:self.n - 1 - j] for j in range(self.Nd)]
        ) if self.Nd else np.empty((self.n - self.L, 0))
        self._const_lags = [None if t.block is not None else self._lags(t.const, t.lags)
                            for t in self.terms]

        # Rows that are unusable whatever the nonlinear parameters are: a
        # missing output, or a missing value in one of the fixed columns.
        ok = np.isfinite(self.y) & np.all(np.isfinite(self.ar), axis=1)
        for const in self._const_lags:
            if const is not None:
                ok &= np.all(np.isfinite(const), axis=1)
        self.base_ok = ok
        if not ok.any():
            raise ValueError(
                f"no usable samples: all {len(ok)} rows of the design matrix "
                "contain a missing value")

    # ---- helpers --------------------------------------------------------
    def _lags(self, x, q):
        """Columns x_{k-q} .. x_{k-1} (oldest lag first) for k = L .. n-1."""
        x = np.asarray(x, float)
        return np.column_stack([x[self.L - q + i:self.n - q + i] for i in range(q)])

    def dynamic_terms(self):
        """Indices of the terms that depend on the output and must be recomputed
        during simulation."""
        return [i for i, t in enumerate(self.terms) if t.uses_y]

    def blocks(self):
        """[(term, block, slice)] for the terms that have a nonlinear block."""
        return [(t, t.block, s) for t, s in zip(self.terms, self.slices)
                if t.block is not None]

    # ---- the parameter vector ------------------------------------------
    def init(self):
        if self.nparam == 0:
            return np.zeros(0)
        return np.concatenate([t.block.init() for t, _, _ in self.blocks()])

    def bounds(self):
        b = []
        for t, block, _ in self.blocks():
            b += list(block.bounds())
        return b

    def constraints(self):
        """(A, lb, ub) over the whole parameter vector, or None if there are none."""
        rows, lbs, ubs = [], [], []
        for t, block, sl in self.blocks():
            c = block.constraints()
            if c is None:
                continue
            A, lb, ub = c
            A = np.atleast_2d(A)
            full = np.zeros((A.shape[0], self.nparam))
            full[:, sl] = A
            rows.append(full)
            lbs.append(np.atleast_1d(lb))
            ubs.append(np.atleast_1d(ub))
        if not rows:
            return None
        return np.vstack(rows), np.concatenate(lbs), np.concatenate(ubs)

    def valid(self, params):
        return all(block.valid(params[sl]) for _, block, sl in self.blocks())

    def unpack(self, params):
        """{term name: that term's nonlinear parameters}."""
        params = np.asarray(params, float)
        return {t.name: params[sl] for t, sl in zip(self.terms, self.slices)
                if t.nparam}

    # ---- the model ------------------------------------------------------
    def columns(self, params):
        """One column per term, before lagging."""
        params = np.asarray(params, float)
        return [t.column(params[sl]) for t, sl in zip(self.terms, self.slices)]

    def design(self, params):
        """Design matrix X and target y."""
        params = np.asarray(params, float)
        blocks = [self.ar]
        for t, sl, const in zip(self.terms, self.slices, self._const_lags):
            blocks.append(const if const is not None else self._lags(t.column(params[sl]), t.lags))
        return np.hstack(blocks), self.y

    def used_rows(self, X=None, y=None):
        """Boolean mask of the design matrix rows the least squares can use.

        A row is used when the target and every column of that row are finite.
        With X and y omitted only the fixed part is checked (base_ok).
        """
        if X is None:
            return self.base_ok
        if y is None:
            y = self.y
        return self.base_ok & np.isfinite(y) & np.all(np.isfinite(X), axis=1)

    def design_used(self, params):
        """design(params) with the rows holding a missing value dropped.

        Returns (X, y, mask); mask is over the rows of design(), so
        X.shape[0] == mask.sum().
        """
        X, y = self.design(params)
        mask = self.used_rows(X, y)
        return X[mask], y[mask], mask

    @property
    def ntheta(self):
        """Number of regression coefficients, i.e. columns of the design matrix."""
        return self.Nd + sum(t.lags for t in self.terms)

    def objective(self, params):
        """Residual sum of squares after solving the regression coefficients by
        least squares.

        Rows with a missing value are left out; if too few remain to identify
        the regression coefficients, bad_fit is returned instead.
        """
        params = np.asarray(params, float)
        if not self.valid(params):
            return self.bad_fit
        try:
            X, y, _ = self.design_used(params)
        except BlockError:
            return self.bad_fit
        if len(y) <= self.ntheta:
            return self.bad_fit
        return least_squares(X, y)[1]

    def parameter_names(self):
        """Names of the regression coefficients, in design matrix order."""
        names = [f"{self.y_name}(k-{j + 1})" for j in range(self.Nd)]
        for t in self.terms:
            names += [f"{t.name}(k-{t.lags - i})" for i in range(t.lags)]
        return names

    def theta_slices(self):
        """{name: slice} of the regression coefficients, "AR" plus one per term."""
        out, start = {AR_KEY: slice(0, self.Nd)}, self.Nd
        for t in self.terms:
            out[t.name] = slice(start, start + t.lags)
            start += t.lags
        return out

    # ---- subsets --------------------------------------------------------
    def restrict(self, n):
        """A copy using only the first n samples (for fitting on a training period)."""
        sl = slice(0, n)
        return CompiledModel(self.y_all[sl], self.y_name,
                             [t.restrict(sl) for t in self.terms], self.Nd,
                             columns={k: v[sl] for k, v in self.columns.items()},
                             bad_fit=self.bad_fit)

    def __repr__(self):
        return (f"CompiledModel({self.y_name}, Nd={self.Nd}, n={self.n}, "
                f"terms={[t.name for t in self.terms]}, nparam={self.nparam})")


def compile_model(data, y, model, lags=None, bad_fit=BAD_FIT):
    """Parse a model specification and bind it to the data.

    data  : data frame; every variable of the formulas is a column of it, and so
            is the output variable
    y     : name of the output column
    model : {"AR": number of output lags, term name: formula, ...}. A formula
            may be a string, or a dict {"formula": ..., "lags": ...} to give
            that term a different number of lags. "AR": 0 leaves the output
            lags out of the model.
    lags  : default number of lags of the terms (default: the AR order, or 1
            when it is 0)
    bad_fit : objective value returned for invalid nonlinear parameters
    """
    if y not in data.columns:
        raise ValueError(f"output variable {y!r} is not a column of the data")
    if AR_KEY not in model:
        raise ValueError(f'the model needs an {AR_KEY!r} entry with the number of output lags')
    Nd = int(model[AR_KEY])
    if Nd < 0:
        raise ValueError(f"{AR_KEY} must be 0 or more")
    default_lags = max(Nd, 1) if lags is None else int(lags)

    columns = {c: _numeric(data[c]) for c in data.columns
               if c.isidentifier() and _is_numeric(data[c])}
    terms = []
    for name, spec in model.items():
        if name == AR_KEY:
            continue
        term_lags = default_lags
        if isinstance(spec, dict):
            spec = dict(spec)
            term_lags = int(spec.pop("lags", default_lags))
            text = spec.pop("formula")
            if spec:
                raise ValueError(f"term {name!r}: unknown options {sorted(spec)}")
        else:
            text = spec
        if term_lags < 1:
            raise ValueError(f"term {name!r}: lags must be at least 1")
        try:
            terms.append(parse_formula(text, columns, name=name, lags=term_lags, y_name=y))
        except FormulaError as e:
            raise FormulaError(f"term {name!r}: {e}")
    if not terms:
        raise ValueError("the model has no input terms")

    # Missing values are dropped row by row further down (CompiledModel), but a
    # column with nothing usable in it at all is a mistake worth naming here.
    used = {y} | {v for t in terms for v in t.variables}
    empty = [c for c in sorted(used) if not np.any(np.isfinite(columns[c]))]
    if empty:
        raise ValueError("no finite values at all in: " + ", ".join(empty))
    return CompiledModel(columns[y], y, terms, Nd, columns=columns, bad_fit=bad_fit)
