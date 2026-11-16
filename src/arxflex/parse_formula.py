"""Turn a model formula string into a Term.

A formula is ordinary Python arithmetic over the columns of the data frame
given to fitmodel(), optionally containing one nonlinear function:

    "Ta - Ti"
    "s(timeofday, k=5) * Ps"
    "v(sigmoid, Tset - Ti) * (Tfor - Ti)"
    "v(pchip, Tset_minus_Ti, n=5) * Tfor_minus_Ti"

Everything that is not a nonlinear function is evaluated once from the data
and kept as a fixed column; the nonlinear function is re-evaluated for every
set of parameters the optimiser tries. A term may contain at most one
nonlinear function (one s(...) or one v(...)), and it may only be multiplied
or divided by data expressions, never added to them - a sum is two terms with
a linear parameter each, so write it as two entries of the model dict.

The formula is parsed with Python's own ast module, so parentheses, unary
minus, keyword arguments and operator precedence all behave as in Python.
Names are looked up in the data frame first, then among a few numpy functions
(log, exp, sqrt, abs, sin, cos, minimum, maximum, clip, where).
"""

import ast
import copy

import numpy as np

from .nonlinear_blocks import make_block, BLOCKS

# Functions allowed in a formula, applied to data expressions
FUNCTIONS = {
    "log": np.log, "log10": np.log10, "exp": np.exp, "sqrt": np.sqrt,
    "abs": np.abs, "sin": np.sin, "cos": np.cos, "tanh": np.tanh,
    "minimum": np.minimum, "maximum": np.maximum, "clip": np.clip,
    "where": np.where,
}


class FormulaError(Exception):
    """A formula that cannot be parsed or does not fit the model structure."""


class Term:
    """One term of the model: block(x) * const, or just const.

    name   label used in tables and plots
    text   the formula it was parsed from
    block  the nonlinear block, or None for a linear term
    const  the fixed part as a column, a scalar, or None for 1
    lags   number of lags of this term in the design matrix
    uses_y True if the formula refers to the output variable (then the term
           has to be recomputed during simulation)
    """

    def __init__(self, name, text, block=None, const=None, lags=1, uses_y=False,
                 variables=(), tree=None):
        self.name = name
        self.text = text
        self.block = block
        self.const = const
        self.lags = lags
        self.uses_y = uses_y
        self.variables = tuple(variables)
        self.tree = tree          # the parsed formula, for bind()

    @property
    def nparam(self):
        return 0 if self.block is None else self.block.nparam

    def column(self, params=None):
        """The term evaluated for the given nonlinear parameters, one value per sample."""
        if self.block is None:
            return self.const
        col = self.block.evaluate(params)
        return col if self.const is None else col * self.const

    def bind(self, columns):
        """The same term evaluated on other values of its variables.

        The formula is not parsed again and the block keeps its settings and
        its fitted parameters; only the input values change. Used by the
        simulation, which recomputes the terms from the simulated output.
        """
        if self.tree is None:
            raise FormulaError(f"term {self.name!r} was built without a formula tree")
        return _build(self.tree, columns, self.name, self.text, self.lags,
                      uses_y=self.uses_y, block=self.block)

    def restrict(self, sl):
        """A copy evaluated on data[sl] only."""
        new = copy.copy(self)
        if self.block is not None:
            new.block = self.block.restrict(sl)
        if self.const is not None and np.ndim(self.const) > 0:
            new.const = self.const[sl]
        return new

    def __repr__(self):
        return f"Term({self.name!r}, {self.text!r}, lags={self.lags}, block={self.block})"


# =============================================================================
# Parsing
# =============================================================================

class _Blocked:
    """Intermediate value: a nonlinear block multiplied by a data expression."""

    def __init__(self, block, scale=None):
        self.block = block
        self.scale = scale

    def scaled(self, factor):
        return _Blocked(self.block, factor if self.scale is None else self.scale * factor)


def parse_formula(text, columns, name=None, lags=1, y_name=None):
    """Parse one formula into a Term.

    text    : the formula, e.g. "v(sigmoid, Tset-Ti) * (Tfor-Ti)"
    columns : {column name: values} the formula may use
    name    : label for the term (default: the formula itself)
    lags    : number of lags of this term in the design matrix
    y_name  : name of the output variable, to flag terms that depend on it
    """
    try:
        tree = ast.parse(text.strip(), mode="eval").body
    except SyntaxError as e:
        raise FormulaError(f"cannot parse formula {text!r}: {e}")
    return _build(tree, columns, name if name is not None else text, text, lags,
                  y_name=y_name)


def _build(tree, columns, name, text, lags, y_name=None, uses_y=False, block=None):
    """Evaluate a parsed formula against columns; block: reuse this block."""
    used = set()
    factory = (lambda kind, variant, x, kwargs: block.rebind(x)) if block is not None else make_block
    value = _eval(tree, columns, used, text, factory)
    if isinstance(value, _Blocked):
        new_block, const = value.block, value.scale
    else:
        new_block, const = None, value
        if np.ndim(const) == 0:
            raise FormulaError(f"formula {text!r} is a constant, not a column")
    return Term(name, text, block=new_block, const=const, lags=lags,
                uses_y=(y_name in used) if y_name is not None else uses_y,
                variables=sorted(used), tree=tree)


def _eval(node, columns, used, text, factory):
    """Evaluate one node of the formula tree to an array or a _Blocked."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        raise FormulaError(f"{text!r}: unsupported constant {node.value!r}")

    if isinstance(node, ast.Name):
        if node.id in columns:
            used.add(node.id)
            return columns[node.id]
        raise FormulaError(
            f"{text!r}: {node.id!r} is not a column of the data "
            f"(columns: {', '.join(sorted(columns))})")

    if isinstance(node, ast.UnaryOp):
        val = _eval(node.operand, columns, used, text, factory)
        if isinstance(node.op, ast.UAdd):
            return val
        if isinstance(node.op, ast.USub):
            return val.scaled(-1.0) if isinstance(val, _Blocked) else -val
        raise FormulaError(f"{text!r}: unsupported unary operator")

    if isinstance(node, ast.BinOp):
        left = _eval(node.left, columns, used, text, factory)
        right = _eval(node.right, columns, used, text, factory)
        lb, rb = isinstance(left, _Blocked), isinstance(right, _Blocked)
        if isinstance(node.op, ast.Mult):
            if lb and rb:
                raise FormulaError(
                    f"{text!r}: a term may contain only one nonlinear function; "
                    f"split it into two entries of the model dict")
            if lb:
                return left.scaled(right)
            if rb:
                return right.scaled(left)
            return left * right
        if isinstance(node.op, ast.Div):
            if rb:
                raise FormulaError(f"{text!r}: cannot divide by a nonlinear function")
            return left.scaled(1.0 / right) if lb else left / right
        if isinstance(node.op, (ast.Add, ast.Sub)):
            if lb or rb:
                raise FormulaError(
                    f"{text!r}: a nonlinear function cannot be added or subtracted; "
                    f"each term of a sum needs its own entry in the model dict")
            return left + right if isinstance(node.op, ast.Add) else left - right
        if isinstance(node.op, ast.Pow):
            if lb or rb:
                raise FormulaError(f"{text!r}: cannot raise a nonlinear function to a power")
            return left ** right
        raise FormulaError(f"{text!r}: unsupported operator {type(node.op).__name__}")

    if isinstance(node, ast.Call):
        return _eval_call(node, columns, used, text, factory)

    raise FormulaError(f"{text!r}: unsupported expression {type(node).__name__}")


def _eval_call(node, columns, used, text, factory):
    if not isinstance(node.func, ast.Name):
        raise FormulaError(f"{text!r}: only plain function names can be called")
    fname = node.func.id

    if fname in BLOCKS:
        return _Blocked(_make_block_from_call(fname, node, columns, used, text, factory))

    if fname in FUNCTIONS:
        args = [_eval(a, columns, used, text, factory) for a in node.args]
        if any(isinstance(a, _Blocked) for a in args):
            raise FormulaError(f"{text!r}: {fname}() cannot be applied to a nonlinear function")
        kwargs = {kw.arg: _literal(kw.value, text) for kw in node.keywords}
        return FUNCTIONS[fname](*args, **kwargs)

    raise FormulaError(f"{text!r}: unknown function {fname!r}; known: "
                       f"{', '.join(sorted(set(BLOCKS) | set(FUNCTIONS)))}")


def _make_block_from_call(kind, node, columns, used, text, factory):
    """Build the block for s(...) / v(...): optional variant name, then x, then options."""
    args = list(node.args)
    if not args:
        raise FormulaError(f"{text!r}: {kind}() needs an input variable")
    variant = None
    # A leading bare name selects the variant: either a registered one that is
    # not also a column, or any name when a second argument follows it
    first = args[0]
    if isinstance(first, ast.Name) and (
            (first.id in BLOCKS[kind] and first.id not in columns) or len(args) > 1):
        if first.id not in BLOCKS[kind]:
            raise FormulaError(
                f"{text!r}: unknown {kind}() variant {first.id!r}; known: "
                f"{', '.join(sorted(BLOCKS[kind]))}")
        variant = first.id
        args = args[1:]
    if len(args) != 1:
        raise FormulaError(f"{text!r}: {kind}() takes one input variable, got {len(args)}")
    x = _eval(args[0], columns, used, text, factory)
    if isinstance(x, _Blocked):
        raise FormulaError(f"{text!r}: the input of {kind}() cannot contain another "
                           f"nonlinear function")
    if np.ndim(x) == 0:
        raise FormulaError(f"{text!r}: the input of {kind}() must depend on the data")
    kwargs = {kw.arg: _literal(kw.value, text) for kw in node.keywords}
    return factory(kind, variant, x, kwargs)


def _literal(node, text):
    """A keyword argument value: a number, string, tuple or list."""
    try:
        return ast.literal_eval(node)
    except ValueError:
        raise FormulaError(f"{text!r}: keyword arguments must be constants")
