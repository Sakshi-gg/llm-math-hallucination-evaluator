import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application
)

# Enable implicit multiplication: (x+1)(x-2)
transformations = standard_transformations + (
    implicit_multiplication_application,
)


def clean_output(output):
    """
    Basic cleaning of model output.
    """
    if output is None:
        return None

    output = output.strip()
    return output


def to_symbolic(expr_str):
    """
    Convert string to SymPy expression.
    Returns None if parsing fails.
    """
    if expr_str is None:
        return None

    try:
        expr = parse_expr(expr_str, transformations=transformations)
        return expr
    except Exception:
        return None