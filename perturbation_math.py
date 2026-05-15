import sympy as sp
import random


def generate_equivalents(expr_str, num_perturbations=5):
    """
    Generate diverse mathematically equivalent forms
    of a given expression.

    num_perturbations controls how many forms are returned.
    """

    perturbations = set()

    try:
        expr = sp.sympify(expr_str)
    except Exception:
        return [expr_str]

    # Always include original
    perturbations.add(str(expr))

    # Expanded
    try:
        perturbations.add(str(sp.expand(expr)))
    except Exception:
        pass

    # Factored
    try:
        perturbations.add(str(sp.factor(expr)))
    except Exception:
        pass

    # Simplified
    try:
        perturbations.add(str(sp.simplify(expr)))
    except Exception:
        pass

    # Reordered multiplication
    if isinstance(expr, sp.Mul):
        args = list(expr.args)
        random.shuffle(args)
        perturbations.add(str(sp.Mul(*args)))

    # Reordered addition
    if isinstance(expr, sp.Add):
        args = list(expr.args)
        random.shuffle(args)
        perturbations.add(str(sp.Add(*args)))

    # Extra parentheses
    perturbations.add(f"({str(expr)})")

    perturbations = list(perturbations)

    # Limit to requested number
    if len(perturbations) > num_perturbations:
        perturbations = random.sample(perturbations, num_perturbations)

    return perturbations