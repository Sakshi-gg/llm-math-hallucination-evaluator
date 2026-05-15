import sympy as sp


def compute_ground_truth(expr):
    """
    Compute the true simplified expression using SymPy.
    """
    try:
        symbolic = sp.sympify(expr)

        # First simplify
        simplified = sp.simplify(symbolic)

        # Force logarithmic simplification
        simplified = sp.expand_log(simplified, force=True)
        simplified = sp.logcombine(simplified, force=True)

        # Final simplify pass
        simplified = sp.simplify(simplified)

        return simplified

    except Exception:
        return None


def is_correct(predicted, ground_truth):
    """
    Check symbolic equivalence.
    """
    if predicted is None or ground_truth is None:
        return False

    try:
        return sp.simplify(predicted - ground_truth) == 0
    except Exception:
        return False


def compute_correctness(symbolic_outputs, ground_truth):
    """
    Compute correctness score across outputs.
    """
    valid_outputs = [s for s in symbolic_outputs if s is not None]

    if len(valid_outputs) == 0:
        return 0.0

    correct = sum(is_correct(s, ground_truth) for s in valid_outputs)

    return correct / len(valid_outputs)