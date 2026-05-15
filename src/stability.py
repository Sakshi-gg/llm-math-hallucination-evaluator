import sympy as sp
from collections import Counter
import numpy as np


# =====================================
# EXPRESSION CONSISTENCY SCORE (ECS)
# =====================================

def compute_stability(symbolic_outputs):
    """
    Expression Consistency Score (ECS)

    ECS = (frequency of most common canonical form)
          /
          (total valid outputs)
    """

    valid_outputs = [expr for expr in symbolic_outputs if expr is not None]

    if len(valid_outputs) <= 1:
        return 1.0

    canonical_forms = []

    for expr in valid_outputs:
        try:
            expanded = sp.expand(expr)
            canonical_forms.append(str(expanded))
        except Exception:
            canonical_forms.append(str(expr))

    counts = Counter(canonical_forms)
    most_common_count = counts.most_common(1)[0][1]

    return most_common_count / len(canonical_forms)


# =====================================
# ROBUSTNESS SCORE
# =====================================

def compute_robustness_score(ecs, correctness, hallucination_rate):
    """
    Combined robustness metric.

    Robustness =
        0.4 * ECS
      + 0.4 * Correctness
      + 0.2 * (1 - Hallucination Rate)
    """

    robustness = (
        0.4 * ecs +
        0.4 * correctness +
        0.2 * (1 - hallucination_rate)
    )

    return round(robustness, 4)


# =====================================
# RESPONSE TIME ANALYSIS
# =====================================

def compute_response_stats(times):
    """
    Compute response time statistics.
    """

    if len(times) == 0:
        return {
            "avg_time": 0,
            "min_time": 0,
            "max_time": 0,
            "std_time": 0
        }

    return {
        "avg_time": round(np.mean(times), 3),
        "min_time": round(np.min(times), 3),
        "max_time": round(np.max(times), 3),
        "std_time": round(np.std(times), 3)
    }