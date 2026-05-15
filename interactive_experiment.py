
from user_input import get_user_expression
from perturbation_math import generate_equivalents
from llm_interface import query_model
from normalizer import to_symbolic
from correctness import compute_ground_truth, compute_correctness
from stability import compute_stability
from hallucination import analyze_hallucination
from config import MODELS

import time

NUM_PERTURBATIONS = 5


def run():

    expr = get_user_expression()

    print("\nExpression:", expr)

    forms = generate_equivalents(expr, NUM_PERTURBATIONS)

    print("\nGenerated forms:")
    for f in forms:
        print("-", f)

    ground_truth = compute_ground_truth(expr)

    for model in MODELS:

        print("\nMODEL:", model)

        outputs = []

        for form in forms:

            print("\nQuery:", form)

            start = time.time()

            out = query_model(model, form)

            elapsed = round(time.time() - start, 2)

            print("Output:", out)
            print("Time:", elapsed)

            symbolic = to_symbolic(out)

            outputs.append(symbolic)

        ecs = compute_stability([o for o in outputs if o is not None])

        correctness = compute_correctness(outputs, ground_truth)

        hallucinations = []

        for o in outputs:
            result = analyze_hallucination(o, ground_truth)
            hallucinations.append(result["type"])

        print("\nRESULTS")
        print("ECS:", ecs)
        print("Correctness:", correctness)
        print("Hallucinations:", hallucinations)


if __name__ == "__main__":
    run()