from config import MODELS, NUM_PERTURBATIONS, NUM_SAMPLES
from dataset_math import generate_expressions
from perturbation_math import generate_equivalents
from adversarial_generator import generate_adversarial_research as generate_adversarial
from llm_interface import query_model
from normalizer import to_symbolic
from stability import compute_stability, compute_robustness_score
from correctness import compute_ground_truth, compute_correctness
from hallucination import analyze_hallucination

import pandas as pd
import time

print("Starting experiment...\n", flush=True)


# ---------------------------------------------------------------
# Perturbation type labels
# These match the order: equivalents first, then adversarial forms
# ---------------------------------------------------------------
EQUIV_LABELS = [
    "original", "expanded", "factored", "simplified",
    "reordered", "parenthesized"
]
ADV_LABELS = [
    "adv_variable_remap", "adv_log_exp_wrap",
    "adv_identity_mul", "adv_zero_trick"
]


def label_forms(equivalents, adversarial_forms):
    """
    Attach a human-readable perturbation type label to each form.
    Returns list of (form_string, perturbation_type) tuples.
    """
    labeled = []
    for i, f in enumerate(equivalents):
        label = EQUIV_LABELS[i] if i < len(EQUIV_LABELS) else f"equiv_{i}"
        labeled.append((f, label))
    for i, f in enumerate(adversarial_forms):
        label = ADV_LABELS[i] if i < len(ADV_LABELS) else f"adv_{i}"
        labeled.append((f, label))
    return labeled


def run():

    expressions = generate_expressions(NUM_SAMPLES)

    results = []

    for expr_index, expr in enumerate(expressions):

        print(f"\nProcessing expression {expr_index+1}/{len(expressions)}: {expr}", flush=True)

        # --------------------------------
        # Generate perturbations
        # --------------------------------

        equivalents     = generate_equivalents(expr, NUM_PERTURBATIONS)
        adversarial_forms = generate_adversarial(expr)
        labeled_forms   = label_forms(equivalents, adversarial_forms)

        if len(labeled_forms) < NUM_PERTURBATIONS:
            print("⚠ Limited perturbation diversity.", flush=True)

        ground_truth = compute_ground_truth(expr)

        # --------------------------------
        # Run models
        # --------------------------------

        for model in MODELS:

            print(f"\nUsing model: {model}", flush=True)

            symbolic_outputs = []
            response_times   = []
            perturbation_types_seen = []

            for form_index, (form, ptype) in enumerate(labeled_forms):

                print(f"   Querying form {form_index+1}/{len(labeled_forms)} [{ptype}]", flush=True)

                try:
                    start   = time.time()
                    output  = query_model(model, form)
                    elapsed = round(time.time() - start, 2)
                    print(f"      Response time: {elapsed}s", flush=True)
                    print(f"      Raw output: {output}", flush=True)
                except Exception as e:
                    print(f"      ❌ API ERROR: {e}", flush=True)
                    output  = None
                    elapsed = None

                response_times.append(elapsed)
                perturbation_types_seen.append(ptype)

                symbolic = to_symbolic(output)
                print(f"      Cleaned symbolic: {symbolic}", flush=True)
                symbolic_outputs.append(symbolic)

            valid_outputs = [s for s in symbolic_outputs if s is not None]

            if len(valid_outputs) == 0:
                print("⚠ All outputs invalid — skipping.", flush=True)
                continue

            # --------------------------------
            # Metrics computation
            # --------------------------------

            ecs         = compute_stability(valid_outputs)
            correctness = compute_correctness(valid_outputs, ground_truth)

            # --------------------------------
            # Hallucination analysis — now with perturbation type
            # --------------------------------

            hallucination_types    = []
            hallucinated_count     = 0
            perturbation_halluc_map = {}   # ptype -> hallucination type

            for s, ptype in zip(symbolic_outputs, perturbation_types_seen):
                result = analyze_hallucination(s, ground_truth)
                hallucination_types.append(result["type"])
                if result["hallucinated"]:
                    hallucinated_count += 1
                # track which perturbation types cause which hallucination
                perturbation_halluc_map[ptype] = result["type"]

            hallucination_rate = hallucinated_count / len(symbolic_outputs)

            # --------------------------------
            # Robustness score
            # --------------------------------

            robustness_score = compute_robustness_score(
                ecs, correctness, hallucination_rate
            )

            # --------------------------------
            # Response time stats
            # --------------------------------

            valid_times = [t for t in response_times if t is not None]
            avg_time    = sum(valid_times) / len(valid_times) if valid_times else None

            # --------------------------------
            # Print results
            # --------------------------------

            print(f"\n   --> ECS = {ecs}", flush=True)
            print(f"   --> Correctness = {correctness}", flush=True)
            print(f"   --> Hallucination Rate = {hallucination_rate}", flush=True)
            print(f"   --> Robustness Score = {robustness_score}", flush=True)
            print(f"   --> Hallucination Types = {hallucination_types}", flush=True)
            print(f"   --> Perturbation→Hallucination Map = {perturbation_halluc_map}", flush=True)

            # --------------------------------
            # Save results
            # --------------------------------

            results.append({
                "expression":              expr,
                "model":                   model,
                "ECS":                     ecs,
                "correctness":             correctness,
                "hallucination_rate":      hallucination_rate,
                "robustness_score":        robustness_score,
                "hallucination_types":     str(hallucination_types),
                "perturbation_types":      str(perturbation_types_seen),
                "perturbation_halluc_map": str(perturbation_halluc_map),
                "avg_response_time":       avg_time,
                "num_outputs":             len(valid_outputs)
            })

    # --------------------------------
    # Save CSV
    # --------------------------------

    df = pd.DataFrame(results)
    df.to_csv("pilot_results.csv", index=False)

    # --------------------------------
    # Per-model summary
    # --------------------------------

    if not df.empty:

        print("\n========== PER-MODEL SUMMARY ==========\n", flush=True)

        for model_name, group in df.groupby("model"):

            avg_ecs          = group["ECS"].mean()
            avg_correctness  = group["correctness"].mean()
            avg_hallucination= group["hallucination_rate"].mean()
            avg_robustness   = group["robustness_score"].mean()
            avg_time         = group["avg_response_time"].mean()
            fully_stable_pct = (group["ECS"] == 1.0).mean() * 100
            worst_ecs        = group["ECS"].min()

            print(f"Model: {model_name}", flush=True)
            print(f"   Average ECS:                {round(avg_ecs,4)}", flush=True)
            print(f"   Average Correctness:        {round(avg_correctness,4)}", flush=True)
            print(f"   Average Hallucination Rate: {round(avg_hallucination,4)}", flush=True)
            print(f"   Average Robustness Score:   {round(avg_robustness,4)}", flush=True)
            print(f"   Average Response Time:      {round(avg_time,2)} sec", flush=True)
            print(f"   % Fully Stable (ECS=1):     {round(fully_stable_pct,2)}%", flush=True)
            print(f"   Worst ECS observed:         {round(worst_ecs,4)}\n", flush=True)

        print("========================================\n", flush=True)

    print("Experiment finished successfully.", flush=True)
    print("Results saved to pilot_results.csv\n", flush=True)


if __name__ == "__main__":
    run()
