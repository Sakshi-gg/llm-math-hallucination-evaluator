"""
intervention_study.py
=====================
Research Layer 3: Intervention Study

Runs all 4 prompting strategies (zero_shot, chain_of_thought,
symbolic_scaffolding, self_consistency) against every model on a
subset of expressions. Saves results to intervention_results.csv.

Research question answered:
  "Does the choice of prompting strategy systematically reduce
   mathematical hallucination rates and improve ECS across models?"
"""

from config import MODELS, INTERVENTION_STRATEGY_NAMES
from dataset_math import generate_expressions
from perturbation_math import generate_equivalents
from llm_interface import query_model_with_strategy
from normalizer import to_symbolic
from stability import compute_stability, compute_robustness_score
from correctness import compute_ground_truth, compute_correctness
from hallucination import analyze_hallucination

import pandas as pd
import time

# Use a smaller subset for the intervention study so it stays tractable
INTERVENTION_SAMPLES = 10   # change to 30 for full run
INTERVENTION_PERTURBATIONS = 4


def run_intervention():

    expressions = generate_expressions()[:INTERVENTION_SAMPLES]
    results = []

    total_runs = len(expressions) * len(MODELS) * len(INTERVENTION_STRATEGY_NAMES)
    run_counter = 0

    print(f"\n{'='*60}", flush=True)
    print(f"  INTERVENTION STUDY", flush=True)
    print(f"  Expressions : {len(expressions)}", flush=True)
    print(f"  Models      : {len(MODELS)}", flush=True)
    print(f"  Strategies  : {INTERVENTION_STRATEGY_NAMES}", flush=True)
    print(f"  Total runs  : {total_runs}", flush=True)
    print(f"{'='*60}\n", flush=True)

    for expr_idx, expr in enumerate(expressions):

        print(f"\n[Expr {expr_idx+1}/{len(expressions)}] {expr}", flush=True)

        forms        = generate_equivalents(expr, INTERVENTION_PERTURBATIONS)
        ground_truth = compute_ground_truth(expr)

        for model in MODELS:
            for strategy in INTERVENTION_STRATEGY_NAMES:

                run_counter += 1
                print(f"\n  Run {run_counter}/{total_runs} | "
                      f"Model: {model} | Strategy: {strategy}", flush=True)

                symbolic_outputs = []
                response_times   = []

                for form_idx, form in enumerate(forms):

                    print(f"    Form {form_idx+1}/{len(forms)}: {form[:60]}", flush=True)

                    try:
                        start   = time.time()
                        output  = query_model_with_strategy(model, form, strategy)
                        elapsed = round(time.time() - start, 2)
                        print(f"      → {output}  ({elapsed}s)", flush=True)
                    except Exception as e:
                        print(f"      ❌ ERROR: {e}", flush=True)
                        output  = None
                        elapsed = None

                    response_times.append(elapsed)
                    symbolic_outputs.append(to_symbolic(output))

                valid_outputs = [s for s in symbolic_outputs if s is not None]

                if not valid_outputs:
                    print("    ⚠ All outputs invalid — skipping.", flush=True)
                    continue

                # ---- Metrics ----
                ecs         = compute_stability(valid_outputs)
                correctness = compute_correctness(valid_outputs, ground_truth)

                hallucinated_count = 0
                hallucination_types = []
                for s in symbolic_outputs:
                    res = analyze_hallucination(s, ground_truth)
                    hallucination_types.append(res["type"])
                    if res["hallucinated"]:
                        hallucinated_count += 1

                hallucination_rate = hallucinated_count / len(symbolic_outputs)
                robustness_score   = compute_robustness_score(
                    ecs, correctness, hallucination_rate
                )

                valid_times = [t for t in response_times if t is not None]
                avg_time    = sum(valid_times) / len(valid_times) if valid_times else None

                print(f"    ECS={ecs:.3f}  Corr={correctness:.3f}  "
                      f"Halluc={hallucination_rate:.3f}  "
                      f"Robust={robustness_score:.3f}", flush=True)

                results.append({
                    "expression":          expr,
                    "model":               model,
                    "strategy":            strategy,
                    "ECS":                 ecs,
                    "correctness":         correctness,
                    "hallucination_rate":  hallucination_rate,
                    "robustness_score":    robustness_score,
                    "hallucination_types": str(hallucination_types),
                    "avg_response_time":   avg_time,
                    "num_valid_outputs":   len(valid_outputs)
                })

    # ---- Save ----
    df = pd.DataFrame(results)
    df.to_csv("intervention_results.csv", index=False)
    print("\n✅ Saved to intervention_results.csv", flush=True)

    # ---- Summary ----
    if df.empty:
        print("No results collected.", flush=True)
        return

    print(f"\n{'='*60}", flush=True)
    print("  INTERVENTION STUDY SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)

    summary = (
        df.groupby(["model", "strategy"])
        .agg(
            avg_ECS=("ECS", "mean"),
            avg_correctness=("correctness", "mean"),
            avg_hallucination=("hallucination_rate", "mean"),
            avg_robustness=("robustness_score", "mean"),
        )
        .round(4)
        .reset_index()
    )

    print(summary.to_string(index=False), flush=True)

    # ---- Best strategy per model ----
    print(f"\n{'='*60}", flush=True)
    print("  BEST STRATEGY PER MODEL (by Robustness Score)", flush=True)
    print(f"{'='*60}", flush=True)

    for model_name, grp in summary.groupby("model"):
        best = grp.loc[grp["avg_robustness"].idxmax()]
        baseline = grp[grp["strategy"] == "zero_shot"]["avg_robustness"].values
        if len(baseline) > 0:
            delta = best["avg_robustness"] - baseline[0]
            print(f"  {model_name}", flush=True)
            print(f"    Best strategy : {best['strategy']}", flush=True)
            print(f"    Robustness    : {best['avg_robustness']:.4f} "
                  f"(+{delta:.4f} vs zero_shot)", flush=True)

    print(f"\n{'='*60}\n", flush=True)


if __name__ == "__main__":
    run_intervention()
