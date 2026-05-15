import pandas as pd

# Load results
df = pd.read_csv("pilot_results.csv")

print("\n========== GLOBAL CORRELATION ==========")

# Correlation across all models
corr_ecs_correctness = df["ECS"].corr(df["correctness"])
corr_ecs_hallucination = df["ECS"].corr(df["hallucination_rate"])

print(f"ECS vs Correctness correlation: {round(corr_ecs_correctness, 4)}")
print(f"ECS vs Hallucination correlation: {round(corr_ecs_hallucination, 4)}")

print("\n========== PER-MODEL ANALYSIS ==========")

for model_name, group in df.groupby("model"):

    corr1 = group["ECS"].corr(group["correctness"])
    corr2 = group["ECS"].corr(group["hallucination_rate"])
    ecs_std = group["ECS"].std()

    print(f"\nModel: {model_name}")
    print(f"  ECS vs Correctness correlation: {round(corr1, 4)}")
    print(f"  ECS vs Hallucination correlation: {round(corr2, 4)}")
    print(f"  ECS standard deviation: {round(ecs_std, 4)}")

print("\n========================================\n")