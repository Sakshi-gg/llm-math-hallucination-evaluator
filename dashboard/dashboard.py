import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import time

from src.perturbation_math import generate_equivalents
from src.llm_interface import query_model, query_model_with_strategy
from src.normalizer import to_symbolic
from src.correctness import compute_ground_truth, compute_correctness
from src.stability import (
    compute_stability,
    compute_robustness_score,
    compute_response_stats
)
from src.hallucination import analyze_hallucination
from config import MODELS, INTERVENTION_STRATEGY_NAMES
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PILOT_RESULTS_PATH = os.path.join(PROJECT_ROOT, "data", "pilot_results.csv")
INTERVENTION_RESULTS_PATH = os.path.join(PROJECT_ROOT, "data", "intervention_results.csv")

st.set_page_config(page_title="LLM Robustness Dashboard", layout="wide")

st.title("📊 LLM Robustness Evaluation Dashboard")


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    try:
        return pd.read_csv(PILOT_RESULTS_PATH)
    except:
        return pd.DataFrame()


df = load_data()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Controls")

if not df.empty:

    selected_models = st.sidebar.multiselect(
        "Select Models",
        df["model"].unique(),
        default=df["model"].unique()
    )

    filtered_df = df[df["model"].isin(selected_models)]

else:
    filtered_df = pd.DataFrame()


# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Model Comparison",
    "Stability Analysis",
    "Expression Drilldown",
    "Live Expression Test",
    "Intervention Study",
    "Perturbation Causal Map"
])


# =========================================================
# TAB 1
# =========================================================

with tab1:

    st.header("📌 Model-Level Metrics")

    if filtered_df.empty:
        st.warning("No experiment data available.")

    else:

        summary = filtered_df.groupby("model").agg({
            "ECS": ["mean", "std"],
            "correctness": "mean",
            "hallucination_rate": "mean"
        })

        summary.columns = [
            "ECS_mean",
            "ECS_std",
            "correctness_mean",
            "hallucination_mean"
        ]

        summary = summary.reset_index()

        summary["robustness"] = (
            0.4 * summary["ECS_mean"] +
            0.4 * summary["correctness_mean"] +
            0.2 * (1 - summary["hallucination_mean"])
        )

        st.subheader("Model Summary Table")
        st.dataframe(summary)

        st.subheader("🏆 Model Ranking")

        ranked = summary.sort_values("robustness", ascending=False)

        st.dataframe(ranked)

        fig, ax = plt.subplots()

        sns.barplot(
            data=ranked,
            x="model",
            y="robustness",
            ax=ax
        )

        plt.xticks(rotation=45)

        ax.set_ylabel("Robustness Score")

        st.pyplot(fig)


# =========================================================
# TAB 2
# =========================================================

with tab2:

    st.header("📈 Stability Correlation")

    if filtered_df.empty:
        st.warning("No experiment data available.")

    else:

        fig1, ax1 = plt.subplots()

        sns.scatterplot(
            data=filtered_df,
            x="ECS",
            y="hallucination_rate",
            hue="model",
            ax=ax1
        )

        st.pyplot(fig1)

        corr = filtered_df["ECS"].corr(
            filtered_df["hallucination_rate"]
        )

        st.write("Correlation:", round(corr, 4))


# =========================================================
# TAB 3
# =========================================================

with tab3:

    st.header("🔎 Expression Drilldown")

    if filtered_df.empty:

        st.warning("No experiment data available.")

    else:

        model_choice = st.selectbox(
            "Choose Model",
            filtered_df["model"].unique()
        )

        expr_choice = st.selectbox(
            "Choose Expression",
            filtered_df[
                filtered_df["model"] == model_choice
            ]["expression"].unique()
        )

        expr_data = filtered_df[
            (filtered_df["model"] == model_choice) &
            (filtered_df["expression"] == expr_choice)
        ]

        st.dataframe(expr_data)


# =========================================================
# TAB 4
# =========================================================

with tab4:

    st.header("🧪 Live LLM Robustness Test")

    expr = st.text_input("Enter a mathematical expression")

    num_forms = st.slider(
        "Number of perturbations",
        min_value=2,
        max_value=10,
        value=5
    )

    if st.button("Run Evaluation"):

        if expr.strip() == "":
            st.warning("Please enter an expression")

        else:

            forms = generate_equivalents(expr, num_forms)

            st.subheader("Generated Expressions")

            for f in forms:
                st.write("-", f)

            ground_truth = compute_ground_truth(expr)

            results = []

            for model in MODELS:

                st.subheader(f"Model: {model}")

                outputs = []
                times = []

                for form in forms:

                    start = time.time()

                    try:
                        out = query_model(model, form)
                    except:
                        out = None

                    elapsed = round(time.time() - start, 2)

                    symbolic = to_symbolic(out)

                    outputs.append(symbolic)

                    times.append(elapsed)

                    st.write("Input:", form)
                    st.write("Output:", out)
                    st.write("Time:", elapsed)

                valid_outputs = [o for o in outputs if o is not None]

                ecs = compute_stability(valid_outputs)

                correctness = compute_correctness(
                    valid_outputs,
                    ground_truth
                )

                hallucinated_count = 0

                for o in outputs:

                    result = analyze_hallucination(
                        o,
                        ground_truth
                    )

                    if result["hallucinated"]:
                        hallucinated_count += 1

                hallucination_rate = hallucinated_count / len(outputs)

                robustness = compute_robustness_score(
                    ecs,
                    correctness,
                    hallucination_rate
                )

                time_stats = compute_response_stats(times)

                st.write("ECS:", ecs)
                st.write("Correctness:", correctness)
                st.write("Hallucination Rate:", hallucination_rate)
                st.write("Robustness Score:", robustness)
                st.write("Avg Response Time:", time_stats["avg_time"])

                results.append({
                    "model": model,
                    "ECS": ecs,
                    "correctness": correctness,
                    "hallucination_rate": hallucination_rate,
                    "robustness": robustness,
                    "avg_time": time_stats["avg_time"]
                })

            result_df = pd.DataFrame(results)

            st.subheader("Model Comparison")

            st.dataframe(result_df)

            fig, ax = plt.subplots()

            sns.barplot(
                data=result_df,
                x="model",
                y="robustness",
                ax=ax
            )

            plt.xticks(rotation=45)

            st.pyplot(fig)

# =========================================================
# TAB 5 — INTERVENTION STUDY
# =========================================================

with tab5:

    st.header("🧬 Intervention Study: Strategy Comparison")

    @st.cache_data
    def load_intervention_data():
        try:
            return pd.read_csv(INTERVENTION_RESULTS_PATH)
        except:
            return pd.DataFrame()

    idf = load_intervention_data()

    if idf.empty:
        st.warning(
            "No intervention results found. "
            "Run `python intervention_study.py` first to generate `intervention_results.csv`."
        )
    else:
        st.subheader("Strategy × Model — Average Metrics")

        summary_i = (
            idf.groupby(["model", "strategy"])
            .agg(
                avg_ECS=("ECS", "mean"),
                avg_correctness=("correctness", "mean"),
                avg_hallucination=("hallucination_rate", "mean"),
                avg_robustness=("robustness_score", "mean"),
            )
            .round(4)
            .reset_index()
        )

        st.dataframe(summary_i)

        # ---- Grouped bar: Robustness by strategy per model ----
        st.subheader("Robustness Score by Strategy (per Model)")

        fig_i, ax_i = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=summary_i,
            x="model",
            y="avg_robustness",
            hue="strategy",
            ax=ax_i
        )
        plt.xticks(rotation=30, ha="right")
        ax_i.set_ylabel("Avg Robustness Score")
        ax_i.set_title("Prompting Strategy Impact on Robustness")
        ax_i.legend(title="Strategy", bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        st.pyplot(fig_i)

        # ---- Hallucination reduction ----
        st.subheader("Hallucination Rate by Strategy (per Model)")

        fig_h, ax_h = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=summary_i,
            x="model",
            y="avg_hallucination",
            hue="strategy",
            ax=ax_h
        )
        plt.xticks(rotation=30, ha="right")
        ax_h.set_ylabel("Avg Hallucination Rate")
        ax_h.set_title("Hallucination Rate by Prompting Strategy")
        ax_h.legend(title="Strategy", bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        st.pyplot(fig_h)

        # ---- Delta vs baseline ----
        st.subheader("Improvement over Zero-Shot Baseline (Δ Robustness)")

        baseline = summary_i[summary_i["strategy"] == "zero_shot"][["model", "avg_robustness"]]
        baseline = baseline.rename(columns={"avg_robustness": "baseline_robustness"})
        delta_df = summary_i.merge(baseline, on="model")
        delta_df["delta_robustness"] = (
            delta_df["avg_robustness"] - delta_df["baseline_robustness"]
        ).round(4)
        delta_non_baseline = delta_df[delta_df["strategy"] != "zero_shot"]

        fig_d, ax_d = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=delta_non_baseline,
            x="model",
            y="delta_robustness",
            hue="strategy",
            ax=ax_d
        )
        ax_d.axhline(0, color="black", linewidth=0.8, linestyle="--")
        plt.xticks(rotation=30, ha="right")
        ax_d.set_ylabel("Δ Robustness vs Zero-Shot")
        ax_d.set_title("Which strategy beats zero-shot?")
        ax_d.legend(title="Strategy", bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        st.pyplot(fig_d)

        # ---- Best strategy per model ----
        st.subheader("Best Strategy Per Model")
        best_per_model = (
            summary_i.loc[summary_i.groupby("model")["avg_robustness"].idxmax()]
            [["model", "strategy", "avg_robustness", "avg_hallucination"]]
        )
        st.dataframe(best_per_model)


# =========================================================
# TAB 6 — PERTURBATION CAUSAL MAP
# =========================================================

with tab6:

    st.header("🗺️ Perturbation → Hallucination Causal Map")

    st.markdown(
        "This tab shows **which perturbation type triggers which hallucination type** "
        "for each model — the causal analysis layer of the research."
    )

    if filtered_df.empty or "perturbation_halluc_map" not in filtered_df.columns:
        st.warning(
            "No causal map data found. "
            "Re-run `python experiment_math.py` with the updated code to populate this column."
        )
    else:
        model_choice_cm = st.selectbox(
            "Choose Model", filtered_df["model"].unique(), key="cm_model"
        )

        model_df = filtered_df[filtered_df["model"] == model_choice_cm].copy()

        # Parse the perturbation_halluc_map dict column
        rows = []
        for _, row in model_df.iterrows():
            try:
                mapping = ast.literal_eval(row["perturbation_halluc_map"])
                for ptype, htype in mapping.items():
                    rows.append({
                        "expression":        row["expression"],
                        "perturbation_type": ptype,
                        "hallucination_type": htype
                    })
            except Exception:
                continue

        if not rows:
            st.warning("Could not parse causal map data.")
        else:
            causal_df = pd.DataFrame(rows)

            # ---- Frequency heatmap ----
            pivot = (
                causal_df
                .groupby(["perturbation_type", "hallucination_type"])
                .size()
                .unstack(fill_value=0)
            )

            st.subheader("Heatmap: Perturbation Type × Hallucination Type (count)")
            fig_c, ax_c = plt.subplots(
                figsize=(max(8, len(pivot.columns)), max(4, len(pivot)))
            )
            sns.heatmap(
                pivot,
                annot=True,
                fmt="d",
                cmap="YlOrRd",
                ax=ax_c,
                linewidths=0.5
            )
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig_c)

            # ---- Raw table ----
            st.subheader("Raw Causal Data")
            st.dataframe(causal_df)

            # ---- Most dangerous perturbation ----
            halluc_counts = (
                causal_df[causal_df["hallucination_type"] != "none"]
                .groupby("perturbation_type")
                .size()
                .sort_values(ascending=False)
            )
            if not halluc_counts.empty:
                st.subheader("Most Hallucination-Inducing Perturbation Types")
                st.bar_chart(halluc_counts)
