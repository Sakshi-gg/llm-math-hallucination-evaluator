# LLM Math Hallucination Evaluator

This project implements a research framework designed to detect, classify, and mitigate **mathematical hallucinations** in Large Language Models (LLMs). Unlike standard factual errors, these hallucinations are "surface-dependent," meaning an LLM may solve a problem correctly in one format but fail when the same expression is slightly rearranged.

## 🚀 Key Features

* **Perturbation Engine**: Automatically generates 10 variants of symbolic expressions, including 6 semantic-preserving forms and 4 adversarial "traps" (e.g., variable remapping and identity multiplication).
* **Hallucination Taxonomy**: A formal classification system that categorizes errors into six grounded types, such as **Extrinsic Variable Invention** and **Domain Transformation**.
* **Quantitative Metrics**: Uses three specialized metrics to measure model reliability:
    * **Expression Consistency Score (ECS)**: Measures if the model provides the same answer across different syntactic forms.
    * **Correctness Rate**: Verifies algebraic accuracy against SymPy ground truth.
    * **Robustness Score**: A weighted composite of consistency and accuracy.
* **Prompting Interventions**: Evaluates strategies like **Symbolic Scaffolding** and **Chain-of-Thought** to reduce error rates.

## 📊 Results
<img width="1366" height="768" alt="Screenshot from 2026-05-15 20-41-02" src="https://github.com/user-attachments/assets/46cdd164-27c9-4b54-8d9e-6b8dfa9362d0" />
<img width="1366" height="768" alt="Screenshot from 2026-05-15 20-41-15" src="https://github.com/user-attachments/assets/5860bcf3-001c-443f-896c-f46acbac9178" />
<img width="1366" height="768" alt="Screenshot from 2026-05-15 20-41-21" src="https://github.com/user-attachments/assets/f61c75b5-dabc-4fe3-8fb1-6351d7170acc" />
<img width="1366" height="768" alt="Screenshot from 2026-05-15 20-41-41" src="https://github.com/user-attachments/assets/1afcc0e0-1b7e-403a-ab69-8de9c49d628a" />
<img width="1366" height="768" alt="Screenshot from 2026-05-15 20-41-47" src="https://github.com/user-attachments/assets/bd3aa15e-fbc1-4c91-86d0-57522720d301" />
<img width="1366" height="768" alt="Screenshot from 2026-05-15 20-41-59" src="https://github.com/user-attachments/assets/9d4cbd10-b5a6-4032-a8c8-4eccb26a6196" />


### Core Findings

* **Systemic Fragility**: In a study of 900 queries, models were correct only **57.6%** of the time.
* **Adversarial Brittleness**: The "identity multiplication" trap triggered 52 instances of **Extrinsic Variable Invention**, the highest count in the dataset.
* **Scaffolding Success**: **Symbolic Scaffolding** completely eliminated hallucinations for DeepSeek-Chat, achieving a perfect 1.0 Robustness Score.

## 🛠️ Tech Stack

* **Language**: Python
* **Symbolic Mathematics**: `SymPy` for parsing and canonical normalization
* **LLM Integration**: OpenRouter API

