Overview: Perturbation-Based LLM Evaluation

This project implements a research framework designed to detect, classify, and mitigate mathematical hallucinations in Large Language Models (LLMs). Unlike standard factual errors, these hallucinations are "surface-dependent," meaning an LLM may solve a problem correctly in one format but fail when the same expression is slightly rearranged.  

Key Features

Perturbation Engine: Automatically generates 10 variants of symbolic expressions, including 6 semantic-preserving forms and 4 adversarial "traps" (e.g., variable remapping and identity multiplication).  

Hallucination Taxonomy: A formal classification system that categorizes errors into six grounded types, such as Extrinsic Variable Invention and Domain Transformation.  

Quantitative Metrics: Uses three specialized metrics to measure model reliability:  

Expression Consistency Score (ECS): Measures if the model provides the same answer across different syntactic forms.  

Correctness Rate: Verifies algebraic accuracy against SymPy ground truth.  

Robustness Score: A weighted composite of consistency and accuracy.  

Prompting Interventions: Evaluates strategies like Symbolic Scaffolding and Chain-of-Thought to reduce error rates.  

Core Findings

 Systemic Fragility: In a study of 900 queries, models were correct only 57.6% of the time.  

Adversarial Brittleness: The "identity multiplication" trap (e⋅y/y) was the most effective failure trigger, causing models to invent variables that weren't in the original problem.  

Scaffolding Success: Symbolic Scaffolding (a four-step structured decomposition) completely eliminated hallucinations for DeepSeek-Chat during the intervention study.  

Tech Stack

 Language: Python   

Symbolic Mathematics: SymPy (for parsing and canonical normalization)   

LLM Integration: OpenRouter API
