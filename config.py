# ====== MODELS ======
MODELS = [
    "meta-llama/llama-3-8b-instruct",
    "google/gemma-2-9b-it",
    "deepseek/deepseek-chat",
    "qwen/qwen-2.5-72b-instruct"
]

# ====== EXPERIMENT SETTINGS ======
NUM_SAMPLES = 30
NUM_PERTURBATIONS = 6

# ====== LLM SETTINGS ======
TEMPERATURE = 0.0
MAX_TOKENS = 256   # increased — CoT needs more tokens

# ====== INTERVENTION STRATEGIES ======
# Each strategy key maps to a (system_prompt, user_template) pair.
# {expr} in the user_template is replaced with the actual expression.

INTERVENTION_STRATEGIES = {

    # ------------------------------------------------------------------ #
    # BASELINE: plain zero-shot — same as original code                   #
    # ------------------------------------------------------------------ #
    "zero_shot": {
        "system": (
            "You are a mathematics engine.\n"
            "Return ONLY the final simplified symbolic expression.\n"
            "No explanation. Do NOT include equals signs. Do NOT include steps.\n"
            "Use Python-style math (x**2, sin(x))."
        ),
        "user": "Simplify: {expr}"
    },

    # ------------------------------------------------------------------ #
    # STRATEGY 1: Chain-of-Thought                                         #
    # Forces the model to reason step-by-step before answering.           #
    # Last line must be the bare expression so our parser can extract it. #
    # ------------------------------------------------------------------ #
    "chain_of_thought": {
        "system": (
            "You are a mathematics engine.\n"
            "Think step by step. Show your reasoning clearly.\n"
            "At the very end, write EXACTLY one line starting with 'ANSWER:' "
            "followed by the simplified expression in Python-style math (x**2, sin(x)).\n"
            "Example last line: ANSWER: x + 3"
        ),
        "user": "Simplify step by step: {expr}"
    },

    # ------------------------------------------------------------------ #
    # STRATEGY 2: Symbolic Scaffolding                                     #
    # Guides the model to first identify structure, then simplify.        #
    # ------------------------------------------------------------------ #
    "symbolic_scaffolding": {
        "system": (
            "You are a mathematics engine. Follow these steps strictly:\n"
            "Step 1 — List all variables present.\n"
            "Step 2 — Identify the expression type (polynomial, trigonometric, logarithmic, rational).\n"
            "Step 3 — Apply the relevant simplification rule.\n"
            "Step 4 — Write the final simplified form.\n"
            "End your response with EXACTLY one line: ANSWER: <expression>\n"
            "Use Python-style math (x**2, sin(x)). No equals signs."
        ),
        "user": "Simplify using structured steps: {expr}"
    },

    # ------------------------------------------------------------------ #
    # STRATEGY 3: Self-Consistency                                         #
    # We query the model 3 times (temperature slightly raised) and take   #
    # the majority symbolic answer. Handled specially in query logic.     #
    # ------------------------------------------------------------------ #
    "self_consistency": {
        "system": (
            "You are a mathematics engine.\n"
            "Return ONLY the final simplified symbolic expression.\n"
            "No explanation. Do NOT include equals signs. Do NOT include steps.\n"
            "Use Python-style math (x**2, sin(x))."
        ),
        "user": "Simplify: {expr}",
        "_sc_samples": 3,        # internal flag — number of samples
        "_sc_temperature": 0.4   # slight randomness to get diverse samples
    },
}

# Strategies used in the intervention experiment
INTERVENTION_STRATEGY_NAMES = list(INTERVENTION_STRATEGIES.keys())
