import os
import requests
from collections import Counter
from dotenv import load_dotenv
from config import TEMPERATURE, MAX_TOKENS, INTERVENTION_STRATEGIES

load_dotenv()
try:
    import streamlit as st
    API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
except Exception:
    API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# ============================================================
# OUTPUT CLEANING
# ============================================================

def clean_output(text, strategy="zero_shot"):
    """
    Extract the final symbolic expression from model output.

    For chain_of_thought and symbolic_scaffolding, the model is
    instructed to end with 'ANSWER: <expr>', so we look for that tag.
    For zero_shot / self_consistency, we take the last non-empty line.
    """
    if text is None:
        return None

    # Remove markdown backticks
    text = text.replace("`", "").strip()

    # --- CoT / Scaffolding: look for explicit ANSWER: tag ---
    if strategy in ("chain_of_thought", "symbolic_scaffolding"):
        for line in reversed(text.split("\n")):
            line = line.strip()
            if line.upper().startswith("ANSWER:"):
                expr = line[len("ANSWER:"):].strip()
                # strip any trailing equals
                if "=" in expr:
                    expr = expr.split("=")[-1].strip()
                return expr if expr else None
        # fallback: take last non-empty line (model didn't follow format)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            return None
        last = lines[-1]
        if "=" in last:
            last = last.split("=")[-1].strip()
        return last

    # --- Zero-shot / Self-consistency: last non-empty line ---
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return None
    last = lines[-1]
    if "=" in last:
        last = last.split("=")[-1].strip()
    return last


# ============================================================
# SINGLE API CALL
# ============================================================

def _call_api(model_name, system_prompt, user_prompt, temperature, max_tokens):
    """
    Low-level single API call. Returns raw text or None.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ]
    }
    try:
        response = requests.post(OPENROUTER_URL, headers=headers,
                                 json=payload, timeout=30)
    except Exception as e:
        print(f"      Request failed: {e}", flush=True)
        return None

    if response.status_code != 200:
        print(f"      API Error {response.status_code}: {response.text}", flush=True)
        return None

    result = response.json()
    return result["choices"][0]["message"]["content"].strip()


# ============================================================
# ORIGINAL INTERFACE (unchanged — used by experiment_math.py)
# ============================================================

def query_model(model_name, prompt):
    """
    Original zero-shot query. Kept for backward compatibility.
    """
    cfg = INTERVENTION_STRATEGIES["zero_shot"]
    raw = _call_api(
        model_name,
        cfg["system"],
        cfg["user"].format(expr=prompt),
        TEMPERATURE,
        MAX_TOKENS
    )
    return clean_output(raw, strategy="zero_shot")


# ============================================================
# INTERVENTION INTERFACE (new — used by intervention_study.py)
# ============================================================

def query_model_with_strategy(model_name, expr, strategy_name):
    """
    Query a model using a named intervention strategy.

    Parameters
    ----------
    model_name   : OpenRouter model string
    expr         : mathematical expression string
    strategy_name: one of INTERVENTION_STRATEGIES keys

    Returns
    -------
    cleaned expression string or None
    """
    cfg = INTERVENTION_STRATEGIES.get(strategy_name)
    if cfg is None:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    system_prompt = cfg["system"]
    user_prompt   = cfg["user"].format(expr=expr)

    # ---- Self-consistency: sample N times, take majority vote ----
    if strategy_name == "self_consistency":
        sc_samples     = cfg.get("_sc_samples", 3)
        sc_temperature = cfg.get("_sc_temperature", 0.4)

        raw_outputs = []
        for _ in range(sc_samples):
            raw = _call_api(model_name, system_prompt, user_prompt,
                            sc_temperature, MAX_TOKENS)
            cleaned = clean_output(raw, strategy="zero_shot")
            if cleaned:
                raw_outputs.append(cleaned)

        if not raw_outputs:
            return None

        # majority vote on string level (before symbolic parsing)
        counts = Counter(raw_outputs)
        majority = counts.most_common(1)[0][0]
        return majority

    # ---- All other strategies: single call ----
    raw = _call_api(model_name, system_prompt, user_prompt,
                    TEMPERATURE, MAX_TOKENS)
    return clean_output(raw, strategy=strategy_name)
