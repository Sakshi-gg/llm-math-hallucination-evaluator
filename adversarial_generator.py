import random
import sympy as sp

def generate_adversarial_research(expr_str):
    """
    RESEARCH MODULE: Generates traps to test 'Symbolic Maturity' vs 'Pattern Matching'.
    """
    variants = []
    
    # TRAP 1: Variable Remapping (Tests if the model is 'brittle' to non-standard letters)
    try:
        expr = sp.sympify(expr_str)
        symbols = list(expr.free_symbols)
        if symbols:
            # Swap 'x' for 'beta' or 'theta' - LLMs often struggle with non-standard tokens
            mapping = {s: sp.Symbol(f"beta_{random.randint(1,5)}") for s in symbols}
            variants.append(str(expr.subs(mapping)))
    except: pass

    # TRAP 2: Identity Wrapping (Tests if the model can 'see through' redundant math)
    variants.append(f"exp(log({expr_str}))") # Log-Exp identity
    variants.append(f"(({expr_str}) * (y/y))") # Identity multiplication (requires y != 0)
    
    # TRAP 3: Zero-Trick Injection (Tests if model treats +0 as a substantive change)
    variants.append(f"({expr_str}) + (x - x)") 

    return list(set(variants))
