import sympy as sp

def extract_features(expr):
    """
    Extracts structural properties for deep research analysis.
    """
    if expr is None:
        return None
    try:
        # We use SymPy to look at the 'DNA' of the math
        return {
            "variables": expr.free_symbols,
            "degree": sp.total_degree(expr) if hasattr(expr, 'is_Polynomial') else None,
            "ops": [type(a) for a in sp.preorder_traversal(expr)], # Structure of ops
            "is_rational": expr.is_rational_function(),
            "has_trig": any(isinstance(node, (sp.sin, sp.cos, sp.tan)) for node in sp.preorder_traversal(expr))
        }
    except:
        return None

def analyze_hallucination(predicted, ground_truth):
    """
    RESEARCH TAXONOMY: Categorizes hallucinations into specific failure modes.
    Used to convince advisors of 'Scientific Depth.'
    """
    if predicted is None:
        return {"hallucinated": True, "type": "Invalid_Syntax_Failure"}

    # 1. Check for Exact Symbolic Equality (No Hallucination)
    try:
        if sp.simplify(predicted - ground_truth) == 0:
            return {"hallucinated": False, "type": "none"}
    except:
        pass

    pred_feat = extract_features(predicted)
    gt_feat = extract_features(ground_truth)

    if not pred_feat or not gt_feat:
        return {"hallucinated": True, "type": "Structural_Collapse"}

    # 2. Variable Hallucination (Entity Mismatch)
    if pred_feat["variables"] != gt_feat["variables"]:
        # Specific check: Did it add a variable that wasn't there?
        if pred_feat["variables"] - gt_feat["variables"]:
            return {"hallucinated": True, "type": "Extrinsic_Variable_Invention"}
        return {"hallucinated": True, "type": "Variable_Loss_Error"}

    # 3. Structural/Logic Hallucination (The 'Brittleness' Check)
    if pred_feat["degree"] != gt_feat["degree"]:
        return {"hallucinated": True, "type": "Degree_Mismatch_Collapse"}

    if pred_feat["has_trig"] != gt_feat["has_trig"]:
        return {"hallucinated": True, "type": "Domain_Transformation_Hallucination"}

    # 4. Identity/Zero-Trick Failures 
    # (If the model fails on expr+0 but got expr right)
    # This is handled in the experiment runner by comparing ECS vs Accuracy
    
    return {"hallucinated": True, "type": "General_Logic_Hallucination"}
