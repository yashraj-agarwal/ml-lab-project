"""Load the saved model and predict from static code features."""
import joblib
import pandas as pd
from pathlib import Path
from .features import STRUCTURAL_FEATURES, extract_features

ARTIFACT = Path(__file__).resolve().parents[1] / "models" / "codetrace.joblib"

def analyze(source: str):
    if not ARTIFACT.exists():
        raise RuntimeError("Model artifact not found. Train the project first with: python backend/train.py")
    artifact = joblib.load(ARTIFACT)
    pipeline = artifact["pipeline"]
    features = extract_features(source)
    row = {"code": source, **features}
    frame = pd.DataFrame([row])
    probabilities = pipeline.predict_proba(frame[["code"] + STRUCTURAL_FEATURES])[0]
    labels = pipeline.classes_
    ai_probability = float(probabilities[list(labels).index("ai")])
    return {
        "prediction": "ai" if ai_probability >= 0.5 else "human",
        "confidence": round(max(ai_probability, 1 - ai_probability), 4),
        "ai_probability": round(ai_probability, 4), "model": artifact["model_name"],
        "features": features,
        "explanation": [
            {"name": "Comment/code ratio", "value": round(features["comment_to_code_ratio"], 3)},
            {"name": "Average identifier length", "value": round(features["average_identifier_length"], 2)},
            {"name": "Functions / loops / conditions", "value": f'{features["function_count"]} / {features["loop_count"]} / {features["conditional_count"]}'},
            {"name": "AST nodes / nesting depth", "value": f'{features["ast_node_count"]} / {features["max_nesting_depth"]}'},
            {"name": "Tokens / unique-token ratio", "value": f'{features["token_count"]} / {features["unique_token_ratio"]:.2f}'},
        ],
    }
