"""Train and evaluate CodeTrace models on a labeled CSV dataset."""
import json
import sys
import hashlib
import platform
import sklearn
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from .features import STRUCTURAL_FEATURES, extract_features
except ImportError:
    from features import STRUCTURAL_FEATURES, extract_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "samples.csv"
MODELS = ROOT / "models"
SEED = 42


def load_dataset(path=DATA):
    df = pd.read_csv(path)
    original_rows = len(df)
    required = {"code", "label"}
    if not required.issubset(df.columns):
        raise ValueError("CSV must contain code and label columns.")
    df = df.dropna(subset=["code", "label"]).copy()
    df["code"] = df["code"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[(df.code != "") & df.label.isin(["human", "ai"])]
    if (df.groupby("code").label.nunique() > 1).any():
        raise ValueError("Identical code has conflicting labels. Resolve these rows before training.")
    df = df.drop_duplicates(subset=["code"])
    if df.label.nunique() != 2 or len(df) < 12:
        raise ValueError("Need at least 12 unique, non-empty examples across human and ai labels.")
    df.attrs["source_rows"] = original_rows
    df.attrs["removed_rows"] = original_rows - len(df)
    fallback_groups = pd.Series([f"sample-{i}" for i in df.index], index=df.index)
    df["group"] = df["problem_id"].fillna(fallback_groups).astype(str) if "problem_id" in df else fallback_groups
    return df.reset_index(drop=True)


def make_pipeline(model):
    prep = ColumnTransformer([
        ("lexical", TfidfVectorizer(max_features=1200, ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b\w+\b"), "code"),
        ("structure", StandardScaler(), STRUCTURAL_FEATURES),
    ])
    return Pipeline([("features", prep), ("classifier", model)])


def metric_row(y, pred):
    return {"accuracy": round(accuracy_score(y, pred), 4), "precision": round(precision_score(y, pred, pos_label="ai", zero_division=0), 4), "recall": round(recall_score(y, pred, pos_label="ai", zero_division=0), 4), "f1": round(f1_score(y, pred, pos_label="ai", zero_division=0), 4)}


def split_groups(df):
    first = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    trainval_idx, test_idx = next(first.split(df, groups=df.group))
    trainval = df.iloc[trainval_idx]
    second = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + 1)
    train_idx, val_idx = next(second.split(trainval, groups=trainval.group))
    return df.iloc[trainval.index[train_idx]], df.iloc[trainval.index[val_idx]], df.iloc[test_idx]


def train(data_path=DATA):
    df = load_dataset(data_path)
    source_rows, removed_rows = df.attrs["source_rows"], df.attrs["removed_rows"]
    if df.group.nunique() < 5:
        raise ValueError("Need at least 5 distinct problem_id groups to make train/validation/test splits without problem leakage.")
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame([extract_features(code) for code in df.code])], axis=1)
    train_df, val_df, test_df = split_groups(df)
    for name, subset in [("training", train_df), ("validation", val_df), ("test", test_df)]:
        if subset.label.nunique() != 2:
            raise ValueError(f"The {name} split needs both labels. Review problem groups or collect more data.")
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, min_samples_leaf=2, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=160, min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1),
    }
    fitted, validation = {}, {}
    for name, estimator in candidates.items():
        pipeline = make_pipeline(estimator)
        pipeline.fit(train_df[["code"] + STRUCTURAL_FEATURES], train_df.label)
        fitted[name] = pipeline
        validation[name] = metric_row(val_df.label, pipeline.predict(val_df[["code"] + STRUCTURAL_FEATURES]))
    selected = max(validation, key=lambda name: (validation[name]["f1"], validation[name]["accuracy"]))
    # Freeze the selected estimator and evaluate once on held-out problems.
    best = fitted[selected]
    test_metrics = {name: metric_row(test_df.label, pipe.predict(test_df[["code"] + STRUCTURAL_FEATURES])) for name, pipe in fitted.items()}
    predictions = best.predict(test_df[["code"] + STRUCTURAL_FEATURES])
    cm = confusion_matrix(test_df.label, predictions, labels=["human", "ai"]).tolist()
    importance = feature_importance(best)
    MODELS.mkdir(exist_ok=True)
    metadata = {"schema_version": 2, "python": platform.python_version(), "scikit_learn": sklearn.__version__, "joblib": joblib.__version__, "dataset_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest()}
    joblib.dump({"pipeline": best, "model_name": selected, "features": STRUCTURAL_FEATURES, "metadata": metadata}, MODELS / "codetrace.joblib")
    result = {
        "artifact_metadata": metadata,
        "selected_model": selected, "selection_metric": "validation F1 (AI as positive class)",
        "validation_metrics": validation, "models": test_metrics, "test_metrics": test_metrics[selected],
        "confusion_matrix": cm, "confusion_matrix_labels": ["human", "ai"], "feature_importance": importance,
        "dataset": {"source_rows": source_rows, "rows": len(df), "removed_missing_invalid_blank_or_duplicate_rows": removed_rows, "after_validation_and_duplicate_removal": len(df), "class_distribution": df.label.value_counts().to_dict(), "problem_groups": int(df.group.nunique()), "split_sizes": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)}, "source": Path(data_path).name},
        "limitations": "Demonstration data is synthetic and metrics do not indicate real-world detection performance." if Path(data_path).name == "samples.csv" else "Metrics describe only this dataset and split; they do not prove authorship or generalize automatically.",
    }
    (MODELS / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def feature_importance(pipeline):
    names = pipeline.named_steps["features"].get_feature_names_out()
    clf = pipeline.named_steps["classifier"]
    values = clf.feature_importances_ if hasattr(clf, "feature_importances_") else abs(clf.coef_[0])
    pairs = sorted(zip(names, values), key=lambda p: float(p[1]), reverse=True)[:12]
    return [{"feature": str(name).replace("lexical__", "word: ").replace("structure__", "structure: "), "importance": round(float(value), 5)} for name, value in pairs]


if __name__ == "__main__":
    result = train(Path(sys.argv[1]) if len(sys.argv) > 1 else DATA)
    print(json.dumps(result, indent=2))
