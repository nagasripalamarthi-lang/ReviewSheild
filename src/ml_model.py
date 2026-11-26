"""ML training, evaluation, persistence, and explainable inference for ReviewShield."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from src.config import (
    LABEL_GENUINE,
    LABEL_SUSPICIOUS,
    MODEL_ARTIFACT_PATH,
    MODEL_METADATA_PATH,
    SAMPLE_DATASET_PATH,
    SUSPICIOUS_THRESHOLD,
    TOP_EXPLANATION_FEATURES,
)
from src.nlp_preprocessing import (
    STYLOMETRIC_DISPLAY_NAMES,
    StylometricTransformer,
    clean_text,
    detect_linguistic_signals,
    extract_stylometric_features,
)


def _build_feature_union(non_negative_stylometric: bool = False) -> FeatureUnion:
    """Construct combined word TF-IDF, char TF-IDF, and scaled stylometric features."""
    scaler = MinMaxScaler() if non_negative_stylometric else StandardScaler()
    return FeatureUnion(
        transformer_list=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    ngram_range=(1, 2),
                    max_features=2500,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=2000,
                    min_df=3,
                    sublinear_tf=True,
                ),
            ),
            (
                "stylometric",
                Pipeline([
                    ("extract", StylometricTransformer()),
                    ("scale", scaler),
                ]),
            ),
        ]
    )


def load_and_validate_dataset(dataset_path: Path = SAMPLE_DATASET_PATH) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load dataset, inspect missing/duplicate rows, and return cleaned DataFrame + audit stats."""
    raw_df = pd.read_csv(dataset_path)
    initial_rows = int(len(raw_df))
    missing_text = int(raw_df["review_text"].isna().sum()) if "review_text" in raw_df.columns else initial_rows
    missing_label = int(raw_df["label"].isna().sum()) if "label" in raw_df.columns else initial_rows

    df = raw_df.dropna(subset=["review_text", "label"]).copy()
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"].str.len() > 0].copy()
    df["label_norm"] = df["label"].astype(str).str.strip().str.lower()
    valid_labels = {"genuine": 0, "suspicious": 1}
    df = df[df["label_norm"].isin(valid_labels)].copy()
    df["target"] = df["label_norm"].map(valid_labels)

    audit = {
        "initial_rows": initial_rows,
        "valid_rows": int(len(df)),
        "dropped_missing_text": missing_text,
        "dropped_missing_label": missing_label,
        "genuine_count": int((df["target"] == 0).sum()),
        "suspicious_count": int((df["target"] == 1).sum()),
        "categories": sorted(df["product_category"].dropna().unique().tolist())
        if "product_category" in df.columns
        else [],
        "data_origin": str(df["data_origin"].iloc[0])
        if "data_origin" in df.columns and not df.empty
        else "synthetic_benchmark",
    }
    return df.reset_index(drop=True), audit


def _evaluate_candidate(
    name: str,
    pipeline: Pipeline,
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Fit candidate pipeline on train split and compute CV + held-out test metrics."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_prob = pipeline.decision_function(X_test)

    cm = confusion_matrix(y_test, y_pred).tolist()
    return {
        "model_name": name,
        "cv_f1_mean": round(float(np.mean(cv_f1)), 4),
        "cv_f1_std": round(float(np.std(cv_f1)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "confusion_matrix": cm,
        "fitted_pipeline": pipeline,
    }


def train_and_save_model(
    dataset_path: Path = SAMPLE_DATASET_PATH,
    model_path: Path = MODEL_ARTIFACT_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
) -> tuple[Pipeline, dict[str, Any]]:
    """Train candidate models on a leak-free stratified split, save primary pipeline and metrics."""
    df, audit = load_and_validate_dataset(dataset_path)

    # Strict featurization ordering: split BEFORE fitting any TF-IDF or scaler
    X_train, X_test, y_train, y_test = train_test_split(
        df["review_text"],
        df["target"],
        test_size=0.20,
        stratify=df["target"],
        random_state=42,
    )

    candidates = [
        (
            "Logistic Regression (TF-IDF + Stylometric)",
            Pipeline([
                ("features", _build_feature_union(non_negative_stylometric=False)),
                ("clf", LogisticRegression(C=1.5, max_iter=1000, class_weight="balanced", random_state=42)),
            ]),
        ),
        (
            "Linear SVM (Calibrated)",
            Pipeline([
                ("features", _build_feature_union(non_negative_stylometric=False)),
                (
                    "clf",
                    CalibratedClassifierCV(
                        LinearSVC(C=1.0, class_weight="balanced", random_state=42),
                        cv=3,
                    ),
                ),
            ]),
        ),
        (
            "Random Forest",
            Pipeline([
                ("features", _build_feature_union(non_negative_stylometric=False)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=18,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]),
        ),
        (
            "Complement Naive Bayes",
            Pipeline([
                ("features", _build_feature_union(non_negative_stylometric=True)),
                ("clf", ComplementNB(alpha=0.5)),
            ]),
        ),
    ]

    results = [
        _evaluate_candidate(name, pipe, X_train, y_train, X_test, y_test)
        for name, pipe in candidates
    ]

    # Select Logistic Regression for direct calibrated probability + feature-level explainability
    # while reporting full benchmark comparisons transparently.
    primary_result = results[0]
    primary_pipeline: Pipeline = primary_result["fitted_pipeline"]

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(primary_pipeline, model_path)

    comparison_table = [
        {k: v for k, v in res.items() if k != "fitted_pipeline"}
        for res in results
    ]

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "selected_model": primary_result["model_name"],
        "selection_rationale": (
            "Logistic Regression achieves strong F1/ROC-AUC alongside calibrated probabilities "
            "and exact linear token/stylometric attribution for transparent explanations."
        ),
        "dataset_audit": audit,
        "split_sizes": {
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "test_fraction": 0.20,
        },
        "primary_metrics": {k: v for k, v in primary_result.items() if k not in {"fitted_pipeline", "model_name"}},
        "model_comparison": comparison_table,
        "disclaimer": (
            "Predictions reflect statistical patterns learned from a synthetic & curated educational benchmark "
            "dataset and do NOT constitute verified proof of fake or fraudulent activity."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return primary_pipeline, metadata


def load_or_train_model(
    model_path: Path = MODEL_ARTIFACT_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
) -> tuple[Pipeline, dict[str, Any]]:
    """Load persisted model pipeline and metadata, training automatically if missing."""
    if model_path.exists() and metadata_path.exists():
        try:
            pipeline = joblib.load(model_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return pipeline, metadata
        except Exception:
            pass
    return train_and_save_model(model_path=model_path, metadata_path=metadata_path)


def _format_feature_label(raw_name: str) -> str:
    """Convert scikit-learn FeatureUnion feature name into a readable label."""
    if raw_name.startswith("word_tfidf__"):
        token = raw_name.removeprefix("word_tfidf__")
        return f"Phrase '{token}'"
    if raw_name.startswith("char_tfidf__"):
        gram = raw_name.removeprefix("char_tfidf__")
        return f"Char pattern '{gram}'"
    if raw_name.startswith("stylometric__"):
        col = raw_name.removeprefix("stylometric__")
        return STYLOMETRIC_DISPLAY_NAMES.get(col, col)
    return raw_name


def _explain_linear_contributions(
    pipeline: Pipeline,
    review_text: str,
    top_k: int = TOP_EXPLANATION_FEATURES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute per-feature linear log-odds contributions for word & stylometric features."""
    features_union: FeatureUnion = pipeline.named_steps["features"]
    clf: LogisticRegression = pipeline.named_steps["clf"]

    X_vec = features_union.transform([review_text])
    feature_names = features_union.get_feature_names_out()
    coefs = clf.coef_[0]

    row_dense = X_vec.toarray()[0]
    contributions = row_dense * coefs

    suspicious_drivers: list[dict[str, Any]] = []
    genuine_drivers: list[dict[str, Any]] = []

    # Prioritize human-interpretable word_tfidf and stylometric features in explanations
    indexed = []
    for idx, contrib in enumerate(contributions):
        if abs(contrib) < 1e-4:
            continue
        fname = str(feature_names[idx])
        if fname.startswith("char_tfidf__"):
            continue
        indexed.append((fname, float(contrib)))

    indexed.sort(key=lambda item: abs(item[1]), reverse=True)

    for fname, contrib in indexed:
        entry = {
            "feature": _format_feature_label(fname),
            "raw_feature": fname,
            "contribution": round(contrib, 4),
        }
        if contrib > 0 and len(suspicious_drivers) < top_k:
            suspicious_drivers.append(entry)
        elif contrib < 0 and len(genuine_drivers) < top_k:
            genuine_drivers.append(entry)

    return suspicious_drivers, genuine_drivers


def _build_explanation_summary(
    prediction: str,
    suspicious_prob: float,
    suspicious_drivers: list[dict[str, Any]],
    genuine_drivers: list[dict[str, Any]],
    signals: list[dict[str, str]],
) -> str:
    """Compose a concise natural-language summary explaining why the model chose its label."""
    if prediction == LABEL_SUSPICIOUS:
        warning_titles = [s["title"].lower() for s in signals if s["severity"] == "warning"]
        driver_names = [d["feature"] for d in suspicious_drivers[:3]]
        parts = [f"Flagged as {LABEL_SUSPICIOUS} ({suspicious_prob * 100:.1f}% suspicion score)."]
        if warning_titles:
            parts.append(f"Key traits: {', '.join(warning_titles[:2])}.")
        if driver_names:
            parts.append(f"Top model drivers: {', '.join(driver_names)}.")
        return " ".join(parts)

    positive_titles = [s["title"].lower() for s in signals if s["severity"] == "positive"]
    driver_names = [d["feature"] for d in genuine_drivers[:3]]
    parts = [f"Classified as {LABEL_GENUINE} ({(1.0 - suspicious_prob) * 100:.1f}% genuine confidence)."]
    if positive_titles:
        parts.append(f"Key traits: {', '.join(positive_titles[:2])}.")
    if driver_names:
        parts.append(f"Supporting signals: {', '.join(driver_names)}.")
    return " ".join(parts)


def predict_single_review(
    review_text: str,
    pipeline: Pipeline | None = None,
    threshold: float = SUSPICIOUS_THRESHOLD,
) -> dict[str, Any]:
    """Analyze a single review and return label, confidence, drivers, and linguistic signals."""
    if pipeline is None:
        pipeline, _ = load_or_train_model()

    cleaned_input = "" if review_text is None else str(review_text).strip()
    if not cleaned_input:
        raise ValueError("Review text cannot be empty.")

    probs = pipeline.predict_proba([cleaned_input])[0]
    genuine_prob = float(probs[0])
    suspicious_prob = float(probs[1])
    prediction = LABEL_SUSPICIOUS if suspicious_prob >= threshold else LABEL_GENUINE
    confidence = suspicious_prob if prediction == LABEL_SUSPICIOUS else genuine_prob

    suspicious_drivers, genuine_drivers = _explain_linear_contributions(pipeline, cleaned_input)
    signals = detect_linguistic_signals(cleaned_input)
    stylometric = extract_stylometric_features(cleaned_input)
    summary = _build_explanation_summary(
        prediction=prediction,
        suspicious_prob=suspicious_prob,
        suspicious_drivers=suspicious_drivers,
        genuine_drivers=genuine_drivers,
        signals=signals,
    )

    return {
        "prediction": prediction,
        "suspicious_probability": suspicious_prob,
        "genuine_probability": genuine_prob,
        "confidence": confidence,
        "top_suspicious_drivers": suspicious_drivers,
        "top_genuine_drivers": genuine_drivers,
        "signals": signals,
        "stylometric_features": stylometric,
        "explanation_summary": summary,
    }


def predict_batch(
    df: pd.DataFrame,
    text_column: str = "review_text",
    pipeline: Pipeline | None = None,
    threshold: float = SUSPICIOUS_THRESHOLD,
) -> pd.DataFrame:
    """Run batch inference and generate concise explanations for a DataFrame of reviews."""
    if pipeline is None:
        pipeline, _ = load_or_train_model()

    work_df = df.copy()
    work_df[text_column] = work_df[text_column].fillna("").astype(str).str.strip()
    work_df = work_df[work_df[text_column].str.len() > 0].reset_index(drop=True)
    if work_df.empty:
        return work_df

    probs = pipeline.predict_proba(work_df[text_column].tolist())
    suspicious_probs = probs[:, 1].astype(float)
    genuine_probs = probs[:, 0].astype(float)

    predictions = np.where(suspicious_probs >= threshold, LABEL_SUSPICIOUS, LABEL_GENUINE)
    confidences = np.where(predictions == LABEL_SUSPICIOUS, suspicious_probs, genuine_probs)

    summaries = []
    word_counts = []
    exclamation_counts = []
    promo_hits = []

    for text, pred, s_prob in zip(work_df[text_column], predictions, suspicious_probs):
        susp_drv, gen_drv = _explain_linear_contributions(pipeline, text, top_k=3)
        sigs = detect_linguistic_signals(text)
        feats = extract_stylometric_features(text)
        summaries.append(_build_explanation_summary(str(pred), float(s_prob), susp_drv, gen_drv, sigs))
        word_counts.append(int(feats["word_count"]))
        exclamation_counts.append(int(feats["exclamation_count"]))
        promo_hits.append(int(feats["promo_spam_hits"]))

    work_df["prediction"] = predictions
    work_df["suspicious_probability"] = np.round(suspicious_probs, 4)
    work_df["confidence"] = np.round(confidences, 4)
    work_df["confidence_pct"] = np.round(confidences * 100.0, 1)
    work_df["word_count"] = word_counts
    work_df["exclamation_count"] = exclamation_counts
    work_df["promo_cue_hits"] = promo_hits
    work_df["explanation_summary"] = summaries

    if text_column != "review_text":
        work_df["review_text"] = work_df[text_column]

    return work_df
