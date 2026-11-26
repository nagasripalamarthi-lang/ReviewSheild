"""Automated unit and integration tests for ReviewShield."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import BULK_SAMPLE_PATH, LABEL_GENUINE, LABEL_SUSPICIOUS, SAMPLE_DATASET_PATH
from src.database import (
    clear_analysis_history,
    init_db,
    load_analysis_history,
    save_analysis_record,
    save_bulk_records,
)
from src.duplicate_detector import analyze_duplicates_and_patterns
from src.ml_model import (
    load_and_validate_dataset,
    predict_batch,
    predict_single_review,
    train_and_save_model,
)
from src.nlp_preprocessing import clean_text, detect_linguistic_signals, extract_stylometric_features


def test_clean_text_and_stylometric_features() -> None:
    raw = "<b>BEST PRODUCT EVER!!!!</b> Visit https://example.com for promo code!!"
    cleaned = clean_text(raw)
    assert "<b>" not in cleaned
    assert "urltoken" in cleaned

    feats = extract_stylometric_features(raw)
    assert feats["exclamation_count"] == 6.0
    assert feats["url_or_contact_flag"] == 1.0
    assert feats["promo_spam_hits"] >= 2.0
    assert feats["punctuation_burst"] >= 2.0

    signals = detect_linguistic_signals(raw)
    assert any(s["severity"] == "warning" for s in signals)


def test_dataset_validation_and_training(tmp_path: Path) -> None:
    df, audit = load_and_validate_dataset(SAMPLE_DATASET_PATH)
    assert len(df) >= 180
    assert audit["genuine_count"] > 0
    assert audit["suspicious_count"] > 0

    model_path = tmp_path / "model.joblib"
    meta_path = tmp_path / "meta.json"
    pipeline, metadata = train_and_save_model(
        dataset_path=SAMPLE_DATASET_PATH,
        model_path=model_path,
        metadata_path=meta_path,
    )
    assert model_path.exists()
    assert meta_path.exists()
    assert metadata["primary_metrics"]["f1_score"] >= 0.85
    assert len(metadata["model_comparison"]) == 4

    suspicious_res = predict_single_review(
        "BEST PRODUCT EVER!!!! 100% GENUINE MUST BUY NOW!! DM me on telegram for cashback!!",
        pipeline=pipeline,
    )
    assert suspicious_res["prediction"] == LABEL_SUSPICIOUS
    assert suspicious_res["confidence"] >= 0.65
    assert suspicious_res["explanation_summary"]

    genuine_res = predict_single_review(
        "Used this charger on a week-long trip. Charges my laptop at 65W and gets slightly warm, though the included cable is a bit short.",
        pipeline=pipeline,
    )
    assert genuine_res["prediction"] == LABEL_GENUINE
    assert genuine_res["confidence"] >= 0.60


def test_bulk_prediction_and_duplicate_detection(tmp_path: Path) -> None:
    model_path = tmp_path / "model.joblib"
    meta_path = tmp_path / "meta.json"
    pipeline, _ = train_and_save_model(
        dataset_path=SAMPLE_DATASET_PATH,
        model_path=model_path,
        metadata_path=meta_path,
    )

    bulk_df = pd.read_csv(BULK_SAMPLE_PATH)
    analyzed = predict_batch(bulk_df, text_column="review_text", pipeline=pipeline)
    assert len(analyzed) == len(bulk_df)
    assert set(analyzed["prediction"]).issubset({LABEL_GENUINE, LABEL_SUSPICIOUS})
    assert (analyzed["prediction"] == LABEL_SUSPICIOUS).any()
    assert (analyzed["prediction"] == LABEL_GENUINE).any()

    dup_result = analyze_duplicates_and_patterns(
        bulk_df["review_text"].tolist(),
        similarity_threshold=0.80,
    )
    assert dup_result.summary["exact_duplicate_groups"] >= 1
    assert dup_result.summary["near_duplicate_pairs"] >= 2
    assert dup_result.summary["repeated_phrases_count"] >= 1


def test_sqlite_persistence(tmp_path: Path) -> None:
    db_path = tmp_path / "test_reviewshield.db"
    init_db(db_path)

    row_id = save_analysis_record(
        review_text="Battery lasts around 6 hours and charges via USB-C.",
        prediction=LABEL_GENUINE,
        suspicious_probability=0.14,
        confidence=0.86,
        explanation_summary="Classified as Potentially Genuine.",
        product_name="Test Earbuds",
        rating=4,
        db_path=db_path,
    )
    assert row_id == 1

    batch_df = pd.DataFrame(
        [
            {
                "product_name": "Test Adapter",
                "rating": 5,
                "review_text": "BEST CHARGER EVER BUY NOW!!!!",
                "prediction": LABEL_SUSPICIOUS,
                "suspicious_probability": 0.94,
                "confidence": 0.94,
                "explanation_summary": "Flagged as Potentially Suspicious.",
            }
        ]
    )
    inserted = save_bulk_records(batch_df, db_path=db_path)
    assert inserted == 1

    history = load_analysis_history(db_path=db_path)
    assert len(history) == 2

    clear_analysis_history(db_path=db_path)
    assert load_analysis_history(db_path=db_path).empty
