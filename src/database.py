"""SQLite persistence layer for ReviewShield analysis logs and dashboard insights."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DB_PATH


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Create a SQLite connection ensuring parent directories exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """Initialize SQLite tables for single/bulk review analyses."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyzed_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL,
                product_name TEXT,
                rating REAL,
                review_text TEXT NOT NULL,
                prediction TEXT NOT NULL,
                suspicious_probability REAL NOT NULL,
                confidence REAL NOT NULL,
                explanation_summary TEXT,
                signals_json TEXT
            )
            """
        )
        conn.commit()


def save_analysis_record(
    review_text: str,
    prediction: str,
    suspicious_probability: float,
    confidence: float,
    explanation_summary: str,
    signals: list[dict[str, Any]] | None = None,
    source: str = "single_analysis",
    product_name: str | None = None,
    rating: float | None = None,
    db_path: Path = DB_PATH,
) -> int:
    """Insert a single analyzed review record into SQLite."""
    init_db(db_path)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO analyzed_reviews (
                created_at, source, product_name, rating, review_text,
                prediction, suspicious_probability, confidence,
                explanation_summary, signals_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                source,
                product_name or "",
                rating,
                review_text,
                prediction,
                float(suspicious_probability),
                float(confidence),
                explanation_summary,
                json.dumps(signals or []),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def save_bulk_records(
    df: pd.DataFrame,
    source: str = "bulk_csv",
    db_path: Path = DB_PATH,
) -> int:
    """Persist bulk analysis results DataFrame into SQLite."""
    if df.empty:
        return 0
    init_db(db_path)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    rows = []
    for _, row in df.iterrows():
        rows.append(
            (
                now,
                source,
                str(row.get("product_name", "") or ""),
                float(row["rating"]) if pd.notna(row.get("rating")) else None,
                str(row["review_text"]),
                str(row["prediction"]),
                float(row["suspicious_probability"]),
                float(row["confidence"]),
                str(row.get("explanation_summary", "") or ""),
                "[]",
            )
        )
    with get_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO analyzed_reviews (
                created_at, source, product_name, rating, review_text,
                prediction, suspicious_probability, confidence,
                explanation_summary, signals_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def load_analysis_history(
    limit: int | None = 500,
    db_path: Path = DB_PATH,
) -> pd.DataFrame:
    """Load saved review analysis history ordered newest first."""
    init_db(db_path)
    query = "SELECT * FROM analyzed_reviews ORDER BY id DESC"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    with get_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn)
    return df


def clear_analysis_history(db_path: Path = DB_PATH) -> None:
    """Remove all saved analysis history records."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM analyzed_reviews")
        conn.commit()
