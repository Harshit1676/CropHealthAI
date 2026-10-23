"""
SQLite database layer for CropHealth AI.

Uses the standard-library sqlite3 module directly (no ORM) to keep the
project simple, with a small connection-per-call pattern that is safe for
Streamlit + FastAPI running as separate local processes.
"""
from __future__ import annotations

import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = config.get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    crop TEXT NOT NULL,
    prediction TEXT NOT NULL,
    confidence REAL NOT NULL,
    risk_score REAL,
    risk_level TEXT,
    temperature REAL,
    humidity REAL,
    rainfall REAL,
    location TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_crop ON predictions(crop);
"""


@contextmanager
def get_connection(db_path: Path = config.DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path = config.DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)
    logger.info("Database initialized at %s", db_path)


def insert_prediction(
    crop: str,
    prediction: str,
    confidence: float,
    risk_score: Optional[float] = None,
    risk_level: Optional[str] = None,
    temperature: Optional[float] = None,
    humidity: Optional[float] = None,
    rainfall: Optional[float] = None,
    location: Optional[str] = None,
    db_path: Path = config.DB_PATH,
) -> int:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """INSERT INTO predictions
               (timestamp, crop, prediction, confidence, risk_score, risk_level,
                temperature, humidity, rainfall, location)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                crop, prediction, confidence, risk_score, risk_level,
                temperature, humidity, rainfall, location,
            ),
        )
        return cur.lastrowid


def get_predictions(limit: int = 100, offset: int = 0,
                     db_path: Path = config.DB_PATH) -> list[dict]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_predictions(db_path: Path = config.DB_PATH) -> list[dict]:
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM predictions ORDER BY id ASC").fetchall()
        return [dict(r) for r in rows]


def count_predictions(db_path: Path = config.DB_PATH) -> int:
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()
        return int(row["c"])


def delete_all_predictions(db_path: Path = config.DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM predictions")
