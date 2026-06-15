"""SQLite storage for settings (API keys) and OCR results."""

import contextlib
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def init() -> None:
    with contextlib.closing(_conn()) as c, c:
        c.execute(
            "CREATE TABLE IF NOT EXISTS settings ("
            " key TEXT PRIMARY KEY,"
            " value TEXT NOT NULL)"
        )
        c.execute(
            "CREATE TABLE IF NOT EXISTS results ("
            " id TEXT PRIMARY KEY,"
            " filename TEXT,"
            " image_path TEXT NOT NULL,"
            " ocr_json TEXT,"
            " full_text TEXT,"
            " refined_text TEXT,"
            " created_at TEXT NOT NULL DEFAULT (datetime('now')))"
        )


def get_settings() -> dict[str, str]:
    with contextlib.closing(_conn()) as c:
        rows = c.execute("SELECT key, value FROM settings").fetchall()
    return {r["key"]: r["value"] for r in rows}


def set_settings(values: dict[str, str]) -> None:
    with contextlib.closing(_conn()) as c, c:
        c.executemany(
            "INSERT INTO settings (key, value) VALUES (?, ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            list(values.items()),
        )


def create_result(rid: str, filename: str | None, image_path: str) -> None:
    with contextlib.closing(_conn()) as c, c:
        c.execute(
            "INSERT INTO results (id, filename, image_path) VALUES (?, ?, ?)",
            (rid, filename, image_path),
        )


def get_result(rid: str) -> sqlite3.Row | None:
    with contextlib.closing(_conn()) as c:
        return c.execute("SELECT * FROM results WHERE id = ?", (rid,)).fetchone()


def save_ocr(rid: str, ocr_json: str, full_text: str) -> None:
    with contextlib.closing(_conn()) as c, c:
        c.execute(
            "UPDATE results SET ocr_json = ?, full_text = ? WHERE id = ?",
            (ocr_json, full_text, rid),
        )


def save_refined(rid: str, text: str) -> None:
    with contextlib.closing(_conn()) as c, c:
        c.execute("UPDATE results SET refined_text = ? WHERE id = ?", (text, rid))
