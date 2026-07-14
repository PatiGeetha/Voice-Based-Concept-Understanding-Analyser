"""
modules/db_manager.py

SQLite Database Persistence Manager for VBCUA.
Implements standard CRUD operations for all 10 tables matching the ER diagram,
with foreign key enforcement on every connection.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("assets") / "vbcua.db"


def get_connection(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Opens a connection to the SQLite database and enforces foreign keys."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    # Return rows as dict-like objects for easier parsing
    conn.row_factory = sqlite3.Row
    return conn


def save_user(name: str, email: str, role: str = "learner", db_path: Path = DEFAULT_DB_PATH) -> int:
    """Saves or retrieves a user by email to maintain uniqueness."""
    with get_connection(db_path) as conn:
        try:
            cursor = conn.execute(
                "INSERT INTO user (name, email, role) VALUES (?, ?, ?)",
                (name, email, role),
            )
            conn.commit()
            return cursor.lastrowid
        except sqlite3.IntegrityError:
            # User already exists, retrieve and return ID
            cursor = conn.execute("SELECT user_id FROM user WHERE email = ?", (email,))
            row = cursor.fetchone()
            if row:
                return row["user_id"]
            raise


def get_user_by_email(email: str, db_path: Path = DEFAULT_DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieves a user profile by email address."""
    with get_connection(db_path) as conn:
        cursor = conn.execute("SELECT * FROM user WHERE email = ?", (email,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_all_users(db_path: Path = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves all users."""
    with get_connection(db_path) as conn:
        cursor = conn.execute("SELECT * FROM user ORDER BY name ASC")
        return [dict(row) for row in cursor.fetchall()]


def save_reference_concept(title: str, text: str, db_path: Path = DEFAULT_DB_PATH) -> int:
    """Saves or updates a reference concept."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "SELECT ref_concept_id FROM reference_concept WHERE concept_title = ?", (title,)
        )
        row = cursor.fetchone()
        if row:
            conn.execute(
                "UPDATE reference_concept SET concept_text = ? WHERE ref_concept_id = ?",
                (text, row["ref_concept_id"]),
            )
            conn.commit()
            return row["ref_concept_id"]
        else:
            cursor = conn.execute(
                "INSERT INTO reference_concept (concept_title, concept_text) VALUES (?, ?)",
                (title, text),
            )
            conn.commit()
            return cursor.lastrowid


def get_all_reference_concepts(db_path: Path = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves all reference concepts."""
    with get_connection(db_path) as conn:
        cursor = conn.execute("SELECT * FROM reference_concept ORDER BY concept_title ASC")
        return [dict(row) for row in cursor.fetchall()]


def save_audio_file(
    user_id: int, file_name: str, file_path: str, duration_sec: float, db_path: Path = DEFAULT_DB_PATH
) -> int:
    """Logs a new audio file entry."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO audio_file (user_id, file_name, file_path, duration_sec, status) VALUES (?, ?, ?, ?, 'processed')",
            (user_id, file_name, file_path, duration_sec),
        )
        conn.commit()
        return cursor.lastrowid


def save_transcript(audio_id: int, transcript_text: str, db_path: Path = DEFAULT_DB_PATH) -> int:
    """Saves the speech-to-text transcript."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO transcript (audio_id, transcript_text) VALUES (?, ?)",
            (audio_id, transcript_text),
        )
        conn.commit()
        return cursor.lastrowid


def save_audio_features(
    audio_id: int,
    pause_ratio: float,
    rms_energy: float,
    zero_crossing_rate: float,
    duration_sec: float,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    """Saves the extracted signal features."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO audio_feature (audio_id, pause_ratio, rms_energy, zero_crossing_rate, duration_sec)
            VALUES (?, ?, ?, ?, ?)
            """,
            (audio_id, pause_ratio, rms_energy, zero_crossing_rate, duration_sec),
        )
        conn.commit()
        return cursor.lastrowid


def save_filler_word_stats(
    transcript_id: int, filler_count: int, total_words: int, filler_ratio: float, db_path: Path = DEFAULT_DB_PATH
) -> int:
    """Saves filler word statistics."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO filler_word_stats (transcript_id, filler_word_count, total_words, filler_ratio) VALUES (?, ?, ?, ?)",
            (transcript_id, filler_count, total_words, filler_ratio),
        )
        conn.commit()
        return cursor.lastrowid


def save_semantic_similarity(
    transcript_id: int, ref_concept_id: int, similarity_score: float, db_path: Path = DEFAULT_DB_PATH
) -> int:
    """Saves SBERT semantic similarity score."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO semantic_similarity (transcript_id, ref_concept_id, similarity_score) VALUES (?, ?, ?)",
            (transcript_id, ref_concept_id, similarity_score),
        )
        conn.commit()
        return cursor.lastrowid


def save_evaluation_result(
    audio_id: int,
    ref_concept_id: int,
    overall_score: int,
    understanding_level: str,
    notes: str,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    """Saves the overall evaluation result."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO evaluation_result (audio_id, ref_concept_id, overall_score, understanding_level, notes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (audio_id, ref_concept_id, overall_score, understanding_level, notes),
        )
        conn.commit()
        return cursor.lastrowid


def save_report(result_id: int, pdf_path: str, file_size_kb: float, db_path: Path = DEFAULT_DB_PATH) -> int:
    """Saves the compiled PDF report path."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO report (result_id, pdf_path, file_size_kb) VALUES (?, ?, ?)",
            (result_id, pdf_path, file_size_kb),
        )
        conn.commit()
        return cursor.lastrowid


def save_session(user_id: int, status: str = "active", db_path: Path = DEFAULT_DB_PATH) -> int:
    """Saves session info."""
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO session (user_id, status) VALUES (?, ?)",
            (user_id, status),
        )
        conn.commit()
        return cursor.lastrowid


def get_evaluation_history(db_path: Path = DEFAULT_DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves previous evaluations with user, concept details and reports."""
    with get_connection(db_path) as conn:
        query = """
            SELECT 
                er.result_id,
                er.overall_score,
                er.understanding_level,
                er.notes,
                rc.concept_title,
                rc.concept_text,
                af.file_name,
                af.file_path,
                af.uploaded_at,
                t.transcript_text,
                rep.pdf_path,
                u.name as user_name,
                u.email as user_email
            FROM evaluation_result er
            JOIN audio_file af ON er.audio_id = af.audio_id
            JOIN reference_concept rc ON er.ref_concept_id = rc.ref_concept_id
            JOIN transcript t ON af.audio_id = t.audio_id
            JOIN user u ON af.user_id = u.user_id
            LEFT JOIN report rep ON er.result_id = rep.result_id
            ORDER BY af.uploaded_at DESC
        """
        cursor = conn.execute(query)
        return [dict(row) for row in cursor.fetchall()]
