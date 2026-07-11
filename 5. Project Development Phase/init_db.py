"""
init_db.py

SQLite relational database initializer for VBCUA.

Creates all 10 tables with explicit primary keys, foreign keys, and
UNIQUE constraints as specified in the project architecture. Safe to
re-run: uses CREATE TABLE IF NOT EXISTS and enables foreign_keys pragma
on every connection.

Usage:
    python init_db.py
    python init_db.py --db-path assets/vbcua.db --reset
"""

import argparse
import logging
import sqlite3
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("assets") / "vbcua.db"

# Order matters: parents before children, to satisfy FK constraints
# during table creation and to make DROP order (reversed) safe.
SCHEMA_STATEMENTS: list[str] = [
    """
    CREATE TABLE IF NOT EXISTS user (
        user_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT NOT NULL,
        email       TEXT NOT NULL UNIQUE,
        role        TEXT NOT NULL DEFAULT 'learner',
        created_at  TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS reference_concept (
        ref_concept_id  INTEGER PRIMARY KEY AUTOINCREMENT,
        concept_title   TEXT NOT NULL,
        concept_text    TEXT NOT NULL,
        created_at      TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS audio_file (
        audio_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id      INTEGER NOT NULL,
        file_name    TEXT NOT NULL,
        file_path    TEXT NOT NULL,
        duration_sec REAL,
        uploaded_at  TEXT NOT NULL DEFAULT (datetime('now')),
        status       TEXT NOT NULL DEFAULT 'uploaded'
                     CHECK (status IN ('uploaded', 'processing', 'processed', 'failed')),
        FOREIGN KEY (user_id) REFERENCES user (user_id)
            ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS transcript (
        transcript_id  INTEGER PRIMARY KEY AUTOINCREMENT,
        audio_id       INTEGER NOT NULL UNIQUE,
        transcript_text TEXT NOT NULL,
        created_at     TEXT NOT NULL DEFAULT (datetime('now')),
        FOREIGN KEY (audio_id) REFERENCES audio_file (audio_id)
            ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS audio_feature (
        feature_id         INTEGER PRIMARY KEY AUTOINCREMENT,
        audio_id           INTEGER NOT NULL UNIQUE,
        pause_ratio        REAL NOT NULL,
        rms_energy         REAL NOT NULL,
        zero_crossing_rate REAL NOT NULL,
        duration_sec       REAL NOT NULL,
        FOREIGN KEY (audio_id) REFERENCES audio_file (audio_id)
            ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS filler_word_stats (
        filler_id          INTEGER PRIMARY KEY AUTOINCREMENT,
        transcript_id      INTEGER NOT NULL UNIQUE,
        filler_word_count  INTEGER NOT NULL DEFAULT 0,
        total_words        INTEGER NOT NULL DEFAULT 0,
        filler_ratio       REAL NOT NULL DEFAULT 0.0,
        FOREIGN KEY (transcript_id) REFERENCES transcript (transcript_id)
            ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS semantic_similarity (
        similarity_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        transcript_id    INTEGER NOT NULL,
        ref_concept_id   INTEGER NOT NULL,
        similarity_score REAL NOT NULL,
        FOREIGN KEY (transcript_id) REFERENCES transcript (transcript_id)
            ON DELETE CASCADE,
        FOREIGN KEY (ref_concept_id) REFERENCES reference_concept (ref_concept_id)
            ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS evaluation_result (
        result_id           INTEGER PRIMARY KEY AUTOINCREMENT,
        audio_id             INTEGER NOT NULL UNIQUE,
        ref_concept_id       INTEGER NOT NULL,
        overall_score        INTEGER NOT NULL CHECK (overall_score BETWEEN 0 AND 100),
        understanding_level  TEXT NOT NULL
                             CHECK (understanding_level IN (
                                 'Strong Understanding',
                                 'Moderate Understanding',
                                 'Poor Understanding'
                             )),
        notes                TEXT,
        FOREIGN KEY (audio_id) REFERENCES audio_file (audio_id)
            ON DELETE CASCADE,
        FOREIGN KEY (ref_concept_id) REFERENCES reference_concept (ref_concept_id)
            ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS report (
        report_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        result_id     INTEGER NOT NULL UNIQUE,
        pdf_path      TEXT NOT NULL,
        generated_at  TEXT NOT NULL DEFAULT (datetime('now')),
        file_size_kb  REAL,
        FOREIGN KEY (result_id) REFERENCES evaluation_result (result_id)
            ON DELETE CASCADE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS session (
        session_id  INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id     INTEGER NOT NULL,
        started_at  TEXT NOT NULL DEFAULT (datetime('now')),
        ended_at    TEXT,
        status      TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'ended', 'expired')),
        FOREIGN KEY (user_id) REFERENCES user (user_id)
            ON DELETE CASCADE
    );
    """,
]

# Reverse creation order = safe drop order (children before parents)
TABLE_NAMES_CREATION_ORDER: list[str] = [
    "user",
    "reference_concept",
    "audio_file",
    "transcript",
    "audio_feature",
    "filler_word_stats",
    "semantic_similarity",
    "evaluation_result",
    "report",
    "session",
]


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Open a connection with foreign_keys enforcement turned on.

    SQLite disables FK enforcement by default per-connection, so every
    module that opens its own connection to this DB must also set this
    pragma, not just init_db.py.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def reset_schema(conn: sqlite3.Connection) -> None:
    """Drop all known tables in child-first order. Destructive — used only
    when --reset is explicitly passed."""
    logger.warning("Resetting schema: dropping all VBCUA tables.")
    conn.execute("PRAGMA foreign_keys = OFF;")
    for table in reversed(TABLE_NAMES_CREATION_ORDER):
        conn.execute(f"DROP TABLE IF EXISTS {table};")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.commit()


def create_schema(conn: sqlite3.Connection) -> None:
    """Create all 10 tables. Idempotent via IF NOT EXISTS."""
    try:
        for statement in SCHEMA_STATEMENTS:
            conn.execute(statement)
        conn.commit()
        logger.info("Schema created/verified: %d tables.", len(TABLE_NAMES_CREATION_ORDER))
    except sqlite3.Error:
        conn.rollback()
        logger.exception("Failed to create schema; transaction rolled back.")
        raise


def verify_schema(conn: sqlite3.Connection) -> bool:
    """Sanity check: confirm all expected tables exist after creation."""
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    )
    existing = {row[0] for row in cursor.fetchall()}
    missing = set(TABLE_NAMES_CREATION_ORDER) - existing
    if missing:
        logger.error("Missing tables after init: %s", missing)
        return False
    logger.info("Schema verified: all %d tables present.", len(TABLE_NAMES_CREATION_ORDER))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize the VBCUA SQLite database.")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite DB file (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop all existing VBCUA tables before recreating them (destructive).",
    )
    args = parser.parse_args()

    conn = get_connection(args.db_path)
    try:
        if args.reset:
            reset_schema(conn)
        create_schema(conn)
        ok = verify_schema(conn)
        if not ok:
            raise RuntimeError("Schema verification failed after initialization.")
        logger.info("VBCUA database ready at: %s", args.db_path.resolve())
    finally:
        conn.close()


if __name__ == "__main__":
    main()
