"""
test_pipeline.py

Automated unit + integration testing suite for VBCUA.

Covers:
    1. scoring_engine.py boundary correctness (the exact >/< threshold cases)
    2. scoring_engine.py input validation (ScoringError on bad data)
    3. init_db.py schema creation against an isolated temp database
       (never touches the real assets/vbcua.db)
    4. An audio_utils -> scoring_engine integration pass using a synthetic
       sine-wave clip, with speech_to_text / semantic_eval mocked out so
       this suite doesn't require downloading Whisper/SBERT model weights
       just to verify pipeline wiring.

Run with:
    python test_pipeline.py
"""

import logging
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# Ensure project root is importable regardless of the working directory
# this script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import init_db  # noqa: E402
from modules import scoring_engine  # noqa: E402

# audio_utils pulls in librosa/soundfile/matplotlib, which are heavier
# optional dependencies — degrade gracefully (skip, don't fail) if they
# aren't installed in the environment running this suite.
try:
    from modules import audio_utils
    _AUDIO_LIBS_AVAILABLE = True
except ImportError as _audio_import_error:
    _AUDIO_LIBS_AVAILABLE = False
    _AUDIO_IMPORT_ERROR = _audio_import_error

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

logging.disable(logging.CRITICAL)  # keep test output focused on pass/fail, not module logs


# ==========================================================================
# 1. Scoring matrix boundary tests
# ==========================================================================
class TestScoringEngineBoundaries(unittest.TestCase):
    """Verifies the strict '>' / '<' operators — values sitting exactly on
    a threshold must fall to the LOWER-point branch, never the higher one."""

    def test_semantic_similarity_exactly_0_7_falls_to_30pt_branch(self):
        # 0.7 fails ">0.7" but passes ">0.4" -> 30 points, NOT 50.
        points = scoring_engine.score_semantic_similarity(0.7)
        self.assertEqual(points, 30, "similarity=0.7 must score 30pts, not 50pts")

    def test_semantic_similarity_just_above_0_7_gets_50pt_branch(self):
        points = scoring_engine.score_semantic_similarity(0.7000001)
        self.assertEqual(points, 50)

    def test_filler_ratio_exactly_0_05_falls_to_10pt_branch(self):
        # 0.05 fails "<0.05" -> 10 points, NOT 20.
        points = scoring_engine.score_filler_fluency(0.05)
        self.assertEqual(points, 10, "filler_ratio=0.05 must score 10pts, not 20pts")

    def test_filler_ratio_just_below_0_05_gets_20pt_branch(self):
        points = scoring_engine.score_filler_fluency(0.0499999)
        self.assertEqual(points, 20)

    def test_pause_ratio_exactly_0_25_falls_to_5pt_branch(self):
        # 0.25 fails "<0.25" -> 5 points, NOT 15.
        points = scoring_engine.score_pause_ratio(0.25)
        self.assertEqual(points, 5, "pause_ratio=0.25 must score 5pts, not 15pts")

    def test_pause_ratio_just_below_0_25_gets_15pt_branch(self):
        points = scoring_engine.score_pause_ratio(0.2499999)
        self.assertEqual(points, 15)

    def test_rms_energy_exactly_0_01_falls_to_5pt_branch(self):
        # 0.01 fails ">0.01" -> 5 points, NOT 15.
        points = scoring_engine.score_rms_energy(0.01)
        self.assertEqual(points, 5, "rms_energy=0.01 must score 5pts, not 15pts")

    def test_rms_energy_just_above_0_01_gets_15pt_branch(self):
        points = scoring_engine.score_rms_energy(0.0100001)
        self.assertEqual(points, 15)

    def test_full_boundary_combo_via_evaluate_understanding(self):
        """All four boundary values at once, through the public entry point."""
        result = scoring_engine.evaluate_understanding(
            similarity=0.7,
            filler_ratio=0.05,
            audio={"pause_ratio": 0.25, "rms_energy": 0.01},
        )
        expected_total = 30 + 10 + 5 + 5  # = 50
        self.assertEqual(result.overall_score, expected_total)
        self.assertEqual(result.understanding_level, "Moderate Understanding")
        self.assertEqual(result.tier_color, "#f39c12")

    def test_tier_classification_boundaries(self):
        self.assertEqual(scoring_engine.classify_tier(80), "Strong Understanding")
        self.assertEqual(scoring_engine.classify_tier(79), "Moderate Understanding")
        self.assertEqual(scoring_engine.classify_tier(50), "Moderate Understanding")
        self.assertEqual(scoring_engine.classify_tier(49), "Poor Understanding")


# ==========================================================================
# 2. Input validation tests
# ==========================================================================
class TestScoringEngineValidation(unittest.TestCase):
    """Confirms malformed metrics raise ScoringError instead of silently
    producing a wrong score or crashing with an unrelated exception."""

    def test_negative_similarity_raises_scoring_error(self):
        with self.assertRaises(scoring_engine.ScoringError):
            scoring_engine.score_semantic_similarity(-0.1)

    def test_negative_filler_ratio_raises_scoring_error(self):
        with self.assertRaises(scoring_engine.ScoringError):
            scoring_engine.score_filler_fluency(-0.01)

    def test_negative_pause_ratio_raises_scoring_error(self):
        with self.assertRaises(scoring_engine.ScoringError):
            scoring_engine.score_pause_ratio(-0.5)

    def test_negative_rms_energy_raises_scoring_error(self):
        with self.assertRaises(scoring_engine.ScoringError):
            scoring_engine.score_rms_energy(-0.001)

    def test_non_numeric_string_raises_scoring_error(self):
        with self.assertRaises(scoring_engine.ScoringError):
            scoring_engine.score_semantic_similarity("high")  # type: ignore[arg-type]

    def test_none_value_raises_scoring_error(self):
        with self.assertRaises(scoring_engine.ScoringError):
            scoring_engine.score_pause_ratio(None)  # type: ignore[arg-type]

    def test_boolean_raises_scoring_error(self):
        # bool is technically an int subclass in Python; explicitly rejected
        # so a stray True/False doesn't silently score as 1/0.
        with self.assertRaises(scoring_engine.ScoringError):
            scoring_engine.score_rms_energy(True)  # type: ignore[arg-type]

    def test_nan_raises_scoring_error(self):
        with self.assertRaises(scoring_engine.ScoringError):
            scoring_engine.score_semantic_similarity(math.nan)

    def test_evaluate_understanding_propagates_scoring_error(self):
        with self.assertRaises(scoring_engine.ScoringError):
            scoring_engine.evaluate_understanding(
                similarity=0.5,
                filler_ratio=0.1,
                audio={"pause_ratio": -0.2, "rms_energy": 0.02},
            )

    def test_evaluate_understanding_missing_audio_key_raises(self):
        with self.assertRaises(scoring_engine.ScoringError):
            scoring_engine.evaluate_understanding(
                similarity=0.5, filler_ratio=0.1, audio={"pause_ratio": 0.1}
            )

    def test_evaluate_understanding_non_dict_audio_raises(self):
        with self.assertRaises(scoring_engine.ScoringError):
            scoring_engine.evaluate_understanding(
                similarity=0.5, filler_ratio=0.1, audio="not-a-dict"  # type: ignore[arg-type]
            )


class TestFillerRatioCalculation(unittest.TestCase):
    """Sanity checks for the text-tokenization filler ratio helper."""

    def test_known_filler_ratio(self):
        text = "So, um, basically the mitochondria is like the powerhouse, uh."
        ratio, filler_count, total_words = scoring_engine.calculate_filler_ratio(text)
        # tokens: so, um, basically, the, mitochondria, is, like, the,
        #         powerhouse, uh  -> 10 words, 5 fillers (so, um, basically, like, uh)
        self.assertEqual(total_words, 10)
        self.assertEqual(filler_count, 5)
        self.assertAlmostEqual(ratio, 0.5)

    def test_empty_transcript_returns_zero_ratio(self):
        ratio, filler_count, total_words = scoring_engine.calculate_filler_ratio("")
        self.assertEqual((ratio, filler_count, total_words), (0.0, 0, 0))

    def test_none_transcript_raises_scoring_error(self):
        with self.assertRaises(scoring_engine.ScoringError):
            scoring_engine.calculate_filler_ratio(None)  # type: ignore[arg-type]

    def test_filler_words_are_whole_word_matched(self):
        # "ahead" contains "ah" as a substring but must NOT count as a filler.
        text = "Walking ahead of schedule with no issues so far."
        ratio, filler_count, total_words = scoring_engine.calculate_filler_ratio(text)
        self.assertEqual(filler_count, 1)  # only "so" counts


# ==========================================================================
# 3. Isolated database schema tests (never touches assets/vbcua.db)
# ==========================================================================
class TestDatabaseSchemaIsolated(unittest.TestCase):
    """Uses a throwaway temp-directory SQLite file for every test so the
    production database is never opened, modified, or reset by this suite."""

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp_dir.name) / "test_vbcua.db"
        self.conn = init_db.get_connection(self.db_path)

    def tearDown(self):
        self.conn.close()
        self._tmp_dir.cleanup()

    def test_schema_creates_all_ten_tables(self):
        init_db.create_schema(self.conn)
        self.assertTrue(init_db.verify_schema(self.conn))

        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        table_names = {row[0] for row in cursor.fetchall()}
        self.assertEqual(table_names, set(init_db.TABLE_NAMES_CREATION_ORDER))

    def test_schema_creation_is_idempotent(self):
        init_db.create_schema(self.conn)
        init_db.create_schema(self.conn)  # must not raise on second call
        self.assertTrue(init_db.verify_schema(self.conn))

    def test_foreign_key_enforcement_blocks_orphan_row(self):
        init_db.create_schema(self.conn)
        with self.assertRaises(Exception):
            # audio_file.user_id references a user_id that doesn't exist
            self.conn.execute(
                "INSERT INTO audio_file (user_id, file_name, file_path) "
                "VALUES (9999, 'clip.wav', '/tmp/clip.wav');"
            )
            self.conn.commit()

    def test_insert_and_read_round_trip(self):
        init_db.create_schema(self.conn)
        self.conn.execute(
            "INSERT INTO user (name, email, role) VALUES (?, ?, ?);",
            ("Test User", "test.user@example.com", "learner"),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT name, email FROM user WHERE email = ?;", ("test.user@example.com",)
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "Test User")

    def test_reset_schema_drops_tables(self):
        init_db.create_schema(self.conn)
        init_db.reset_schema(self.conn)
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        self.assertEqual(cursor.fetchall(), [])


# ==========================================================================
# 4. Pipeline integration test (synthetic audio + mocked STT/SBERT)
# ==========================================================================
@unittest.skipUnless(
    _AUDIO_LIBS_AVAILABLE and _NUMPY_AVAILABLE,
    "librosa/soundfile/numpy not installed in this environment — "
    "install requirements.txt to run the audio integration test.",
)
class TestAudioToScoringIntegration(unittest.TestCase):
    """
    Simulates a full pipeline pass on a synthetic sine-wave clip:
        audio_utils.load_audio_signal -> audio_utils.extract_raw_audio_metrics
            -> scoring_engine.evaluate_understanding

    speech_to_text and semantic_eval are mocked out here because loading
    real Whisper/SBERT model weights is unnecessary overhead for a test
    whose job is verifying that data flows correctly between modules —
    not re-validating third-party model correctness.
    """

    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.audio_path = Path(self._tmp_dir.name) / "synthetic_clip.wav"
        self._write_synthetic_wav(self.audio_path, duration_sec=2.0, sr=16000)

    def tearDown(self):
        self._tmp_dir.cleanup()

    @staticmethod
    def _write_synthetic_wav(path: Path, duration_sec: float, sr: int) -> None:
        import soundfile as sf

        t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
        # 220Hz tone with a slowly varying envelope so RMS/ZCR/pause_ratio
        # calculations have realistic non-trivial values to chew on.
        envelope = 0.5 * (1 + np.sin(2 * np.pi * 0.5 * t))
        signal = 0.3 * envelope * np.sin(2 * np.pi * 220 * t)
        sf.write(str(path), signal.astype(np.float32), sr)

    def test_audio_utils_produces_valid_metrics_dict(self):
        y, sr = audio_utils.load_audio_signal(self.audio_path)
        metrics = audio_utils.extract_raw_audio_metrics(y, sr)

        for key in ("pause_ratio", "rms_energy", "zero_crossing_rate", "duration_sec"):
            self.assertIn(key, metrics)
            self.assertIsInstance(metrics[key], float)

        self.assertGreaterEqual(metrics["pause_ratio"], 0.0)
        self.assertLessEqual(metrics["pause_ratio"], 1.0)
        self.assertGreater(metrics["rms_energy"], 0.0)
        self.assertAlmostEqual(metrics["duration_sec"], 2.0, delta=0.05)

    def test_full_mocked_pipeline_pass_end_to_end(self):
        """Audio -> real metrics; STT/SBERT mocked; -> real scoring. No
        unhandled exceptions should propagate anywhere in this chain."""
        with mock.patch(
            "modules.speech_to_text.SpeechToTextEngine.transcribe",
            return_value="Machine learning models learn patterns directly from data.",
        ), mock.patch(
            "modules.semantic_eval.SemanticEvaluator.compute_similarity",
            return_value=0.83,
        ):
            from modules import semantic_eval, speech_to_text

            y, sr = audio_utils.load_audio_signal(self.audio_path)
            raw_metrics = audio_utils.extract_raw_audio_metrics(y, sr)

            stt_engine = speech_to_text.SpeechToTextEngine(model_size="base")
            transcript = stt_engine.transcribe(self.audio_path)

            evaluator = semantic_eval.SemanticEvaluator()
            similarity = evaluator.compute_similarity(transcript, "reference text")

            filler_ratio, _, _ = scoring_engine.calculate_filler_ratio(transcript)

            result = scoring_engine.evaluate_understanding(
                similarity=similarity,
                filler_ratio=filler_ratio,
                audio={
                    "pause_ratio": raw_metrics["pause_ratio"],
                    "rms_energy": raw_metrics["rms_energy"],
                },
            )

            self.assertIsInstance(result.overall_score, int)
            self.assertIn(
                result.understanding_level,
                ("Strong Understanding", "Moderate Understanding", "Poor Understanding"),
            )
            self.assertEqual(result.semantic_points, 50)  # similarity=0.83 -> >0.7 branch

    def test_short_audio_raises_audio_load_error(self):
        short_path = Path(self._tmp_dir.name) / "too_short.wav"
        self._write_synthetic_wav(short_path, duration_sec=0.3, sr=16000)
        with self.assertRaises(audio_utils.AudioLoadError):
            audio_utils.load_audio_signal(short_path)

    def test_silent_audio_raises_audio_load_error(self):
        import soundfile as sf

        silent_path = Path(self._tmp_dir.name) / "silent.wav"
        silence = np.zeros(16000 * 2, dtype=np.float32)
        sf.write(str(silent_path), silence, 16000)
        with self.assertRaises(audio_utils.AudioLoadError):
            audio_utils.load_audio_signal(silent_path)

    def test_missing_file_raises_audio_load_error(self):
        with self.assertRaises(audio_utils.AudioLoadError):
            audio_utils.load_audio_signal(Path(self._tmp_dir.name) / "does_not_exist.wav")


# ==========================================================================
# 5. NLP Analysis & db_manager Unit Tests
# ==========================================================================
class TestNLPAnalyzer(unittest.TestCase):
    """Verifies text lexical diversity and VADER sentiment analyzer."""

    def test_nlp_metrics_empty(self):
        from modules import nlp_utils
        analyzer = nlp_utils.NLPAnalyzer()
        metrics = analyzer.analyze_text("")
        self.assertEqual(metrics["sentence_count"], 0)
        self.assertEqual(metrics["lexical_diversity"], 0.0)

    def test_nlp_metrics_simple(self):
        from modules import nlp_utils
        analyzer = nlp_utils.NLPAnalyzer()
        text = "This is a simple test sentence. It has clear structures."
        metrics = analyzer.analyze_text(text)
        self.assertEqual(metrics["sentence_count"], 2)
        self.assertGreater(metrics["lexical_diversity"], 0.0)
        self.assertIn(metrics["sentiment"], ("Confident / Positive", "Neutral", "Hesitant / Negative"))


class TestDBManagerOperations(unittest.TestCase):
    """Verifies database manager saving/retrieval functions on a temporary DB."""

    def setUp(self):
        from modules import db_manager
        self.tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.tmp_db.name)
        # Initialize schema
        conn = db_manager.get_connection(self.db_path)
        init_db.create_schema(conn)
        conn.close()

    def tearDown(self):
        try:
            Path(self.tmp_db.name).unlink(missing_ok=True)
        except OSError:
            pass

    def test_user_and_concept_persistence(self):
        from modules import db_manager
        # Save user
        u_id = db_manager.save_user("Test Name", "test@vbcua.org", "learner", db_path=self.db_path)
        self.assertIsInstance(u_id, int)
        
        # Save concept
        c_id = db_manager.save_reference_concept("AI Title", "AI explanation concept", db_path=self.db_path)
        self.assertIsInstance(c_id, int)
        
        # Fetch users
        users = db_manager.get_all_users(db_path=self.db_path)
        self.assertEqual(len(users), 1)
        self.assertEqual(users[0]["name"], "Test Name")


# ==========================================================================
# Main execution block — clean pass/fail summary + non-zero exit on failure
# ==========================================================================
def _build_suite() -> unittest.TestSuite:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestScoringEngineBoundaries))
    suite.addTests(loader.loadTestsFromTestCase(TestScoringEngineValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestFillerRatioCalculation))
    suite.addTests(loader.loadTestsFromTestCase(TestDatabaseSchemaIsolated))
    suite.addTests(loader.loadTestsFromTestCase(TestAudioToScoringIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestNLPAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestDBManagerOperations))
    return suite


if __name__ == "__main__":
    if not _AUDIO_LIBS_AVAILABLE:
        print(
            f"[NOTICE] Skipping audio integration tests — "
            f"import failed: {_AUDIO_IMPORT_ERROR}\n"
            f"         Run 'pip install -r requirements.txt' to enable them.\n"
        )

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(_build_suite())

    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    passed = total - failures - errors - skipped

    print("\n" + "=" * 60)
    print("VBCUA TEST PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Total tests run : {total}")
    print(f"  Passed          : {passed}")
    print(f"  Failed          : {failures}")
    print(f"  Errors          : {errors}")
    print(f"  Skipped         : {skipped}")
    print("=" * 60)

    if failures or errors:
        print("RESULT: FAILURE — see details above.")
        sys.exit(1)
    else:
        print("RESULT: ALL TESTS PASSED.")
        sys.exit(0)
