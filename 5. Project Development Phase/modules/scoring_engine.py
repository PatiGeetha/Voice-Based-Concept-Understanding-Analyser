"""
modules/scoring_engine.py

Single source of truth for VBCUA's multi-criteria scoring matrix.
All point-value thresholds live here — audio_utils.py and semantic_eval.py
must only ever return raw metrics (ratios/energies), never point values,
so this module stays the sole place the grading rules can drift.

Public entry point: evaluate_understanding(similarity, filler_ratio, audio)
"""

import logging
import re
from dataclasses import dataclass
from typing import Literal, TypedDict

logger = logging.getLogger(__name__)

UnderstandingLevel = Literal[
    "Strong Understanding", "Moderate Understanding", "Poor Understanding"
]

TIER_COLORS: dict[UnderstandingLevel, str] = {
    "Strong Understanding": "#2ecc71",
    "Moderate Understanding": "#f39c12",
    "Poor Understanding": "#e74c3c",
}

# 50 (semantic) + 20 (filler) + 15 (pause) + 15 (energy)
MAX_POSSIBLE_SCORE = 100

FILLER_WORDS: frozenset[str] = frozenset({"um", "uh", "like", "ah", "so", "basically"})

_WORD_RE = re.compile(r"[a-zA-Z']+")


class ScoringError(ValueError):
    """Raised when a scoring input is missing, malformed, non-numeric, or out of range."""


class AudioMetrics(TypedDict):
    """Shape expected for the `audio` dict passed to evaluate_understanding()."""
    pause_ratio: float
    rms_energy: float


@dataclass(frozen=True)
class ScoreBreakdown:
    semantic_points: int
    filler_points: int
    pause_points: int
    energy_points: int
    overall_score: int
    understanding_level: UnderstandingLevel
    tier_color: str

    def as_dict(self) -> dict:
        """Convenience for st.session_state caching and DB row construction."""
        return {
            "semantic_points": self.semantic_points,
            "filler_points": self.filler_points,
            "pause_points": self.pause_points,
            "energy_points": self.energy_points,
            "overall_score": self.overall_score,
            "understanding_level": self.understanding_level,
            "tier_color": self.tier_color,
        }


def _validate_ratio(name: str, value: float) -> None:
    if value is None:
        raise ScoringError(f"{name} is missing (None).")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScoringError(f"{name} must be numeric, got {type(value).__name__}.")
    if isinstance(value, float) and value != value:  # NaN check, no math import needed
        raise ScoringError(f"{name} is NaN.")
    if value < 0:
        raise ScoringError(f"{name} cannot be negative (got {value}).")


def _validate_audio_dict(audio: dict) -> None:
    if not isinstance(audio, dict):
        raise ScoringError(f"audio must be a dict, got {type(audio).__name__}.")
    for key in ("pause_ratio", "rms_energy"):
        if key not in audio:
            raise ScoringError(f"audio dict is missing required key '{key}'.")
        _validate_ratio(f"audio['{key}']", audio[key])


# --------------------------------------------------------------------------
# Filler ratio calculation (text tokenization -> ratio)
# --------------------------------------------------------------------------
def calculate_filler_ratio(transcript_text: str) -> tuple[float, int, int]:
    """
    Tokenizes transcript_text and computes the filler-word ratio.

    Returns (filler_ratio, filler_word_count, total_words) so callers can
    persist all three columns required by filler_word_stats without a
    second pass over the text.

    Matching is case-insensitive and whole-word only (regex \\b-style via
    tokenization), so 'liked' or 'ahead' do not falsely count as fillers.
    """
    if transcript_text is None:
        raise ScoringError("transcript_text is missing (None).")
    if not isinstance(transcript_text, str):
        raise ScoringError(f"transcript_text must be str, got {type(transcript_text).__name__}.")

    tokens = [tok.lower() for tok in _WORD_RE.findall(transcript_text)]
    total_words = len(tokens)

    if total_words == 0:
        logger.warning("Empty or non-alphabetic transcript passed to calculate_filler_ratio.")
        return 0.0, 0, 0

    filler_word_count = sum(1 for tok in tokens if tok in FILLER_WORDS)
    filler_ratio = filler_word_count / total_words

    logger.info(
        "Filler ratio computed: %d/%d = %.4f", filler_word_count, total_words, filler_ratio
    )
    return filler_ratio, filler_word_count, total_words


# --------------------------------------------------------------------------
# Step 1: Semantic Similarity (Sentence-BERT cosine match)
# --------------------------------------------------------------------------
def score_semantic_similarity(similarity: float) -> int:
    _validate_ratio("similarity", similarity)
    if similarity > 0.7:
        return 50
    elif similarity > 0.4:
        return 30
    return 10


# --------------------------------------------------------------------------
# Step 2: Filler Word Fluency
# --------------------------------------------------------------------------
def score_filler_fluency(filler_ratio: float) -> int:
    _validate_ratio("filler_ratio", filler_ratio)
    return 20 if filler_ratio < 0.05 else 10


# --------------------------------------------------------------------------
# Step 3: Pause Ratio (silence frames below 10% max RMS)
# --------------------------------------------------------------------------
def score_pause_ratio(pause_ratio: float) -> int:
    _validate_ratio("pause_ratio", pause_ratio)
    return 15 if pause_ratio < 0.25 else 5


# --------------------------------------------------------------------------
# Step 4: RMS Energy (loudness baseline)
# --------------------------------------------------------------------------
def score_rms_energy(rms_energy: float) -> int:
    _validate_ratio("rms_energy", rms_energy)
    return 15 if rms_energy > 0.01 else 5


def classify_tier(overall_score: int) -> UnderstandingLevel:
    if overall_score >= 80:
        return "Strong Understanding"
    elif overall_score >= 50:
        return "Moderate Understanding"
    return "Poor Understanding"


def evaluate_understanding(similarity: float, filler_ratio: float, audio: AudioMetrics) -> ScoreBreakdown:
    """
    Primary scoring entry point matching the required signature:
        evaluate_understanding(similarity, filler_ratio, audio)

    Args:
        similarity: Sentence-BERT cosine similarity score (0.0-1.0 typical).
        filler_ratio: Precomputed filler-word ratio (use calculate_filler_ratio()
                       upstream if you only have raw transcript text).
        audio: dict containing 'pause_ratio' and 'rms_energy' keys, e.g.
               {"pause_ratio": 0.18, "rms_energy": 0.014}

    Returns:
        ScoreBreakdown with per-component points, overall_score (0-100),
        understanding_level, and tier_color.

    Raises:
        ScoringError: on missing/non-numeric/negative inputs or a malformed
                      audio dict. Callers (main.py) should catch this and
                      show a Streamlit warning rather than writing a
                      corrupt evaluation_result row.
    """
    try:
        _validate_audio_dict(audio)
        semantic_points = score_semantic_similarity(similarity)
        filler_points = score_filler_fluency(filler_ratio)
        pause_points = score_pause_ratio(audio["pause_ratio"])
        energy_points = score_rms_energy(audio["rms_energy"])
    except ScoringError:
        logger.exception(
            "evaluate_understanding failed. Inputs: similarity=%r filler_ratio=%r audio=%r",
            similarity, filler_ratio, audio,
        )
        raise

    overall_score = semantic_points + filler_points + pause_points + energy_points
    understanding_level = classify_tier(overall_score)
    tier_color = TIER_COLORS[understanding_level]

    logger.info(
        "Score computed: semantic=%d filler=%d pause=%d energy=%d total=%d/%d level=%s",
        semantic_points, filler_points, pause_points, energy_points,
        overall_score, MAX_POSSIBLE_SCORE, understanding_level,
    )

    return ScoreBreakdown(
        semantic_points=semantic_points,
        filler_points=filler_points,
        pause_points=pause_points,
        energy_points=energy_points,
        overall_score=overall_score,
        understanding_level=understanding_level,
        tier_color=tier_color,
    )

