"""
modules/semantic_eval.py

Sentence-BERT semantic similarity engine for VBCUA.

Compares a student's spoken transcript against a reference concept using
'all-MiniLM-L6-v2' embeddings and cosine similarity. This module returns
a RAW similarity ratio only — scoring_engine.py owns the point-threshold
conversion (>0.7 / >0.4 / else), so this file must never apply those
cutoffs itself.
"""

import functools
import logging

from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class SemanticEvalError(ValueError):
    """Raised for invalid inputs (non-string text) — not for ordinary empty text,
    which is handled as a valid 0.0-similarity case, not an error."""


@functools.lru_cache(maxsize=2)
def _load_sbert_model(model_name: str) -> SentenceTransformer:
    """
    Loads and caches a SentenceTransformer model by name. lru_cache ensures
    the model weights are loaded into memory only once per model_name per
    process, mirroring the pattern used for the Whisper model in
    speech_to_text.py.
    """
    logger.info("Loading Sentence-BERT model '%s'... this may take a moment.", model_name)
    model = SentenceTransformer(model_name)
    logger.info("Sentence-BERT model '%s' loaded successfully.", model_name)
    return model


class SemanticEvaluator:
    """
    Thin wrapper around a cached Sentence-BERT model for computing
    transcript-to-reference-concept semantic similarity.

    Usage:
        evaluator = SemanticEvaluator()
        score = evaluator.compute_similarity(transcript_text, reference_text)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None  # lazy-loaded

    def _ensure_model_loaded(self) -> None:
        if self._model is None:
            self._model = _load_sbert_model(self.model_name)

    def compute_similarity(self, student_transcript: str, reference_concept: str) -> float:
        """
        Computes cosine similarity between the student's transcript and the
        reference concept text.

        Args:
            student_transcript: The transcribed spoken explanation.
            reference_concept: The target concept_text to compare against.

        Returns:
            A float in [0.0, 1.0]. Returns 0.0 immediately (no model
            inference run) if either input is None or entirely whitespace —
            there is nothing meaningful to embed, and this avoids wasting a
            forward pass on empty strings.

        Raises:
            SemanticEvalError: if either argument is a non-string type,
                which indicates an upstream bug (e.g. passing None from a
                failed transcription without checking first) rather than a
                legitimate "no speech" case.
        """
        if student_transcript is not None and not isinstance(student_transcript, str):
            raise SemanticEvalError(
                f"student_transcript must be a str, got {type(student_transcript).__name__}."
            )
        if reference_concept is not None and not isinstance(reference_concept, str):
            raise SemanticEvalError(
                f"reference_concept must be a str, got {type(reference_concept).__name__}."
            )

        transcript_stripped = (student_transcript or "").strip()
        reference_stripped = (reference_concept or "").strip()

        if not transcript_stripped or not reference_stripped:
            logger.warning(
                "Empty/whitespace-only text detected (transcript_empty=%s, reference_empty=%s); "
                "skipping model inference and returning similarity=0.0.",
                not transcript_stripped, not reference_stripped,
            )
            return 0.0

        try:
            self._ensure_model_loaded()
            embeddings = self._model.encode(
                [transcript_stripped, reference_stripped],
                convert_to_tensor=True,
            )
            raw_score = util.cos_sim(embeddings[0], embeddings[1]).item()
        except Exception:
            logger.exception(
                "Semantic similarity inference failed; returning 0.0 as a safe fallback."
            )
            return 0.0

        # Cosine similarity is mathematically in [-1, 1]; sentence embeddings
        # rarely go negative for related text, but we clamp defensively so
        # downstream scoring_engine thresholds (which assume [0, 1]) never
        # see an out-of-range value.
        similarity = max(0.0, min(1.0, raw_score))

        logger.info(
            "Computed semantic similarity: %.4f (raw=%.4f)", similarity, raw_score
        )
        return similarity
