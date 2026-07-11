"""
modules/speech_to_text.py

OpenAI Whisper transcription wrapper for VBCUA.

Design note on caching: this module intentionally does NOT import
streamlit, so it stays testable from test_pipeline.py and reusable
outside the app (e.g. batch scripts). Model weights are cached via
functools.lru_cache at the module level, which already prevents reloads
across repeated SpeechToTextEngine() instantiations within one Python
process/Streamlit server. If you want Streamlit's own resource-cache
semantics (e.g. explicit cache-clearing from the UI), wrap the engine
constructor in main.py like this:

    @st.cache_resource
    def get_engine():
        return SpeechToTextEngine(model_size="base")
"""

import functools
import logging
import re
from pathlib import Path

import whisper

logger = logging.getLogger(__name__)

DEFAULT_MODEL_SIZE = "base"
_WHITESPACE_RE = re.compile(r"\s+")


class TranscriptionError(RuntimeError):
    """Raised for unrecoverable input problems (missing file, unreadable path).

    NOT raised for ordinary decode/inference failures — those are caught
    internally by transcribe() and logged, returning an empty string so a
    single bad upload cannot crash the Streamlit server thread.
    """


@functools.lru_cache(maxsize=4)
def _load_whisper_model(model_size: str):
    """
    Loads and caches a Whisper model by size. lru_cache ensures the (large)
    model weights are loaded into memory only once per model_size per process,
    regardless of how many SpeechToTextEngine instances are created.
    """
    logger.info("Loading Whisper model (size='%s')... this may take a moment.", model_size)
    model = whisper.load_model(model_size)
    logger.info("Whisper model '%s' loaded successfully.", model_size)
    return model


def _normalize_text(raw_text: str) -> str:
    """Collapses internal whitespace and strips leading/trailing spaces."""
    if not raw_text:
        return ""
    normalized = _WHITESPACE_RE.sub(" ", raw_text).strip()
    return normalized


class SpeechToTextEngine:
    """
    Thin wrapper around openai-whisper for transcribing uploaded audio.

    Usage:
        engine = SpeechToTextEngine(model_size="base")
        text = engine.transcribe("assets/uploaded_clip.wav")
    """

    def __init__(self, model_size: str = DEFAULT_MODEL_SIZE):
        self.model_size = model_size
        self._model = None  # lazy-loaded on first transcribe() call

    def _ensure_model_loaded(self) -> None:
        if self._model is None:
            self._model = _load_whisper_model(self.model_size)

    def transcribe(self, file_path: str | Path) -> str:
        """
        Transcribes the given audio file to text.

        Args:
            file_path: Path to a .wav or .mp3 file. Whisper handles its own
                internal resampling/decoding via ffmpeg, so pre-loaded numpy
                arrays are not required here.

        Returns:
            A normalized (stripped, whitespace-collapsed) transcript string.
            Returns "" if no audible speech was detected, or if transcription
            failed for a recoverable reason (unsupported codec, corrupted
            audio, empty result) — callers should treat "" as "no usable
            transcript" rather than a crash signal.

        Raises:
            TranscriptionError: only for missing/invalid file paths, i.e.
                problems the caller can fix before retrying. Runtime
                inference failures are caught internally and logged instead.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise TranscriptionError(f"Audio file not found: {file_path}")
        if not file_path.is_file():
            raise TranscriptionError(f"Path is not a file: {file_path}")

        try:
            self._ensure_model_loaded()
        except Exception as exc:
            # If the model itself can't load (e.g. corrupted cache, missing
            # ffmpeg), that's an environment problem, not a per-file one —
            # surface it rather than silently returning "".
            logger.exception("Whisper model failed to load (size='%s').", self.model_size)
            raise TranscriptionError(f"Whisper model failed to load: {exc}") from exc

        try:
            result = self._model.transcribe(str(file_path))
        except Exception as exc:
            # Covers unsupported/exotic codecs, ffmpeg decode failures,
            # truncated files, etc. Logged and swallowed so one bad upload
            # doesn't take down the Streamlit server thread.
            logger.exception(
                "Transcription failed for '%s' (treating as no usable audio).", file_path.name
            )
            return ""

        raw_text = result.get("text", "") if isinstance(result, dict) else ""
        transcript = _normalize_text(raw_text)

        if not transcript:
            logger.warning("No audible speech detected in: %s", file_path.name)

        logger.info(
            "Transcribed '%s': %d characters.", file_path.name, len(transcript)
        )
        return transcript
