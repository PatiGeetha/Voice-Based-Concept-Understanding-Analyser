# VBCUA — Voice-Based Concept Understanding Analyser

An AI-powered Streamlit application that evaluates how effectively a user
explains a conceptual topic through spoken communication. Upload an audio
clip, and VBCUA transcribes it, compares it semantically against a
reference concept, scores fluency and delivery, and generates a
downloadable PDF report.

---

## Project Structure

```
vbcua_project/
│
├── .streamlit/
│   └── config.toml        # maxUploadSize + dark theme branding
├── assets/                # Local cache for audio, waveforms, and generated PDFs
├── modules/
│   ├── __init__.py
│   ├── audio_utils.py     # Librosa/SoundFile signal loading + feature extraction
│   ├── db_manager.py      # SQLite database persistence operations
│   ├── gemini_feedback.py # Google Gemini qualitative feedback generator
│   ├── nlp_utils.py       # NLTK tokenization & sentiment polarity analyzer
│   ├── speech_to_text.py  # OpenAI Whisper transcription wrapper
│   ├── semantic_eval.py   # Sentence-BERT ('all-MiniLM-L6-v2') cosine similarity
│   ├── scoring_engine.py  # Multi-criteria points matrix + tier classification
│   └── report_generator.py# ReportLab PDF report compiler
│
├── app.py                 # Streamlit dashboard (UI + frontend)
├── server.py              # FastAPI service backend
├── init_db.py              # SQLite schema initializer (10 tables, FK-enforced)
├── test_pipeline.py        # Automated unit + integration test suite
└── requirements.txt
```

---

## 1. Environment Setup

**Prerequisite:** Python 3.10 or higher, and `ffmpeg` installed on your system
(required by `openai-whisper` for audio decoding).

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg

# Windows (using Chocolatey)
choco install ffmpeg
```

Create and activate a virtual environment:

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the Whisper `base` model and the
> `all-MiniLM-L6-v2` Sentence-BERT model automatically — this requires an
> internet connection the first time only. Subsequent runs use the local
> cache (`~/.cache/whisper` and `~/.cache/torch/sentence_transformers`).

---

## 2. Initialize the Database

VBCUA persists structured evaluation history across 10 relational tables
(user, audio_file, transcript, audio_feature, filler_word_stats,
semantic_similarity, evaluation_result, report, session, and
reference_concept). Build the schema before first use:

```bash
python init_db.py
```

This creates `assets/vbcua.db` with foreign keys enforced. To wipe and
rebuild the schema from scratch (destructive — deletes all existing data):

```bash
python init_db.py --reset
```

---

## 3. Run the Automated Test Suite

Before launching the app, verify the scoring engine's boundary math and
core pipeline wiring are intact:

```bash
python test_pipeline.py
```

This runs in an isolated temporary SQLite database (your real
`assets/vbcua.db` is never touched) and validates:
- Exact-threshold boundary cases in `scoring_engine.py` (e.g. similarity
  at precisely `0.7`, filler ratio at precisely `0.05`, etc.)
- `ScoringError` is raised correctly for invalid/negative/NaN inputs
- The 10-table schema creates and enforces foreign keys correctly
- An end-to-end audio → scoring pipeline pass using a synthetic test clip

A clean pass/fail summary prints at the end, with a non-zero exit code on
any failure (useful for CI pipelines).

---

## 4. Launch the Application

VBCUA can run either in dual-server mode (FastAPI API backend + Streamlit frontend) or single-process standalone fallback mode (Streamlit queries local python modules directly).

### Startup backend API:
```bash
uvicorn server:app --reload --port 8000
```

### Startup frontend UI:
```bash
streamlit run app.py
```

This opens the dashboard at `http://localhost:8501` by default. From
there:
1. Upload a `.wav` or `.mp3` file (up to 200MB, per `.streamlit/config.toml`).
2. Review or edit the reference concept text on the right.
3. Click **Analyze Concept Understanding**.
4. Review your transcript, composite score, waveform, and metric
   breakdown, then download the generated PDF report.

---

## Troubleshooting

| Symptom | Likely Cause |
|---|---|
| `FileNotFoundError` during transcription | `ffmpeg` not installed or not on `PATH` |
| First analysis run is very slow | Whisper/SBERT models downloading for the first time |
| `AudioLoadError: Audio too short` | Clip is under 1.0 second — record a longer sample |
| `AudioLoadError: ... completely silent` | Check microphone input level before recording |
| Upload rejected above a certain size | Confirm `.streamlit/config.toml` is present in the project root (not just `assets/`) |
