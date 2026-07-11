# Implementation Plan — Voice-Based Concept Understanding Analyser (VBCUA)

This plan outlines the architecture and implementation steps to build the complete, production-grade VBCUA web application. We will integrate a FastAPI backend (`server.py`), enable full SQLite database persistence matching the ER diagram, integrate NLTK for local NLP analysis, leverage Google Gemini API for advanced qualitative conceptual feedback, and unify everything under a high-fidelity Streamlit user interface (`app.py`).

---

## User Review Required

> [!IMPORTANT]
> **Key Architecture Decisions:**
> 1. **Entrypoint Names:** The main Streamlit frontend file will be named [app.py](file:///c:/Users/HP/Desktop/Voice%20Based%20Concept%20Understanding%20Analyser/app.py) (renamed/copied from `main.py`) as per the project specification. The FastAPI backend will run separately inside [server.py](file:///c:/Users/HP/Desktop/Voice%20Based%20Concept%20Understanding%20Analyser/server.py) (on port `8000`).
> 2. **Database Logging:** Every analysis run will fully populate all 10 SQLite database tables in `assets/vbcua.db` as defined in the ER diagram (User, Audio File, Transcript, Audio Feature, Filler Stats, Semantic Similarity, Evaluation Result, Report, Session).
> 3. **Google Gemini Feedback:** We will use the Google Gemini API to analyze transcript conceptual alignment and delivery tone. We will provide a sidebar configuration in Streamlit for users to enter their `GEMINI_API_KEY`, defaulting to a local NLTK/rule-based NLP analyzer if no key is provided.
> 4. **NLTK NLP & Sentiment:** We will download NLTK data (`vader_lexicon`, `punkt`) programmatically at startup to calculate lexical diversity, sentence count, and delivery sentiment.

---

## Proposed Changes

We will group our development logically by components:

### 1. NLP & AI Extensions Component
Add local sentiment analysis, lexical metrics, and Google Gemini integration.

#### [NEW] [nlp_utils.py](file:///c:/Users/HP/Desktop/Voice%20Based%20Concept%20Understanding%20Analyser/modules/nlp_utils.py)
* Uses NLTK's `SentimentIntensityAnalyzer` (VADER) to analyze delivery tone (positive, neutral, negative/hesitant).
* Calculates lexical metrics: Type-Token Ratio (vocabulary diversity) and average sentence length.
* Programmatically downloads required NLTK resources on the first import.

#### [NEW] [gemini_feedback.py](file:///c:/Users/HP/Desktop/Voice%20Based%20Concept%20Understanding%20Analyser/modules/gemini_feedback.py)
* Uses the Google Gemini API (`google-generativeai` package) to generate structured feedback:
  - **Strengths**: Core points explained correctly.
  - **Gaps**: Important concepts or details that were missed.
  - **Improvement Areas**: Specific delivery or articulation tips.
* Implements a detailed rule-based fallback if no Gemini API Key is configured.

---

### 2. Database Management Component
Add persistence layer to populate the SQLite schema.

#### [NEW] [db_manager.py](file:///c:/Users/HP/Desktop/Voice%20Based%20Concept%20Understanding%20Analyser/modules/db_manager.py)
* Integrates database persistence using standard `sqlite3` with foreign keys enabled.
* Exposes clean functions to save users, audio metadata, transcripts, filler word counts, zero-crossing rate/pause features, similarity scores, PDF reports, and session states.
* Implements query functions to fetch session history and render previous evaluations in the dashboard.

---

### 3. FastAPI Service Component
Expose modular API endpoints for the analysis pipeline.

#### [NEW] [server.py](file:///c:/Users/HP/Desktop/Voice%20Based%20Concept%20Understanding%20Analyser/server.py)
* Built using FastAPI and `uvicorn`.
* Endpoints:
  - `POST /api/evaluate`: Ingests an audio file and reference concept, runs the modules (Whisper, SBERT, NLTK, Gemini), saves to database, and returns the full score breakdown.
  - `GET /api/history`: Returns list of previous evaluations.
  - `GET /api/report/{result_id}`: Stream/download a generated report.
  - `GET /api/concepts`: Returns predefined reference concepts.

---

### 4. High-Fidelity UI & Report Updates
Integrate API connection, feedback, and history tabs.

#### [NEW] [app.py](file:///c:/Users/HP/Desktop/Voice%20Based%20Concept%20Understanding%20Analyser/app.py)
* The primary Streamlit frontend file.
* Update UI with sidebar for Gemini API Key configuration.
* Connect the dashboard widgets to the FastAPI service endpoints (with local module fallback if FastAPI is offline).
* Add a **History Tab** allowing users to select previous runs, view detailed transcripts/waveforms, and re-download report PDFs.
* Integrate NLP sentiment metrics and Gemini conceptual feedback panels into the visual dashboard.

#### [MODIFY] [report_generator.py](file:///c:/Users/HP/Desktop/Voice%20Based%20Concept%20Understanding%20Analyser/modules/report_generator.py)
* Update PDF layout to include NLTK sentiment scores and the structured qualitative feedback (Strengths, Gaps) under the summary section.

---

### 5. Dependency Updates

#### [MODIFY] [requirements.txt](file:///c:/Users/HP/Desktop/Voice%20Based%20Concept%20Understanding%20Analyser/requirements.txt)
* Add `fastapi`, `uvicorn`, `python-multipart`, and `google-generativeai`.

---

## Verification Plan

### Automated Tests
- Run `python test_pipeline.py` to confirm that all core boundary and scoring assertions remain unaffected.
- Add test coverage in `test_pipeline.py` for NLTK analysis and fallback qualitative feedback.

### Manual Verification
- Start FastAPI with `uvicorn server:app --reload --port 8000`.
- Start Streamlit with `streamlit run app.py`.
- Upload a spoken explanation, configure a Gemini API key, and check that the qualitative feedback, sentiment analysis, and DB history tab load perfectly.
