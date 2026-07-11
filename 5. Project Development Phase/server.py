"""
server.py

FastAPI backend service for VBCUA.
Handles audio uploads, speech transcription, NLP analysis, scoring,
SQLite database transactions, and PDF report generation.
"""

import json
import logging
import uuid
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from modules import (
    audio_utils,
    db_manager,
    gemini_feedback,
    nlp_utils,
    report_generator,
    scoring_engine,
    semantic_eval,
    speech_to_text,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VBCUA Service API",
    description="Backend API for Voice-Based Concept Understanding Analyser",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ASSETS_DIR = Path("assets")
UPLOAD_DIR = ASSETS_DIR / "uploads"
WAVEFORM_DIR = ASSETS_DIR / "waveforms"
REPORT_DIR = ASSETS_DIR / "reports"

# Ensure dirs exist
for d in (UPLOAD_DIR, WAVEFORM_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Engines loaded once at startup
stt_engine = speech_to_text.SpeechToTextEngine(model_size="base")
semantic_evaluator = semantic_eval.SemanticEvaluator()
nlp_analyzer = nlp_utils.NLPAnalyzer()


@app.get("/api/concepts")
def get_concepts():
    """Returns pre-loaded reference concepts from db/memory."""
    # Ensure some defaults are populated in DB reference_concept table
    defaults = {
        "Machine Learning": (
            "Machine learning is a subfield of artificial intelligence in which systems "
            "learn patterns directly from data rather than following explicitly programmed "
            "rules. A model is trained on historical examples, adjusts its internal "
            "parameters to minimize prediction error, and then generalizes that learned "
            "pattern to make predictions on new, unseen data."
        ),
        "Cloud Computing": (
            "Cloud computing is the on-demand delivery of computing resources — including "
            "servers, storage, databases, networking, software, analytics, and intelligence "
            "— over the internet to offer faster innovation, flexible resources, and "
            "economies of scale."
        )
    }
    for title, text in defaults.items():
        db_manager.save_reference_concept(title, text)
        
    return db_manager.get_all_reference_concepts()


@app.get("/api/history")
def get_history():
    """Fetches full evaluation history from database."""
    try:
        return db_manager.get_evaluation_history()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database query failed: {exc}")


@app.get("/api/report/{result_id}")
def download_report(result_id: int):
    """Retrieves PDF report for a given result ID."""
    history = db_manager.get_evaluation_history()
    pdf_path = None
    for row in history:
        if row["result_id"] == result_id:
            pdf_path = row["pdf_path"]
            break
            
    if not pdf_path or not Path(pdf_path).is_file():
        raise HTTPException(status_code=404, detail="PDF report not found.")
        
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename="vbcua_report.pdf"
    )


@app.post("/api/evaluate")
async def evaluate_audio(
    audio: UploadFile = File(...),
    concept_title: str = Form(...),
    concept_text: str = Form(...),
    user_name: str = Form("Default Learner"),
    user_email: str = Form("learner@vbcua.edu"),
    user_role: str = Form("learner"),
    gemini_api_key: Optional[str] = Form(None)
):
    """
    Runs full speech-to-text, audio signal metrics, scoring, NLP, and Gemini feedback.
    Saves everything to the 10-table SQLite schema, generates the PDF, and returns results.
    """
    temp_audio_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{audio.filename}"
    try:
        # Step 1: Save audio file locally
        with open(temp_audio_path, "wb") as buffer:
            content = await audio.read()
            buffer.write(content)

        # Step 2: Extract audio metrics and wave visualization
        y, sr = audio_utils.load_audio_signal(temp_audio_path)
        raw_metrics = audio_utils.extract_raw_audio_metrics(y, sr)
        
        waveform_path = WAVEFORM_DIR / f"{uuid.uuid4().hex}.png"
        audio_utils.save_waveform(temp_audio_path, output_img_path=waveform_path)

        # Step 3: Transcription
        transcript_text = stt_engine.transcribe(temp_audio_path)

        # Step 4: Semantic Similarity match
        similarity_score = semantic_evaluator.compute_similarity(transcript_text, concept_text)

        # Step 5: Filler Ratio stats
        filler_ratio, filler_count, total_words = scoring_engine.calculate_filler_ratio(transcript_text)

        # Step 6: Scoring Engine
        breakdown = scoring_engine.evaluate_understanding(
            similarity=similarity_score,
            filler_ratio=filler_ratio,
            audio={
                "pause_ratio": raw_metrics["pause_ratio"],
                "rms_energy": raw_metrics["rms_energy"]
            }
        )

        # Step 7: NLTK local analysis (sentiment and lexical)
        nlp_results = nlp_analyzer.analyze_text(transcript_text)

        # Step 8: Gemini Qualitative feedback
        feedback = gemini_feedback.generate_qualitative_feedback(
            student_transcript=transcript_text,
            reference_concept=concept_text,
            overall_score=breakdown.overall_score,
            filler_ratio=filler_ratio,
            pause_ratio=raw_metrics["pause_ratio"],
            similarity_score=similarity_score,
            api_key=gemini_api_key
        )

        # Step 9: Database persistence (10-table transaction)
        user_id = db_manager.save_user(user_name, user_email, user_role)
        ref_concept_id = db_manager.save_reference_concept(concept_title, concept_text)
        
        audio_id = db_manager.save_audio_file(
            user_id=user_id,
            file_name=audio.filename,
            file_path=str(temp_audio_path),
            duration_sec=raw_metrics["duration_sec"]
        )
        
        transcript_id = db_manager.save_transcript(audio_id, transcript_text)
        
        db_manager.save_audio_features(
            audio_id=audio_id,
            pause_ratio=raw_metrics["pause_ratio"],
            rms_energy=raw_metrics["rms_energy"],
            zero_crossing_rate=raw_metrics["zero_crossing_rate"],
            duration_sec=raw_metrics["duration_sec"]
        )
        
        db_manager.save_filler_word_stats(
            transcript_id=transcript_id,
            filler_count=filler_count,
            total_words=total_words,
            filler_ratio=filler_ratio
        )
        
        db_manager.save_semantic_similarity(
            transcript_id=transcript_id,
            ref_concept_id=ref_concept_id,
            similarity_score=similarity_score
        )

        # Build JSON notes containing advanced feedback
        notes_dict = {
            "strengths": feedback["strengths"],
            "gaps": feedback["gaps"],
            "tips": feedback["tips"],
            "sentiment": nlp_results["sentiment"],
            "lexical_diversity": nlp_results["lexical_diversity"],
            "sentence_count": nlp_results["sentence_count"],
            "raw_api_output": feedback.get("raw_output", "")
        }
        notes_json = json.dumps(notes_dict)

        result_id = db_manager.save_evaluation_result(
            audio_id=audio_id,
            ref_concept_id=ref_concept_id,
            overall_score=breakdown.overall_score,
            understanding_level=breakdown.understanding_level,
            notes=notes_json
        )

        # Step 10: PDF report generation
        pdf_path = REPORT_DIR / f"{uuid.uuid4().hex}_report.pdf"
        
        full_report_metrics = {
            "semantic_similarity": similarity_score,
            "filler_ratio": filler_ratio,
            "pause_ratio": raw_metrics["pause_ratio"],
            "rms_energy": raw_metrics["rms_energy"],
            "overall_score": breakdown.overall_score,
            "understanding_level": breakdown.understanding_level,
            "zero_crossing_rate": raw_metrics["zero_crossing_rate"],
            "lexical_diversity": nlp_results["lexical_diversity"],
            "sentiment": nlp_results["sentiment"],
            "strengths": feedback["strengths"],
            "gaps": feedback["gaps"],
            "tips": feedback["tips"]
        }
        
        report_generator.generate_pdf_report(
            output_filename=str(pdf_path),
            reference_concept=concept_text,
            student_transcript=transcript_text,
            waveform_img_path=str(waveform_path),
            metrics_dict=full_report_metrics
        )

        # Calculate file size
        file_size_kb = round(os.path.getsize(pdf_path) / 1024, 2)
        db_manager.save_report(result_id, str(pdf_path), file_size_kb)
        
        # Save session
        db_manager.save_session(user_id=user_id, status="ended")

        # Return full payload
        return {
            "result_id": result_id,
            "transcript": transcript_text,
            "overall_score": breakdown.overall_score,
            "understanding_level": breakdown.understanding_level,
            "tier_color": breakdown.tier_color,
            "metrics": {
                "similarity": similarity_score,
                "filler_ratio": filler_ratio,
                "filler_count": filler_count,
                "total_words": total_words,
                "pause_ratio": raw_metrics["pause_ratio"],
                "rms_energy": raw_metrics["rms_energy"],
                "zero_crossing_rate": raw_metrics["zero_crossing_rate"],
                "duration_sec": raw_metrics["duration_sec"],
                "lexical_diversity": nlp_results["lexical_diversity"],
                "sentence_count": nlp_results["sentence_count"],
                "sentiment": nlp_results["sentiment"]
            },
            "feedback": {
                "strengths": feedback["strengths"],
                "gaps": feedback["gaps"],
                "tips": feedback["tips"]
            },
            "waveform_path": str(waveform_path),
            "pdf_path": str(pdf_path)
        }

    except Exception as exc:
        logger.exception("Evaluation endpoint failure.")
        raise HTTPException(status_code=500, detail=str(exc))
        
    finally:
        # Optional: Clean up temporary files if required,
        # but we keep them in assets/uploads and assets/waveforms for rendering.
        pass
