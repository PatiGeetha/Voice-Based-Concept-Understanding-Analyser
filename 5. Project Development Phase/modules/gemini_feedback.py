"""
modules/gemini_feedback.py

Google Gemini API qualitative feedback engine for VBCUA.
Generates structured educational assessments (strengths, gaps, articulation recommendations)
with a robust local rule-based fallback if no Gemini API key is provided or the API call fails.
"""

import os
import logging
from typing import Dict
import google.generativeai as genai

logger = logging.getLogger(__name__)


def generate_qualitative_feedback(
    student_transcript: str,
    reference_concept: str,
    overall_score: int,
    filler_ratio: float,
    pause_ratio: float,
    similarity_score: float,
    api_key: str = None
) -> Dict[str, str]:
    """
    Generates actionable feedback: strengths, gaps in understanding, and delivery tips.
    Prioritizes Google Gemini; falls back to deterministic rule-based analysis on failure or if no key is present.
    """
    # 1. API Key Resolution
    resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or st_sidebar_fallback_check()
    
    if resolved_key:
        try:
            logger.info("Initializing Google Gemini model for feedback generation.")
            genai.configure(api_key=resolved_key)
            # Use gemini-1.5-flash for fast, responsive text analysis
            model = genai.GenerativeModel("gemini-1.5-flash")
            
            prompt = f"""
            You are an expert academic evaluator. Analyze the student's spoken explanation compared to the reference concept.
            
            Reference Concept:
            "{reference_concept}"
            
            Student's Spoken Explanation:
            "{student_transcript}"
            
            Metrics:
            - Semantic Similarity Score: {similarity_score:.2f} / 1.00
            - Overall Concept Understanding Score: {overall_score} / 100
            - Filler Word Ratio: {filler_ratio:.2f}
            - Silent Pause Ratio: {pause_ratio:.2f}
            
            Provide structured evaluation feedback containing exactly three sections formatted in clean Markdown. Keep it brief, professional, and constructive (maximum 3-4 bullet points per section):
            
            ### Strengths
            (Identify what part of the concept was explained correctly or what key terms were accurate)
            
            ### Gaps in Understanding
            (Specify any key facts, definitions, or patterns from the reference concept that were omitted or misstated)
            
            ### Delivery & Articulation Tips
            (Give actionable tips based on their filler word usage and silence pauses)
            """
            
            response = model.generate_content(prompt)
            text = response.text.strip()
            
            # Simple splitter to return structured chunks
            strengths = ""
            gaps = ""
            tips = ""
            
            if "### Strengths" in text:
                parts = text.split("### Strengths")[1].split("### Gaps in Understanding")
                strengths = parts[0].strip()
                if len(parts) > 1:
                    sub_parts = parts[1].split("### Delivery & Articulation Tips")
                    gaps = sub_parts[0].strip()
                    if len(sub_parts) > 1:
                        tips = sub_parts[1].strip()
            
            # If splitting failed or was formatted differently, return the full text in notes
            if not strengths and not gaps:
                return {
                    "strengths": "• Good effort in explaining the concept.",
                    "gaps": "• Refer to the reference concept details.",
                    "tips": "• Try to speak more continuously and reduce fillers.",
                    "raw_output": text
                }
                
            return {
                "strengths": strengths,
                "gaps": gaps,
                "tips": tips,
                "raw_output": text
            }
            
        except Exception as exc:
            logger.exception("Gemini feedback generation failed; falling back to rule-based metrics.")
            
    # 2. Local Fallback Generator
    return _generate_fallback_feedback(similarity_score, filler_ratio, pause_ratio, reference_concept)


def st_sidebar_fallback_check() -> str:
    """Helper to safely check if streamlit is running and import key if set."""
    try:
        import streamlit as st
        return st.session_state.get("gemini_api_key", "")
    except ImportError:
        return ""


def _generate_fallback_feedback(
    similarity: float, filler_ratio: float, pause_ratio: float, reference_concept: str
) -> Dict[str, str]:
    """Generates qualitative, rule-based feedback based on scores and transcript metrics."""
    # Strengths
    if similarity > 0.7:
        strengths = (
            "• Demonstrates solid mastery of the concept.\n"
            "• Correctly covers most of the core conceptual definitions.\n"
            "• Vocabulary matches the academic terminology well."
        )
    elif similarity > 0.4:
        strengths = (
            "• Recalls basic definitions and covers some fundamental terms.\n"
            "• The explanation has partial alignment with the target reference concept."
        )
    else:
        strengths = (
            "• Basic attempts made to vocalize the explanation.\n"
            "• Some topic context terms detected."
        )

    # Gaps
    if similarity > 0.7:
        gaps = "• Minimal semantic gaps. Minor nuances or secondary details could be articulated more fully."
    elif similarity > 0.4:
        gaps = (
            "• Misses key structural or operational patterns mentioned in the reference.\n"
            "• Explanation lacks comprehensive coverage of the sub-components."
        )
    else:
        gaps = (
            "• Significant conceptual mismatch. Key definitions are absent.\n"
            "• Explanation deviates significantly from the ground-truth concept text."
        )

    # Articulation Tips
    tips_list = []
    if filler_ratio >= 0.05:
        tips_list.append("• High filler word count detected. Try to speak slower and pause silently instead of saying 'um' or 'like'.")
    else:
        tips_list.append("• Excellent delivery with low filler word usage. Keep up this clear articulation pattern.")

    if pause_ratio >= 0.25:
        tips_list.append("• Frequent silences observed. Organize thoughts before speaking or use short notes to reduce pauses.")
    else:
        tips_list.append("• Good pacing and speech continuity. Pauses are within normal conversational ranges.")

    return {
        "strengths": strengths,
        "gaps": gaps,
        "tips": "\n".join(tips_list),
        "raw_output": "Fallback analysis applied (No active Gemini API key or call failed)."
    }
