"""
modules/report_generator.py

ReportLab-based PDF compilation for VBCUA evaluation results.
Builds a professional downloadable PDF report.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

logger = logging.getLogger(__name__)

REQUIRED_METRIC_KEYS = (
    "semantic_similarity",
    "filler_ratio",
    "pause_ratio",
    "rms_energy",
    "overall_score",
    "understanding_level",
)

TIER_COLORS: dict[str, str] = {
    "Strong Understanding": "#2ecc71",
    "Moderate Understanding": "#f39c12",
    "Poor Understanding": "#e74c3c",
}

BANNER_COLOR = colors.HexColor("#1f2937")
HEADER_ROW_COLOR = colors.HexColor("#d3d3d3")
GRID_COLOR = colors.HexColor("#4a4a4a")


class ReportGenerationError(RuntimeError):
    """Raised when the PDF cannot be built."""


def _validate_metrics_dict(metrics_dict: dict[str, Any]) -> None:
    if not isinstance(metrics_dict, dict):
        raise ReportGenerationError(
            f"metrics_dict must be a dict, got {type(metrics_dict).__name__}."
        )
    missing = [key for key in REQUIRED_METRIC_KEYS if key not in metrics_dict]
    if missing:
        raise ReportGenerationError(f"metrics_dict is missing required keys: {missing}")


def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "banner_title": ParagraphStyle(
            "BannerTitle",
            parent=base["Title"],
            textColor=colors.white,
            fontSize=18,
            leading=22,
            alignment=1,  # centered
        ),
        "h2": ParagraphStyle(
            "SectionHeading",
            parent=base["Heading2"],
            textColor=colors.HexColor("#1f2937"),
            spaceBefore=14,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "BodyText",
            parent=base["BodyText"],
            fontSize=10,
            leading=14,
        ),
        "bold_body": ParagraphStyle(
            "BoldBody",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=14,
        ),
    }


def _build_banner(styles: dict[str, ParagraphStyle], page_width: float) -> Table:
    banner_para = Paragraph(
        "Voice-Based Concept Understanding Analyser Report", styles["banner_title"]
    )
    banner = Table([[banner_para]], colWidths=[page_width])
    banner.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), BANNER_COLOR),
                ("TOPPADDING", (0, 0), (-1, -1), 14),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return banner


def _build_metrics_table(metrics_dict: dict[str, Any], page_width: float) -> Table:
    understanding_level = str(metrics_dict["understanding_level"])
    level_color = colors.HexColor(TIER_COLORS.get(understanding_level, "#999999"))

    rows = [
        ["Metric", "Value"],
        ["Semantic Similarity", f"{float(metrics_dict['semantic_similarity']):.3f}"],
        ["Filler Word Ratio", f"{float(metrics_dict['filler_ratio']):.3f}"],
        ["Pause Ratio", f"{float(metrics_dict['pause_ratio']):.3f}"],
        ["Confidence Energy", f"{float(metrics_dict['rms_energy']):.4f}"],
    ]

    # Add extra NLP metrics if present in the dictionary
    if "zero_crossing_rate" in metrics_dict:
        rows.append(["Zero-Crossing Rate", f"{float(metrics_dict['zero_crossing_rate']):.4f}"])
    if "lexical_diversity" in metrics_dict:
        rows.append(["Lexical Diversity", f"{float(metrics_dict['lexical_diversity']):.3f}"])
    if "sentiment" in metrics_dict:
        rows.append(["Delivery Sentiment", str(metrics_dict["sentiment"])])

    rows.extend([
        ["Final Score", f"{int(metrics_dict['overall_score'])} / 100"],
        ["Understanding Level", understanding_level],
    ])

    table = Table(rows, colWidths=[page_width * 0.5, page_width * 0.5])
    level_row_index = len(rows) - 1

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HEADER_ROW_COLOR),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.75, GRID_COLOR),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("BACKGROUND", (1, level_row_index), (1, level_row_index), level_color),
                ("TEXTCOLOR", (1, level_row_index), (1, level_row_index), colors.white),
                ("FONTNAME", (1, level_row_index), (1, level_row_index), "Helvetica-Bold"),
            ]
        )
    )
    return table


def generate_pdf_report(
    output_filename: str,
    reference_concept: str,
    student_transcript: str,
    waveform_img_path: str,
    metrics_dict: dict[str, Any],
) -> str:
    """
    Assembles and saves the VBCUA evaluation PDF report.
    """
    _validate_metrics_dict(metrics_dict)

    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = _build_styles()
    page_width = A4[0] - 1.5 * inch

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="VBCUA Evaluation Report",
    )

    story: list = []

    # Section 1: Title banner
    story.append(_build_banner(styles, page_width))
    story.append(Spacer(1, 12))

    # Section 2: Reference Concept
    story.append(Paragraph("Reference Concept", styles["h2"]))
    story.append(Paragraph(reference_concept or "(No reference concept provided.)", styles["body"]))
    story.append(Spacer(1, 8))

    # Section 3: Student Transcription
    story.append(Paragraph("Student Transcription", styles["h2"]))
    story.append(Paragraph(student_transcript or "(No transcript available.)", styles["body"]))
    story.append(Spacer(1, 8))

    # Section 4: Waveform visualization
    story.append(Paragraph("Audio Waveform", styles["h2"]))
    if waveform_img_path and os.path.isfile(waveform_img_path):
        try:
            img_width = page_width * 0.9
            img = Image(waveform_img_path, width=img_width, height=img_width * 0.3)
            img.hAlign = "CENTER"
            story.append(img)
        except Exception:
            logger.exception("Failed to embed waveform image: %s", waveform_img_path)
            story.append(Paragraph("(Waveform image could not be rendered.)", styles["body"]))
    else:
        story.append(Paragraph("(Waveform image unavailable.)", styles["body"]))
    story.append(Spacer(1, 8))

    # Section 5: Metrics summary table
    story.append(Paragraph("Evaluation Summary", styles["h2"]))
    story.append(_build_metrics_table(metrics_dict, page_width))
    story.append(Spacer(1, 10))

    # Section 6: Qualitative Feedback (AI generated or Fallback)
    if "strengths" in metrics_dict or "gaps" in metrics_dict or "tips" in metrics_dict:
        story.append(Paragraph("Qualitative Feedback & Recommendations", styles["h2"]))
        
        if metrics_dict.get("strengths"):
            story.append(Paragraph("Strengths", styles["bold_body"]))
            story.append(Paragraph(metrics_dict["strengths"].replace("\n", "<br/>"), styles["body"]))
            story.append(Spacer(1, 5))
            
        if metrics_dict.get("gaps"):
            story.append(Paragraph("Gaps in Understanding", styles["bold_body"]))
            story.append(Paragraph(metrics_dict["gaps"].replace("\n", "<br/>"), styles["body"]))
            story.append(Spacer(1, 5))
            
        if metrics_dict.get("tips"):
            story.append(Paragraph("Delivery & Articulation Tips", styles["bold_body"]))
            story.append(Paragraph(metrics_dict["tips"].replace("\n", "<br/>"), styles["body"]))

    try:
        doc.build(story)
    except Exception as exc:
        logger.exception("Failed to build PDF report at: %s", output_path)
        raise ReportGenerationError(f"Failed to write PDF report: {exc}") from exc

    logger.info("PDF report generated: %s", output_path)
    return str(output_path)
