"""
modules/nlp_utils.py

NLTK-based local Natural Language Processing and Sentiment Analysis for VBCUA.
Calculates lexical metrics (Type-Token Ratio, sentence counts) and analyzes
delivery sentiment/tone using NLTK's VADER.
"""

import logging
from typing import Dict, Any
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# Ensure NLTK datasets are downloaded programmatically
try:
    nltk.download("punkt", quiet=True)
    nltk.download("vader_lexicon", quiet=True)
except Exception:
    logger.exception("Failed programmatically downloading NLTK datasets; proceeding with local fallback.")


class NLPAnalyzer:
    """
    Local NLP metrics analyzer using NLTK.
    Provides lexical density, diversity, and VADER sentiment intensity.
    """

    def __init__(self):
        try:
            self.sia = SentimentIntensityAnalyzer()
        except Exception:
            logger.warning("VADER sentiment analyzer could not be initialized; sentiment will default to Neutral.")
            self.sia = None

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Runs sentence count, lexical diversity (Type-Token Ratio),
        and VADER sentiment analysis on the text.
        """
        text_stripped = (text or "").strip()
        if not text_stripped:
            return {
                "sentence_count": 0,
                "lexical_diversity": 0.0,
                "sentiment": "Neutral",
                "sentiment_scores": {"pos": 0.0, "neu": 1.0, "neg": 0.0, "compound": 0.0}
            }

        # Tokenization & lexical diversity (Type-Token Ratio)
        try:
            sentences = nltk.sent_tokenize(text_stripped)
            sentence_count = len(sentences)
        except Exception:
            sentence_count = max(1, text_stripped.count(".") + text_stripped.count("?"))

        try:
            words = [w.lower() for w in nltk.word_tokenize(text_stripped) if w.isalnum()]
            total_words = len(words)
            unique_words = len(set(words))
            lexical_diversity = (unique_words / total_words) if total_words > 0 else 0.0
        except Exception:
            # Fallback simple split
            words = [w.lower() for w in text_stripped.split() if w.isalnum()]
            total_words = len(words)
            unique_words = len(set(words))
            lexical_diversity = (unique_words / total_words) if total_words > 0 else 0.0

        # VADER Sentiment
        sentiment_scores = {"pos": 0.0, "neu": 1.0, "neg": 0.0, "compound": 0.0}
        sentiment = "Neutral"

        if self.sia:
            try:
                sentiment_scores = self.sia.polarity_scores(text_stripped)
                compound = sentiment_scores.get("compound", 0.0)
                if compound >= 0.05:
                    sentiment = "Confident / Positive"
                elif compound <= -0.05:
                    sentiment = "Hesitant / Negative"
                else:
                    sentiment = "Neutral"
            except Exception:
                logger.exception("VADER polarity scoring failed.")

        return {
            "sentence_count": sentence_count,
            "lexical_diversity": round(lexical_diversity, 4),
            "sentiment": sentiment,
            "sentiment_scores": sentiment_scores
        }
