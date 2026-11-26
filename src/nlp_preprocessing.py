"""NLP preprocessing, stylometric feature extraction, and explainability signals."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Self-contained English stopword list (avoids runtime NLTK downloads)
ENGLISH_STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "same", "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "you", "your", "yours", "yourself", "yourselves",
})

PROMO_SPAM_PHRASES = (
    "best product ever",
    "must buy",
    "buy now",
    "order now",
    "100% genuine",
    "100% original",
    "life changing",
    "changed my life",
    "worth every penny",
    "highly recommend to everyone",
    "recommend to all",
    "five stars",
    "5 stars",
    "promo code",
    "discount code",
    "coupon code",
    "click here",
    "check my bio",
    "visit my profile",
    "free gift",
    "full refund",
    "cashback",
    "whatsapp",
    "telegram",
    "dm me",
    "best seller ever",
    "guaranteed results",
    "miracle product",
    "don't think twice",
    "just buy it",
    "thank me later",
)

SUPERLATIVE_WORDS = frozenset({
    "amazing", "awesome", "incredible", "unbelievable", "perfect", "miracle",
    "fantastic", "phenomenal", "flawless", "extraordinary", "mindblowing",
    "worst", "garbage", "trash", "scam", "fraud", "useless", "pathetic",
})

BALANCED_DETAIL_CUES = (
    "however", "although", "though", "slightly", "minor", "compared to",
    "pros", "cons", "battery", "cable", "stitching", "fabric", "material",
    "packaging", "delivery", "arrived", "weeks", "months", "days", "hours",
    "sizing", "fit", "tight", "loose", "weight", "button", "screen", "noise",
    "temperature", "texture", "scent", "returned", "replacement", "manual",
    "charging", "brightness", "zipper", "handle", "strap", "price", "value",
)

FIRST_PERSON_PRONOUNS = frozenset({"i", "me", "my", "mine", "myself", "we", "our", "ours"})

URL_OR_CONTACT_RE = re.compile(
    r"(https?://\S+|www\.\S+|\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b)",
    re.IGNORECASE,
)
HTML_TAG_RE = re.compile(r"<[^>]+>")
WORD_RE = re.compile(r"[a-zA-Z0-9']+")

STYLOMETRIC_FEATURE_NAMES = [
    "char_count",
    "word_count",
    "avg_word_length",
    "exclamation_count",
    "exclamation_ratio",
    "uppercase_ratio",
    "all_caps_words",
    "digit_count",
    "unique_word_ratio",
    "promo_spam_hits",
    "superlative_hits",
    "balanced_detail_hits",
    "first_person_ratio",
    "url_or_contact_flag",
    "repeated_word_max",
    "punctuation_burst",
]

STYLOMETRIC_DISPLAY_NAMES = {
    "char_count": "Character length",
    "word_count": "Word count",
    "avg_word_length": "Average word length",
    "exclamation_count": "Exclamation marks",
    "exclamation_ratio": "Exclamation density",
    "uppercase_ratio": "Uppercase letter ratio",
    "all_caps_words": "ALL-CAPS words",
    "digit_count": "Specific numbers/digits",
    "unique_word_ratio": "Vocabulary diversity (TTR)",
    "promo_spam_hits": "Promotional / spam phrases",
    "superlative_hits": "Extreme superlatives",
    "balanced_detail_hits": "Specific usage & detail cues",
    "first_person_ratio": "First-person pronoun ratio",
    "url_or_contact_flag": "URL / contact info present",
    "repeated_word_max": "Max single-word repetition",
    "punctuation_burst": "Repeated punctuation bursts (!! / ??)",
}


def clean_text(text: Any) -> str:
    """Normalize review text for TF-IDF vectorization and similarity checks."""
    if text is None:
        return ""
    s = str(text)
    s = HTML_TAG_RE.sub(" ", s)
    s = URL_OR_CONTACT_RE.sub(" urltoken ", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize_without_stopwords(text: str) -> list[str]:
    """Tokenize cleaned text and remove common English stopwords."""
    cleaned = clean_text(text)
    tokens = WORD_RE.findall(cleaned)
    return [t for t in tokens if t not in ENGLISH_STOPWORDS and len(t) > 1]


def extract_stylometric_features(text: Any) -> dict[str, float]:
    """Extract interpretable stylometric and behavioral NLP features from raw text."""
    raw = "" if text is None else str(text).strip()
    cleaned = clean_text(raw)
    words = WORD_RE.findall(cleaned)
    raw_alpha_words = re.findall(r"\b[A-Za-z]{2,}\b", raw)

    char_count = float(len(raw))
    word_count = float(len(words))
    avg_word_length = float(np.mean([len(w) for w in words])) if words else 0.0

    exclamation_count = float(raw.count("!"))
    exclamation_ratio = exclamation_count / max(char_count, 1.0) * 100.0

    alpha_chars = [c for c in raw if c.isalpha()]
    upper_chars = [c for c in alpha_chars if c.isupper()]
    uppercase_ratio = len(upper_chars) / max(len(alpha_chars), 1)
    all_caps_words = float(sum(1 for w in raw_alpha_words if w.isupper()))

    digit_count = float(sum(1 for c in raw if c.isdigit()))
    unique_word_ratio = len(set(words)) / max(len(words), 1)

    lower_raw = raw.lower()
    promo_spam_hits = float(sum(1 for phrase in PROMO_SPAM_PHRASES if phrase in lower_raw))
    superlative_hits = float(sum(1 for w in words if w in SUPERLATIVE_WORDS))
    balanced_detail_hits = float(sum(1 for cue in BALANCED_DETAIL_CUES if cue in lower_raw))

    first_person_count = sum(1 for w in words if w in FIRST_PERSON_PRONOUNS)
    first_person_ratio = first_person_count / max(len(words), 1)

    url_or_contact_flag = 1.0 if URL_OR_CONTACT_RE.search(raw) else 0.0

    content_words = [w for w in words if w not in ENGLISH_STOPWORDS and len(w) > 2]
    repeated_word_max = float(max(Counter(content_words).values())) if content_words else 0.0
    punctuation_burst = float(len(re.findall(r"[!?]{2,}", raw)))

    return {
        "char_count": char_count,
        "word_count": word_count,
        "avg_word_length": avg_word_length,
        "exclamation_count": exclamation_count,
        "exclamation_ratio": exclamation_ratio,
        "uppercase_ratio": uppercase_ratio,
        "all_caps_words": all_caps_words,
        "digit_count": digit_count,
        "unique_word_ratio": unique_word_ratio,
        "promo_spam_hits": promo_spam_hits,
        "superlative_hits": superlative_hits,
        "balanced_detail_hits": balanced_detail_hits,
        "first_person_ratio": first_person_ratio,
        "url_or_contact_flag": url_or_contact_flag,
        "repeated_word_max": repeated_word_max,
        "punctuation_burst": punctuation_burst,
    }


class StylometricTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer converting raw review strings to stylometric arrays."""

    def fit(self, X: Any, y: Any = None) -> "StylometricTransformer":
        return self

    def transform(self, X: Any) -> np.ndarray:
        rows = []
        for item in X:
            feats = extract_stylometric_features(item)
            rows.append([feats[name] for name in STYLOMETRIC_FEATURE_NAMES])
        return np.asarray(rows, dtype=np.float64)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(STYLOMETRIC_FEATURE_NAMES, dtype=object)


def detect_linguistic_signals(text: str) -> list[dict[str, str]]:
    """Produce human-readable linguistic and heuristic signals explaining review traits."""
    raw = "" if text is None else str(text).strip()
    lower_raw = raw.lower()
    feats = extract_stylometric_features(raw)
    signals: list[dict[str, str]] = []

    matched_promos = [p for p in PROMO_SPAM_PHRASES if p in lower_raw]
    if matched_promos:
        signals.append({
            "severity": "warning",
            "title": "Promotional or incentivized phrasing",
            "detail": f"Contains common spam/promotional cues: {', '.join(repr(p) for p in matched_promos[:4])}.",
        })

    if feats["url_or_contact_flag"] > 0:
        signals.append({
            "severity": "warning",
            "title": "External link or contact pattern",
            "detail": "Review includes a URL, email address, or phone-like sequence often seen in spam.",
        })

    if feats["exclamation_count"] >= 3 or feats["punctuation_burst"] >= 1:
        signals.append({
            "severity": "warning",
            "title": "Excessive punctuation emphasis",
            "detail": f"Detected {int(feats['exclamation_count'])} exclamation mark(s) and {int(feats['punctuation_burst'])} repeated punctuation burst(s).",
        })

    if feats["uppercase_ratio"] > 0.25 or feats["all_caps_words"] >= 2:
        signals.append({
            "severity": "warning",
            "title": "Heavy ALL-CAPS usage",
            "detail": f"{feats['uppercase_ratio'] * 100:.1f}% uppercase letters ({int(feats['all_caps_words'])} ALL-CAPS word(s)), common in aggressive hype or bash spam.",
        })

    if feats["superlative_hits"] >= 2 and feats["balanced_detail_hits"] == 0:
        signals.append({
            "severity": "warning",
            "title": "High hype with low product specificity",
            "detail": f"Uses {int(feats['superlative_hits'])} extreme superlative(s) without concrete usage details.",
        })

    if feats["word_count"] <= 6 and feats["balanced_detail_hits"] == 0:
        signals.append({
            "severity": "warning",
            "title": "Very short generic text",
            "detail": f"Only {int(feats['word_count'])} word(s); very brief reviews carry limited verifiable product detail.",
        })

    if feats["repeated_word_max"] >= 3:
        signals.append({
            "severity": "warning",
            "title": "Word repetition",
            "detail": f"A single content word repeats {int(feats['repeated_word_max'])} times, suggesting template stuffing.",
        })

    if feats["balanced_detail_hits"] >= 2:
        signals.append({
            "severity": "positive",
            "title": "Concrete product & usage details",
            "detail": f"Contains {int(feats['balanced_detail_hits'])} specific usage, comparison, or nuanced evaluation cues.",
        })

    if feats["word_count"] >= 18 and feats["unique_word_ratio"] >= 0.68 and feats["promo_spam_hits"] == 0:
        signals.append({
            "severity": "positive",
            "title": "Natural lexical diversity",
            "detail": f"{int(feats['word_count'])} words with {feats['unique_word_ratio'] * 100:.0f}% unique vocabulary.",
        })

    if not signals:
        signals.append({
            "severity": "neutral",
            "title": "Balanced baseline phrasing",
            "detail": "No strong heuristic spam triggers or detailed technical markers stood out; decision relies mainly on n-gram context.",
        })

    return signals
