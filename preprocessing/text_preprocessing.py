"""Deterministic review cleaning used for both model training and inference."""
import re

SLANG_MAP = {
    "goat": "greatest", "lit": "excellent", "fire": "excellent", "slay": "excellent",
    "fr": "for real", "ngl": "not going to lie", "idk": "i do not know",
    "imo": "in my opinion", "imho": "in my honest opinion", "tbh": "to be honest",
    "omg": "oh my god", "wtf": "what the fuck", "smh": "disappointed",
    "luv": "love", "u": "you", "ur": "your", "rly": "really", "btw": "by the way",
    "bruh": "disappointed", "sus": "suspicious", "meh": "average", "thx": "thanks",
    "pls": "please", "plz": "please", "b4": "before", "gr8": "great",
}


def preprocess_text(text: str) -> str:
    """Normalize noisy informal text without removing sentiment-bearing negations."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@[\w_]+", " ", text)
    text = re.sub(r"#([\w_]+)", r"\1", text)
    text = re.sub(r"\b\w+\b", lambda m: SLANG_MAP.get(m.group(), m.group()), text)
    text = re.sub(r"([!?])\1+", r"\1", text)
    text = re.sub(r"[^a-z\s!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_for_analytics(text: str) -> str:
    """Clean text by removing stopwords and punctuation for Word Cloud and analytics."""
    cleaned = preprocess_text(text)
    if not cleaned:
        return ""
    try:
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words("english"))
    except Exception:
        # Fallback if nltk corpora are somehow missing
        stop_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "to", "for", "with", "in", "on", "at", "by", "of", "it", "this", "that", "these", "those"}
    
    words = cleaned.split()
    filtered_words = [w for w in words if w not in stop_words and len(w) > 1]
    return " ".join(filtered_words)

