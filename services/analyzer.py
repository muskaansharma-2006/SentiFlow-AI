"""Model inference and lightweight insight extraction."""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Any
from preprocessing.text_preprocessing import preprocess_text

EMOTION_KEYWORDS = {
    "happy": {"love", "great", "perfect", "happy", "thanks", "amazing", "excellent"},
    "excited": {"excited", "awesome", "wow", "fantastic", "best", "brilliant"},
    "angry": {"angry", "hate", "worst", "useless", "terrible", "furious"},
    "frustrated": {"slow", "broken", "crash", "error", "refund", "issue", "problem", "disappointed"},
    "sad": {"sad", "poor", "bad", "unhappy", "regret"},
}
DOMAIN_KEYWORDS = {
    "food": {"food", "taste", "restaurant", "meal", "pizza", "delivery", "order"},
    "hotel": {"hotel", "room", "stay", "staff", "booking", "clean", "bed"},
    "app": {"app", "ui", "login", "update", "feature", "payment", "crash"},
    "service": {"service", "support", "agent", "customer", "help", "response"},
}
ISSUE_KEYWORDS = {"slow", "crash", "error", "refund", "payment", "support", "delivery", "dirty", "login", "bug"}


class SentimentAnalyzer:
    def __init__(self, model_path: str, vectorizer_path: str):
        self.model_path, self.vectorizer_path = Path(model_path), Path(vectorizer_path)
        self.model = self.vectorizer = None

    def load(self) -> None:
        if self.model is None:
            with self.model_path.open("rb") as f:
                self.model = pickle.load(f)
            with self.vectorizer_path.open("rb") as f:
                self.vectorizer = pickle.load(f)

    def analyze(self, text: str) -> dict[str, Any]:
        cleaned = preprocess_text(text)
        if not cleaned:
            raise ValueError("Enter a review containing letters or words.")
        self.load()
        features = self.vectorizer.transform([cleaned])
        probabilities = self.model.predict_proba(features)[0]
        index = probabilities.argmax()
        sentiment = str(self.model.classes_[index]).title()
        words = set(cleaned.split())
        emotion = self._best_match(words, EMOTION_KEYWORDS, fallback=self._sentiment_emotion(sentiment))
        domain = self._best_match(words, DOMAIN_KEYWORDS, fallback="general")
        issues = sorted(words.intersection(ISSUE_KEYWORDS)) if sentiment == "Negative" else []
        
        confidence = round(float(probabilities[index]) * 100, 2)
        prob_scores = {str(self.model.classes_[i]).title(): round(float(probabilities[i]) * 100, 2) for i in range(len(self.model.classes_))}
        
        explanation = self._generate_explanation(sentiment, confidence, words, emotion)
        recommendations = self._generate_recommendation(sentiment, emotion, domain, issues)
        
        return {
            "original_text": text.strip(),
            "processed_text": cleaned,
            "sentiment": sentiment,
            "confidence": confidence,
            "emotion": emotion,
            "domain": domain,
            "problems": issues,
            "probabilities": prob_scores,
            "explanation": explanation,
            "recommendation": recommendations
        }

    def _generate_explanation(self, sentiment: str, confidence: float, words: set[str], emotion: str) -> str:
        matched_words = []
        for kw_group in EMOTION_KEYWORDS.values():
            matched_words.extend(words.intersection(kw_group))
        for kw_group in DOMAIN_KEYWORDS.values():
            matched_words.extend(words.intersection(kw_group))
        matched_words = sorted(list(set(matched_words)))
        
        explanation = f"The review is classified as **{sentiment}** with a confidence score of **{confidence}%**."
        
        if matched_words:
            kw_list = ", ".join([f"'{w}'" for w in matched_words[:4]])
            explanation += f" This classification is supported by key emotion/context terms detected in the text, such as: {kw_list}."
        else:
            explanation += " The prediction is based on general syntactic patterns and vocabulary distribution in the text."
            
        explanation += f" The dominant emotion is detected as **{emotion}**."
        return explanation

    def _generate_recommendation(self, sentiment: str, emotion: str, domain: str, issues: list[str]) -> list[str]:
        recommendations = []
        if sentiment == "Negative":
            if domain == "app":
                recommendations.append("Prioritize fixing app stability and performance issues in the current engineering sprint.")
            elif domain == "service":
                recommendations.append("Escalate this to customer support supervisors for active user outreach and recovery.")
            elif domain == "food":
                recommendations.append("Audit order prep and delivery times. Offer a refund or voucher to restore goodwill.")
            elif domain == "hotel":
                recommendations.append("Alert guest relations to check on room/cleanliness/staff conditions mentioned by this guest.")
            else:
                recommendations.append("Initiate direct support outreach to investigate the customer's issues.")
            
            if issues:
                issue_str = ", ".join(issues)
                recommendations.append(f"Focus on resolving specific customer pain points: {issue_str}.")
            recommendations.append("Send a personalized follow-up email apologizing and outlining the corrective actions taken.")
        
        elif sentiment == "Positive":
            if domain == "app":
                recommendations.append("Prompt the user to review the app on the App/Play Store to improve store rating.")
            elif domain == "service" or domain == "hotel":
                recommendations.append("Share this stellar feedback with the team/staff members involved to boost morale.")
            else:
                recommendations.append("Send a brief thank-you note and highlight upcoming features/releases.")
            recommendations.append("Consider onboarding this user as a potential brand advocate or case study.")
            
        else: # Neutral
            recommendations.append("Send a brief follow-up survey to identify what would turn their neutral experience into a positive one.")
            recommendations.append("Monitor user interaction trends to prevent shift to negative sentiment.")
            
        return recommendations

    @staticmethod
    def _best_match(words, groups, fallback):
        scored = [(len(words.intersection(keywords)), name) for name, keywords in groups.items()]
        score, name = max(scored)
        return name if score else fallback

    @staticmethod
    def _sentiment_emotion(sentiment):
        return {"Positive": "happy", "Negative": "frustrated", "Neutral": "neutral"}[sentiment]

