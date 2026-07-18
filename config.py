"""Application settings loaded from environment variables."""
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    mongo_uri: str = os.getenv("MONGO_URI", "")
    mongo_database: str = os.getenv("MONGO_DATABASE", "sentiflow")
    model_path: str = os.getenv("MODEL_PATH", "model/sentiment_model.pkl")
    vectorizer_path: str = os.getenv("VECTORIZER_PATH", "model/tfidf_vectorizer.pkl")


settings = Settings()
