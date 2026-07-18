"""MongoDB repository with Streamlit session-state fallback for sandbox testing."""
from datetime import datetime, timezone
from typing import Any


class ReviewRepository:
    def __init__(self, uri: str, database: str):
        self.collection = None
        self._uri = uri
        self._database = database
        if uri:
            try:
                from pymongo import MongoClient
                client = MongoClient(uri, serverSelectionTimeoutMS=2500)
                client.admin.command("ping")
                self.collection = client[database]["reviews"]
                self.collection.create_index("timestamp")
            except Exception:
                self.collection = None

    @property
    def connected(self) -> bool:
        return self.collection is not None

    def save(self, review: dict[str, Any]) -> bool:
        """Save analysis result matching the enhanced schema fields."""
        # Convert timestamp to datetime object if it is a string, otherwise use current time
        ts = review.get("timestamp")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                ts = datetime.now(timezone.utc)
        elif not isinstance(ts, datetime):
            ts = datetime.now(timezone.utc)

        document = {
            "review": review.get("original_text", ""),
            "corrected_review": review.get("corrected_text", review.get("original_text", "")),
            "cleaned_review": review.get("processed_text", ""),
            "sentiment": review.get("sentiment", ""),
            "emotion": review.get("emotion", ""),
            "confidence": float(review.get("confidence", 0.0)),
            "timestamp": ts,
            "source": review.get("source", "Web"),
            "domain": review.get("domain", "general"),
            "problems": review.get("problems", []),
            "probabilities": review.get("probabilities", {}),
            "explanation": review.get("explanation", ""),
            "recommendation": review.get("recommendation", [])
        }

        if self.collection:
            try:
                self.collection.insert_one(document)
                return True
            except Exception:
                pass

        # Fallback to Streamlit session state
        try:
            import streamlit as st
            if "_mock_reviews" not in st.session_state:
                st.session_state["_mock_reviews"] = []
            st.session_state["_mock_reviews"].append(document)
            return True
        except Exception:
            return False

    def recent(self, limit: int = 500) -> list[dict[str, Any]]:
        """Fetch recent reviews, falling back to local session state mock database."""
        if self.collection:
            try:
                cursor = self.collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
                return list(cursor)
            except Exception:
                pass

        try:
            import streamlit as st
            if "_mock_reviews" not in st.session_state:
                st.session_state["_mock_reviews"] = []
            # Sort by timestamp descending
            sorted_reviews = sorted(
                st.session_state["_mock_reviews"], 
                key=lambda x: x["timestamp"], 
                reverse=True
            )
            return sorted_reviews[:limit]
        except Exception:
            return []

    def delete(self, timestamp: Any) -> bool:
        """Delete a review matching the exact timestamp."""
        # Convert timestamp to standard datetime if it is ISO string
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            except ValueError:
                pass

        if self.collection:
            try:
                result = self.collection.delete_one({"timestamp": timestamp})
                return result.deleted_count > 0
            except Exception:
                pass

        try:
            import streamlit as st
            if "_mock_reviews" in st.session_state:
                initial_len = len(st.session_state["_mock_reviews"])
                st.session_state["_mock_reviews"] = [
                    r for r in st.session_state["_mock_reviews"] 
                    if r["timestamp"] != timestamp
                ]
                return len(st.session_state["_mock_reviews"]) < initial_len
            return False
        except Exception:
            return False

