"""Train and persist the TF-IDF + Logistic Regression sentiment pipeline."""
import argparse
import pickle
from pathlib import Path
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from preprocessing.text_preprocessing import preprocess_text


def train(dataset: str, model_dir: str = "model") -> None:
    df = pd.read_csv(dataset).dropna(subset=["text", "sentiment"])
    df["clean_text"] = df["text"].map(preprocess_text)
    df["label"] = df["sentiment"].astype(str).str.strip().str.title()
    df = df[df["label"].isin(["Positive", "Negative", "Neutral"])]
    x_train, x_test, y_train, y_test = train_test_split(df.clean_text, df.label, test_size=.2, random_state=42, stratify=df.label)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, max_features=15000)
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    model.fit(vectorizer.fit_transform(x_train), y_train)
    print(classification_report(y_test, model.predict(vectorizer.transform(x_test)), zero_division=0))
    out = Path(model_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "sentiment_model.pkl").open("wb") as f: pickle.dump(model, f)
    with (out / "tfidf_vectorizer.pkl").open("wb") as f: pickle.dump(vectorizer, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", default="dataset/SD_Dataset - Sheet1.csv")
    args = parser.parse_args(); train(args.dataset)
