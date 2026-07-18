# SentiFlow AI — Enterprise Sentiment Intelligence Platform

SentiFlow AI is a cognitive customer-feedback analytics platform that performs real-time sentiment classification, grammar correction, slang normalization, and emotion classification. It compiles actionable business insights, visualizes distributions and timeline trends, and generates multi-format reports (PDF, CSV, Excel).

---

## ⚡ Key Features

1. **Cognitive Sentiment Analysis**: Multi-class classification (Positive, Negative, Neutral) with confidence metrics and probability breakdowns.
2. **Dynamic Emotion Classification**: Extracts underlying customer emotional states (e.g. *happy, excited, angry, frustrated, sad, neutral*).
3. **Advanced Voice Review Pipeline**:
   $$\text{Voice Recording (st.audio\_input)} \longrightarrow \text{Speech-to-Text} \longrightarrow \text{Grammar Correction} \longrightarrow \text{Slang Map Normalization} \longrightarrow \text{NLP Cleaning} \longrightarrow \text{Inference} \longrightarrow \text{MongoDB}$$
4. **Context & Category Diagnostics**: Detects customer feedback domains (e.g. *food, hotel, app, service*) and extracts pain points (errors, delays, support).
5. **Interactive Analytics Dashboard**: Beautiful metric cards, interactive Altair donuts/bar charts, confidence histograms, time trends (daily/weekly/monthly), and automated word clouds.
6. **Business Insights Center**: Automatic compilation of Customer Satisfaction (CSAT) scores, Brand Reputation Index, and priority suggestions.
7. **Structured Review History**: Full history query registry with keywords, sentiments, emotions, and date filtering, pagination, and individual record deletion.
8. **Multi-Format Export Engine**: Immediate downloads of filtered records as CSV, formatted Excel files (via `openpyxl`), and executive PDF reports (compiled with `reportlab`).
9. **Resilient Local Sandbox mode**: Auto-detects MongoDB availability and switches gracefully to an in-memory session-state storage, ensuring zero feature loss in local testing.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit, Custom HTML5/CSS3 (theme-adaptive styles)
- **Data Visualizations**: Altair, Matplotlib, Wordcloud
- **Natural Language Processing**: NLTK (Stopword filtering), SpeechRecognition (Google API Fallback), Faster-Whisper
- **Machine Learning**: Scikit-Learn (TF-IDF Vectorizer + Logistic Regression)
- **Database Persistence**: MongoDB, PyMongo (with st.session_state mockup database fallback)
- **Reporting Engine**: ReportLab (PDF compiler), Pandas (Dataframes & CSV), openpyxl (Excel engine)
- **Packaging/Variables**: Python-Dotenv, Pickle

---

## 📂 Folder Structure

```text
Sentiment_Analysis_Streamlit/
│
├── .streamlit/                # Streamlit configuration settings
│   └── config.toml
│
├── database/                  # Database persistence layer
│   ├── __init__.py
│   └── mongodb_connection.py  # MongoDB client & session-state mock fallback repository
│
├── dataset/                   # Training dataset sheet
│   └── SD_Dataset - Sheet1.csv
│
├── model/                     # Serialized scikit-learn models
│   ├── sentiment_model.pkl    # Trained Logistic Regression classifier
│   └── tfidf_vectorizer.pkl   # Fit TF-IDF text features extractor
│
├── preprocessing/             # Text cleaning pipeline
│   ├── __init__.py
│   └── text_preprocessing.py  # Slang mapping, cleaning, NLTK analytics tokenizer
│
├── services/                  # Business logic services
│   ├── __init__.py
│   ├── analyzer.py            # Sentiment classification, explanations & recommendation maps
│   └── transcription.py       # Rule-based grammar and contraction correction
│
├── utils/                     # Utility helpers
│   └── helper_functions.py    # PDF and Excel report compilation engines
│
├── tests/                     # Preprocessing verification tests
│   └── test_preprocessing.py
│
├── training/                  # Model training scripts
│   └── train_model.py
│
├── .env.example               # Template environment configuration file
├── requirements.txt           # Package dependencies manifest
├── app.py                     # Primary Streamlit platform application
└── README.md                  # System Documentation
```

---

## 🚀 Setup & Installation

### 1. Prerequisite Requirements
Ensure Python 3.10+ is installed on your operating system.

### 2. Install Package Dependencies
Clone the repository, initialize a virtual environment, and install:
```bash
# Clone or navigate to the directory
cd Sentiment_Analysis_Streamlit

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Install required libraries
pip install -r requirements.txt
```

### 3. Setup Environment variables (Optional for MongoDB Atlas)
Rename `.env.example` to `.env` and fill in your connection credentials. If omitted, the application runs in **Local Sandbox Mode** automatically:
```env
MONGO_URI=mongodb+srv://<username>:<password>@cluster.mongodb.net/?retryWrites=true&w=majority
MONGO_DATABASE=sentiflow
```

### 4. Train/Prepare the ML Classifier
Ensure the TF-IDF and Model pickles exist in `model/`. If they are missing or if you update the training data sheet, retrain them:
```bash
python training/train_model.py
```

### 5. Launch the Platform App
```bash
streamlit run app.py
```

---

## 📸 Interface Screenshots

*Below are placeholders representing the SentiFlow AI dashboard components. Take screenshots of your running app and place them in an assets folder:*

### 💬 Feedback Analysis & Output Badges
![Feedback Analysis Output Placeholder](https://raw.githubusercontent.com/google/images/main/placeholder_analysis.png)

### 📊 Real-Time Analytics Dashboard Charts
![Analytics Dashboard Charts Placeholder](https://raw.githubusercontent.com/google/images/main/placeholder_dashboard.png)

### 📈 Business Intelligence & Suggestions
![Business Insights Dashboard Placeholder](https://raw.githubusercontent.com/google/images/main/placeholder_insights.png)

### 🕒 Paginated History & Date Filter Queries
![Paginated History Search Placeholder](https://raw.githubusercontent.com/google/images/main/placeholder_history.png)

---

## 🔮 Future Improvements

1. **Transformer Inference integration**: Transition from Logistic Regression to RoBERTa or DistilBERT models for increased sentiment categorization nuances.
2. **Fully-Local LLM Explanations**: Incorporate Ollama or Llama-cpp to draft conversational explanations instead of keyword triggers.
3. **Advanced User Authentication**: Add Auth0 or Streamlit-authenticator roles for administrative and guest report views.
4. **Real-time Alert Notifications**: Email/Slack notifications when Reputation Indexes fall below custom thresholds.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author
Developed and designed with ❤️ by **Muskaan Sharma**.
