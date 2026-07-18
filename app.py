"""SentiFlow AI — Cognitive customer sentiment intelligence dashboard."""
from __future__ import annotations

import io
import json
import os
import re
import tempfile
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import altair as alt

from config import settings
from database.mongodb_connection import ReviewRepository
from services.analyzer import SentimentAnalyzer
from services.transcription import polish_transcript, correct_grammar
from preprocessing.text_preprocessing import preprocess_text, clean_text_for_analytics
from utils.helper_functions import generate_excel_report, generate_pdf_report

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

# Streamlit App Configurations
st.set_page_config(
    page_title="SentiFlow AI — Sentiment Intelligence Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium UI Aesthetics
css_path = Path("static/styles.css")
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


@st.cache_resource
def get_services() -> tuple[SentimentAnalyzer, ReviewRepository]:
    """Cache the loaded ML model and database repository across runs."""
    analyzer = SentimentAnalyzer(settings.model_path, settings.vectorizer_path)
    # Ensure local directory folders are created if missing
    os.makedirs(os.path.dirname(settings.model_path), exist_ok=True)
    repository = ReviewRepository(settings.mongo_uri, settings.mongo_database)
    return analyzer, repository


def voice_to_text(audio: Any) -> str:
    """Transcribe recorded WAV audio with local Whisper or speech_recognition fallback."""
    # Attempt local faster-whisper transcription
    try:
        from faster_whisper import WhisperModel
        suffix = Path(audio.name).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp.write(audio.getvalue())
            temp_path = temp.name
        model = WhisperModel("base", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(temp_path, vad_filter=True)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        Path(temp_path).unlink(missing_ok=True)
        if text:
            return text
    except Exception:
        pass

    # Fallback to speech_recognition (Google Web Speech API, small & free)
    try:
        import speech_recognition as sr
        r = sr.Recognizer()
        audio_file = io.BytesIO(audio.getvalue())
        with sr.AudioFile(audio_file) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            return text.strip()
    except Exception as error:
        raise RuntimeError(
            f"Speech transcription failed: {error}. Please ensure the recorded audio is clear and silent gaps are minimal."
        )


def analyze_feedback(
    text: str, source: str = "Web", original_speech: str | None = None
) -> tuple[dict[str, Any], bool]:
    """Run text through transcription, grammar, slang correction, and ML models."""
    analyzer, repository = get_services()
    
    # 1. Grammar Correction
    corrected_text = polish_transcript(text)
    
    # 2. ML Sentiment & Emotion Classification
    result = analyzer.analyze(corrected_text)
    
    # 3. Enhanced Schema Mapping
    result["source"] = source
    result["corrected_text"] = corrected_text
    if original_speech:
        result["original_speech_text"] = original_speech
        
    # 4. Save to Repository
    saved = repository.save(result)
    return result, saved


def generate_wordcloud_chart(reviews: list[dict[str, Any]], sentiment: str | None = None) -> Any:
    """Generate Matplotlib wordcloud image for the selected reviews list."""
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if sentiment:
        texts = [r.get("cleaned_review", "") for r in reviews if r.get("sentiment") == sentiment]
    else:
        texts = [r.get("cleaned_review", "") for r in reviews]
        
    text_corpus = " ".join(texts).strip()
    if not text_corpus:
        return None
        
    colormap = "viridis"
    if sentiment == "Positive":
        colormap = "summer"
    elif sentiment == "Negative":
        colormap = "autumn"
        
    wc = WordCloud(
        width=800, 
        height=380, 
        background_color="white", 
        colormap=colormap, 
        max_words=80,
        contour_width=1,
        contour_color="#e2e8f0"
    ).generate(text_corpus)
    
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def main() -> None:
    analyzer, repository = get_services()
    
    # Ensure model is initialized
    try:
        analyzer.load()
    except FileNotFoundError:
        st.error(
            f"Sentiment classification models were not found at `{settings.model_path}`. "
            "Please run `python training/train_model.py` to train and serialize the models first."
        )
        return

    # Sidebar Navigation Menu
    brand_path = Path("static/brand.html")
    if brand_path.exists():
        with open(brand_path, "r", encoding="utf-8") as f:
            st.sidebar.markdown(f.read(), unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    # Render MongoDB status
    if repository.connected:
        st.sidebar.markdown("<div style='margin-bottom:15px; margin-top:-5px;'><span class='status-badge connected'>● Connected</span></div>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown("<div style='margin-bottom:15px; margin-top:-5px;'><span class='status-badge sandbox'>● Local Sandbox</span></div>", unsafe_allow_html=True)

    page = st.sidebar.radio(
        "Navigation",
        [
            "💬 Analyze Feedback",
            "📊 Analytics Dashboard",
            "📈 Business Insights",
            "🕒 Review History & Search"
        ]
    )
    
    st.sidebar.markdown("---")
    footer_path = Path("static/footer.html")
    if footer_path.exists():
        with open(footer_path, "r", encoding="utf-8") as f:
            st.sidebar.markdown(f.read(), unsafe_allow_html=True)

    # 1. ANALYZE FEEDBACK PAGE
    if page == "💬 Analyze Feedback":
        st.markdown("<h1 class='app-title'>Analyze Feedback</h1>", unsafe_allow_html=True)
        st.markdown("<p class='app-subtitle'>Real-Time Sentiment Prediction, Grammar Polishing & Emotion Extraction</p>", unsafe_allow_html=True)
        
        # Calculate metric cards summary from database
        stats_reviews = repository.recent(500)
        total_cnt = len(stats_reviews)
        pos_cnt = sum(1 for r in stats_reviews if r.get("sentiment") == "Positive")
        neg_cnt = sum(1 for r in stats_reviews if r.get("sentiment") == "Negative")
        avg_conf = round(sum(r.get("confidence", 0.0) for r in stats_reviews) / total_cnt, 1) if total_cnt > 0 else 0.0
        
        stat_c1, stat_c2, stat_c3, stat_c4 = st.columns(4)
        with stat_c1:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Total Reviews</div><div class='metric-val'>{total_cnt}</div><div class='metric-sub'>From database history</div></div>", unsafe_allow_html=True)
        with stat_c2:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Positive</div><div class='metric-val' style='color:#22c55e;'>{pos_cnt}</div><div class='metric-sub'>{round(pos_cnt/total_cnt*100,1) if total_cnt>0 else 0.0}% satisfaction</div></div>", unsafe_allow_html=True)
        with stat_c3:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Negative</div><div class='metric-val' style='color:#ef4444;'>{neg_cnt}</div><div class='metric-sub'>{round(neg_cnt/total_cnt*100,1) if total_cnt>0 else 0.0}% friction</div></div>", unsafe_allow_html=True)
        with stat_c4:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Avg Confidence</div><div class='metric-val' style='color:#3b82f6;'>{avg_conf}%</div><div class='metric-sub'>Model certainty rate</div></div>", unsafe_allow_html=True)
            
        st.write("") # Spacing
        
        # UI Columns
        input_col, output_col = st.columns([1, 1], gap="large")
        
        with input_col:
            st.subheader("Input Text or Voice Review", anchor=False)
            
            # Input format tab selection
            input_mode = st.tabs(["📝 Manual Text", "🎙️ Voice Input", "📂 Text File Upload"])
            
            text_to_analyze = ""
            speech_text = None
            source_medium = "Web"
            
            # Tab 1: Manual Text
            with input_mode[0]:
                manual_text = st.text_area(
                    "Customer Feedback Review",
                    placeholder="Enter customer feedback or product review here...",
                    height=130,
                    key="manual_review_area"
                )
                
                # Dynamic chips below the textarea
                st.caption("💡 Sample Quick Reviews:")
                chip_c1, chip_c2, chip_c3 = st.columns(3)
                with chip_c1:
                    if st.button("Great customer service!", use_container_width=True, key="chip_great"):
                        st.session_state["manual_review_area"] = "Great customer service!"
                        st.rerun()
                with chip_c2:
                    if st.button("Worst experience ever.", use_container_width=True, key="chip_worst"):
                        st.session_state["manual_review_area"] = "Worst experience ever."
                        st.rerun()
                with chip_c3:
                    if st.button("Average quality product.", use_container_width=True, key="chip_avg"):
                        st.session_state["manual_review_area"] = "Average quality product."
                        st.rerun()
                        
                if manual_text.strip():
                    text_to_analyze = manual_text.strip()
                    source_medium = "Web"
            
            # Tab 2: Voice Input (Voice Review Pipeline)
            with input_mode[1]:
                st.write("Record review voice comments directly:")
                recorded_audio = st.audio_input("Record feedback comments")
                
                if recorded_audio is not None:
                    with st.spinner("Executing Speech-to-Text transcription..."):
                        try:
                            speech_text = voice_to_text(recorded_audio)
                            if speech_text:
                                st.info(f"🎙️ **Transcribed Audio Speech:** *\"{speech_text}\"*")
                                # Apply Grammar correction to voice
                                text_to_analyze = speech_text
                                source_medium = "Voice"
                            else:
                                st.warning("No clear voice signals detected. Please record again.")
                        except Exception as e:
                            st.error(str(e))
            
            # Tab 3: Text File Upload
            with input_mode[2]:
                uploaded_file = st.file_uploader("Upload Review Text file (.txt)", type=["txt"])
                if uploaded_file is not None:
                    try:
                        file_text = uploaded_file.getvalue().decode("utf-8", errors="replace").strip()
                        if file_text:
                            text_to_analyze = file_text
                            source_medium = "Web"
                            st.success("File uploaded successfully!")
                        else:
                            st.warning("Uploaded file is empty.")
                    except Exception as e:
                        st.error(f"Error parsing file: {e}")
            
            st.write("")
            submit_btn = st.button("✨ Analyze Customer Feedback", type="primary", use_container_width=True)
        
        with output_col:
            st.subheader("Analysis Insights", anchor=False)
            
            if submit_btn and text_to_analyze:
                with st.spinner("Analyzing Sentiment..."):
                    try:
                        result, saved = analyze_feedback(
                            text_to_analyze, 
                            source=source_medium, 
                            original_speech=speech_text
                        )
                        
                        # Style sentiment output
                        sentiment = result["sentiment"]
                        confidence = result["confidence"]
                        emotion = result["emotion"].title()
                        domain = result["domain"].title()
                        
                        if sentiment == "Positive":
                            sentiment_badge = f"<span class='sentiment-badge badge-positive'>😊 Positive</span>"
                            card_class = "positive"
                        elif sentiment == "Negative":
                            sentiment_badge = f"<span class='sentiment-badge badge-negative'>😡 Negative</span>"
                            card_class = "negative"
                        else:
                            sentiment_badge = f"<span class='sentiment-badge badge-neutral'>😐 Neutral</span>"
                            card_class = "neutral"
                            
                        # Layout output in beautifully styled UI components
                        st.markdown(
                            f"<div style='padding: 1.5rem; border-radius: 16px; margin-bottom: 1rem; border: 1px solid rgba(128, 128, 128, 0.15);'>"
                            f"<h4>Predicted Sentiment: {sentiment_badge}</h4>"
                            f"<p style='margin: 0.5rem 0;'><strong>Confidence Level:</strong> {confidence}%</p>"
                            f"<p style='margin: 0.5rem 0;'><strong>Emotion:</strong> {emotion}</p>"
                            f"<p style='margin: 0.5rem 0;'><strong>Category Domain:</strong> {domain}</p>"
                            f"</div>", 
                            unsafe_allow_html=True
                        )
                        
                        # Probability scores
                        st.markdown("**Probability Class breakdown:**")
                        probs = result.get("probabilities", {})
                        for label, prob in probs.items():
                            st.caption(f"{label} ({prob}%)")
                            st.progress(prob / 100.0)
                            
                        # Pipeline text stages
                        st.markdown("---")
                        st.markdown("**Text Transformation Flow:**")
                        st.write(f"✍️ **Corrected & Polished Review:** *\"{result.get('corrected_text')}\"*")
                        st.write(f"🧹 **Cleaned Review Input:** `{result.get('processed_text')}`")
                        
                        # Explanation
                        st.markdown("---")
                        st.markdown("**Explanation & Diagnostics:**")
                        st.write(result.get("explanation"))
                        
                        # Recommendations
                        st.markdown("---")
                        st.markdown("**Business Recommendations:**")
                        for rec in result.get("recommendation", []):
                            st.markdown(f"• {rec}")
                            
                        if saved:
                            st.toast("Feedback analysis persisted successfully!", icon="💾")
                        else:
                            st.toast("Saved to fallback local session-state cache.", icon="⚡")
                            
                    except Exception as e:
                        st.error(f"Prediction Pipeline failed: {e}")
            elif submit_btn:
                st.warning("Please provide a text or voice input review to run prediction.")
            else:
                st.markdown(
                    "<div class='placeholder-card'>"
                    "<span class='placeholder-icon'>📊</span>"
                    "<div class='placeholder-title'>Prediction results will appear here.</div>"
                    "<div class='placeholder-desc'>Please input a customer feedback review or record speech comments, then click Analyze.</div>"
                    "</div>",
                    unsafe_allow_html=True
                )

    # 2. ANALYTICS DASHBOARD PAGE
    elif page == "📊 Analytics Dashboard":
        st.markdown("<h1 class='app-title'>Analytics Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<p class='app-subtitle'>Real-Time Aggregated Metrics, Trends, and Sentiment Analysis Distributions</p>", unsafe_allow_html=True)
        
        # Load recent reviews
        reviews = repository.recent(500)
        
        if not reviews:
            st.info("No reviews analyzed yet. Analyze feedback on the 'Analyze Feedback' tab to populate this dashboard.", icon="📊")
            return
            
        df = pd.DataFrame(reviews)
        total_count = len(df)
        
        pos_df = df[df["sentiment"] == "Positive"]
        neg_df = df[df["sentiment"] == "Negative"]
        neu_df = df[df["sentiment"] == "Neutral"]
        
        pos_count = len(pos_df)
        neg_count = len(neg_df)
        neu_count = len(neu_df)
        
        pos_pct = round((pos_count / total_count * 100), 1) if total_count > 0 else 0
        neg_pct = round((neg_count / total_count * 100), 1) if total_count > 0 else 0
        neu_pct = round((neu_count / total_count * 100), 1) if total_count > 0 else 0
        
        # Stats Columns
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>Total Reviews</div>"
                f"<div class='metric-val'>{total_count}</div>"
                f"<div class='metric-sub'>Aggregated Database Count</div>"
                f"</div>", 
                unsafe_allow_html=True
            )
        with m2:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>Positive Feedback</div>"
                f"<div class='metric-val' style='color:#16a34a;'>{pos_count}</div>"
                f"<div class='metric-sub'>{pos_pct}% of total</div>"
                f"</div>", 
                unsafe_allow_html=True
            )
        with m3:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>Negative Feedback</div>"
                f"<div class='metric-val' style='color:#dc2626;'>{neg_count}</div>"
                f"<div class='metric-sub'>{neg_pct}% of total</div>"
                f"</div>", 
                unsafe_allow_html=True
            )
        with m4:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>Neutral Feedback</div>"
                f"<div class='metric-val' style='color:#d97706;'>{neu_count}</div>"
                f"<div class='metric-sub'>{neu_pct}% of total</div>"
                f"</div>", 
                unsafe_allow_html=True
            )
            
        st.write("")
        
        # Section 2: Distribution Charts
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.subheader("😊 Sentiment Breakdown", anchor=False)
            df_sent = pd.DataFrame({'Sentiment': ['Positive', 'Neutral', 'Negative'], 'Count': [pos_count, neu_count, neg_count]})
            # Donut chart
            donut = alt.Chart(df_sent).mark_arc(innerRadius=65, stroke="#fff").encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(field="Sentiment", type="nominal", scale=alt.Scale(domain=['Positive', 'Neutral', 'Negative'], range=['#22c55e', '#f59e0b', '#ef4444'])),
                tooltip=['Sentiment', 'Count']
            ).properties(height=300)
            st.altair_chart(donut, use_container_width=True)
            
        with col_c2:
            st.subheader("🎭 Emotion Distribution", anchor=False)
            df_emo = pd.DataFrame(df['emotion'].value_counts().reset_index())
            df_emo.columns = ['Emotion', 'Count']
            bar_emo = alt.Chart(df_emo).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
                x=alt.X('Emotion:N', sort='-y', title="Emotion Class"),
                y=alt.Y('Count:Q', title="Occurrences"),
                color=alt.Color(field="Emotion", type="nominal", legend=None),
                tooltip=['Emotion', 'Count']
            ).properties(height=300)
            st.altair_chart(bar_emo, use_container_width=True)
            
        # Section 3: Trends Over Time
        st.subheader("📈 Feedback Trends Over Time", anchor=False)
        
        trend_range = st.radio("Trend Scope Selection", ["Daily", "Weekly", "Monthly"], horizontal=True)
        df['datetime'] = pd.to_datetime(df['timestamp'])
        
        if trend_range == "Daily":
            df_trend = df.groupby(df['datetime'].dt.date).size().reset_index(name='Reviews Count')
            df_trend.columns = ['Date', 'Reviews Count']
            line_chart = alt.Chart(df_trend).mark_line(point=True, color='#0284c7').encode(
                x=alt.X('Date:T', title="Timeline"),
                y=alt.Y('Reviews Count:Q', title="Total Feedback Reviews"),
                tooltip=['Date:T', 'Reviews Count:Q']
            ).properties(height=260)
        elif trend_range == "Weekly":
            df_trend = df.groupby(df['datetime'].dt.to_period('W').astype(str)).size().reset_index(name='Reviews Count')
            df_trend.columns = ['Week', 'Reviews Count']
            line_chart = alt.Chart(df_trend).mark_bar(color='#6366f1').encode(
                x=alt.X('Week:N', title="Weekly Intervals"),
                y=alt.Y('Reviews Count:Q', title="Total Feedback Reviews"),
                tooltip=['Week:N', 'Reviews Count:Q']
            ).properties(height=260)
        else:
            df_trend = df.groupby(df['datetime'].dt.to_period('M').astype(str)).size().reset_index(name='Reviews Count')
            df_trend.columns = ['Month', 'Reviews Count']
            line_chart = alt.Chart(df_trend).mark_bar(color='#0d9488').encode(
                x=alt.X('Month:N', title="Monthly Intervals"),
                y=alt.Y('Reviews Count:Q', title="Total Feedback Reviews"),
                tooltip=['Month:N', 'Reviews Count:Q']
            ).properties(height=260)
            
        st.altair_chart(line_chart, use_container_width=True)
        
        # Section 4: Confidence Distribution and Word Clouds
        col_c3, col_c4 = st.columns(2)
        
        with col_c3:
            st.subheader("🎯 Confidence Score Histogram", anchor=False)
            hist = alt.Chart(df).mark_bar(color='#8b5cf6').encode(
                x=alt.X("confidence:Q", bin=alt.Bin(maxbins=12), title="ML Classifier Confidence (%)"),
                y=alt.Y("count():Q", title="Volume"),
                tooltip=["count()"]
            ).properties(height=300)
            st.altair_chart(hist, use_container_width=True)
            
        with col_c4:
            st.subheader("☁️ Word Cloud Word-Frequencies", anchor=False)
            wc_sentiment = st.selectbox("Filter Word Cloud by Sentiment", ["All", "Positive", "Negative"])
            
            wc_filter = None if wc_sentiment == "All" else wc_sentiment
            fig_wc = generate_wordcloud_chart(reviews, wc_filter)
            
            if fig_wc is not None:
                st.pyplot(fig_wc)
            else:
                st.info("Not enough token keywords available to compile a Word Cloud image.")

    # 3. BUSINESS INSIGHTS PAGE
    elif page == "📈 Business Insights":
        st.markdown("<h1 class='app-title'>Business Insights</h1>", unsafe_allow_html=True)
        st.markdown("<p class='app-subtitle'>Automated KPIs, Brand Reputation Indexing, and Actionable Recommendations</p>", unsafe_allow_html=True)
        
        reviews = repository.recent(500)
        
        if not reviews:
            st.info("Add user feedback review entries in the 'Analyze Feedback' tab to compile business intelligence KPI indices.", icon="📈")
            return
            
        df = pd.DataFrame(reviews)
        total = len(df)
        positive = sum(1 for r in reviews if r.get('sentiment') == 'Positive')
        negative = sum(1 for r in reviews if r.get('sentiment') == 'Negative')
        neutral = sum(1 for r in reviews if r.get('sentiment') == 'Neutral')
        
        # Customer Satisfaction Score (CSAT)
        csat = round((positive / total * 100), 1) if total > 0 else 0.0
        # Brand Reputation Score (indexed scale 0-100)
        reputation = round(((positive - negative) / total * 50 + 50), 1) if total > 0 else 0.0
        
        # Dominant customer emotion
        emotions_series = df["emotion"].value_counts()
        dominant_mood = emotions_series.idxmax().title() if not emotions_series.empty else "N/A"
        
        # Diagnostic KPIs display
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.markdown(
                f"<div class='metric-card' style='text-align: center;'>"
                f"<div class='metric-label'>Customer Satisfaction (CSAT)</div>"
                f"<div class='metric-val' style='color:#4f46e5; font-size:2.4rem;'>{csat}%</div>"
                f"<div class='metric-sub'>Positive Sentiment share</div>"
                f"</div>", 
                unsafe_allow_html=True
            )
            
        with c2:
            reputation_color = "#16a34a" if reputation >= 70 else ("#dc2626" if reputation < 50 else "#d97706")
            st.markdown(
                f"<div class='metric-card' style='text-align: center;'>"
                f"<div class='metric-label'>Brand Reputation Index</div>"
                f"<div class='metric-val' style='color:{reputation_color}; font-size:2.4rem;'>{reputation} / 100</div>"
                f"<div class='metric-sub'>Net Sentiment score</div>"
                f"</div>", 
                unsafe_allow_html=True
            )
            
        with c3:
            st.markdown(
                f"<div class='metric-card' style='text-align: center;'>"
                f"<div class='metric-label'>Dominant Customer Mood</div>"
                f"<div class='metric-val' style='color:#0d9488; font-size:2.4rem;'>{dominant_mood}</div>"
                f"<div class='metric-sub'>Highest frequency emotion</div>"
                f"</div>", 
                unsafe_allow_html=True
            )
            
        st.write("")
        
        # Split analysis details
        left, right = st.columns([3, 2], gap="large")
        
        with left:
            st.subheader("💡 Key Insights Summary", anchor=False)
            
            insights_list = []
            
            # Reputation level diagnostics
            if reputation >= 75:
                insights_list.append("🌟 **Stellar Reputation:** Brand perception is highly positive. Maintain quality standards and feature testimonials prominently.")
            elif reputation >= 50:
                insights_list.append("⚖️ **Balanced Reputation:** Users are generally content, but there are areas of friction. Track negative spikes.")
            else:
                insights_list.append("⚠️ **Critical Reputation Warning:** High negative reviews. Immediate service/operational review is highly advised.")
                
            # Domain details
            domain_series = df["domain"].value_counts()
            if not domain_series.empty:
                top_domain = domain_series.idxmax().title()
                insights_list.append(f"📦 **Category Focus:** Most customer reviews target the **{top_domain}** category. Direct targeted optimization efforts there.")
                
            # Frequent problem keywords
            all_problems = []
            for r in reviews:
                all_problems.extend(r.get("problems", []))
            if all_problems:
                from collections import Counter
                top_probs = Counter(all_problems).most_common(3)
                probs_format = ", ".join([f"`{p[0]}` (volume: {p[1]})" for p in top_probs])
                insights_list.append(f"🛑 **Critical Friction Points:** The primary pain points detected in reviews are: {probs_format}.")
            else:
                insights_list.append("✅ **Clean Operations:** No recurring negative issue events (like errors, slow support, crashes) were detected.")
                
            for insight in insights_list:
                st.markdown(f"• {insight}")
                
        with right:
            st.subheader("📋 Improvement Suggestions", anchor=False)
            
            # Generate actionable business suggestions
            suggestions = []
            
            if csat < 80:
                suggestions.append("Enact proactive customer recovery steps: contact negative review submitters within 24 hours.")
            
            # Domain suggestions based on feedback volume
            domain_series = df["domain"].value_counts()
            if not domain_series.empty:
                top_d = domain_series.idxmax()
                if top_d == "app":
                    suggestions.append("Optimize application reliability: investigate UI glitches and checkout speed bottlenecks.")
                elif top_d == "service":
                    suggestions.append("Enhance service touchpoints: review customer support agent response times.")
                elif top_d == "food":
                    suggestions.append("Audit food delivery delays and packaging quality to maintain standards.")
                elif top_d == "hotel":
                    suggestions.append("Review housekeeping schedules and guest onboarding protocols.")
                    
            if suggestions:
                for idx, sug in enumerate(suggestions, start=1):
                    st.write(f"**{idx}.** {sug}")
            else:
                st.write("No urgent corrective action required. Continue monitoring reviews to sustain excellence.")

    # 4. REVIEW HISTORY PAGE (Search, pagination, filters, deletion, exports)
    elif page == "🕒 Review History & Search":
        st.markdown("<h1 class='app-title'>Review History & Search</h1>", unsafe_allow_html=True)
        st.markdown("<p class='app-subtitle'>Structured Database Registry, Search Query Indexing, and Report Exports</p>", unsafe_allow_html=True)
        
        # Load raw data from db
        reviews = repository.recent(1000)
        
        if not reviews:
            st.info("History registry is currently empty. Analyze feedback reviews on the Home page first.", icon="🕒")
            return
            
        df = pd.DataFrame(reviews)
        
        # Filtering Expanders
        st.subheader("🔍 Search Query Filters", anchor=False)
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        
        with f_col1:
            keyword_search = st.text_input("Search Keyword", placeholder="Enter review text keyword...")
        with f_col2:
            sentiment_options = ["All", "Positive", "Negative", "Neutral"]
            sentiment_select = st.selectbox("Sentiment", sentiment_options)
        with f_col3:
            # Gather unique emotions
            emotions = ["All"] + sorted(list(df["emotion"].unique()))
            emotion_select = st.selectbox("Emotion", emotions)
        with f_col4:
            date_filter = st.date_input(
                "Filter Date Range",
                value=(date.today() - timedelta(days=90), date.today() + timedelta(days=2))
            )
            
        # Process filter query
        filtered_df = df.copy()
        
        # Text matching
        if keyword_search.strip():
            kw = keyword_search.strip().lower()
            filtered_df = filtered_df[
                filtered_df["review"].str.lower().str.contains(kw) |
                filtered_df["corrected_review"].str.lower().str.contains(kw)
            ]
            
        # Sentiment matching
        if sentiment_select != "All":
            filtered_df = filtered_df[filtered_df["sentiment"] == sentiment_select]
            
        # Emotion matching
        if emotion_select != "All":
            filtered_df = filtered_df[filtered_df["emotion"] == emotion_select]
            
        # Date range matching
        if isinstance(date_filter, tuple) and len(date_filter) == 2:
            start_dt, end_dt = date_filter
            filtered_df["temp_date"] = pd.to_datetime(filtered_df["timestamp"]).dt.date
            filtered_df = filtered_df[
                (filtered_df["temp_date"] >= start_dt) & 
                (filtered_df["temp_date"] <= end_dt)
            ]
            filtered_df = filtered_df.drop(columns=["temp_date"])
            
        # Export Actions Columns
        st.markdown("---")
        exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)
        
        filtered_list = filtered_df.to_dict(orient="records")
        
        with exp_col1:
            st.write(f"📊 **Matched Reviews:** `{len(filtered_df)}` rows")
            
        with exp_col2:
            # Download CSV
            csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Export Match to CSV",
                csv_bytes,
                "sentiflow_feedback_match.csv",
                "text/csv",
                use_container_width=True,
                icon="📄"
            )
            
        with exp_col3:
            # Download Excel
            try:
                excel_bytes = generate_excel_report(filtered_list)
                st.download_button(
                    "⬇️ Export Match to Excel",
                    excel_bytes,
                    "sentiflow_feedback_match.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    icon="📊"
                )
            except Exception as e:
                st.button("Excel Export Unavailable", disabled=True, use_container_width=True)
                st.caption(f"Excel error: {e}")
                
        with exp_col4:
            # Download PDF
            try:
                pdf_bytes = generate_pdf_report(filtered_list)
                st.download_button(
                    "⬇️ Export Match to PDF",
                    pdf_bytes,
                    "sentiflow_feedback_report.pdf",
                    "application/pdf",
                    use_container_width=True,
                    icon="📕"
                )
            except Exception as e:
                st.button("PDF Export Unavailable", disabled=True, use_container_width=True)
                st.caption(f"PDF error: {e}")
                
        st.write("")
        
        # Paginated Reviews Display
        if filtered_df.empty:
            st.info("No reviews match your filter query. Reset search parameters above.")
            return
            
        REVIEWS_PER_PAGE = 8
        total_matched = len(filtered_df)
        total_pages = max(1, (total_matched + REVIEWS_PER_PAGE - 1) // REVIEWS_PER_PAGE)
        
        st.session_state.setdefault("history_page", 1)
        # Cap page index
        st.session_state["history_page"] = min(st.session_state["history_page"], total_pages)
        
        page_num = st.session_state["history_page"]
        
        # Slice DataFrame
        start_idx = (page_num - 1) * REVIEWS_PER_PAGE
        end_idx = min(start_idx + REVIEWS_PER_PAGE, total_matched)
        page_df = filtered_df.iloc[start_idx:end_idx]
        
        # Page Controls
        p_c1, p_c2, p_c3 = st.columns([1, 4, 1])
        with p_c1:
            if st.button("⬅️ Previous", disabled=(page_num <= 1), use_container_width=True):
                st.session_state["history_page"] = max(1, page_num - 1)
                st.rerun()
        with p_c2:
            st.markdown(
                f"<div style='text-align: center; line-height: 38px; font-weight: 600;'>"
                f"Page {page_num} of {total_pages} (Showing {start_idx + 1} - {end_idx} of {total_matched} matches)"
                f"</div>", 
                unsafe_allow_html=True
            )
        with p_c3:
            if st.button("Next ➡️", disabled=(page_num >= total_pages), use_container_width=True):
                st.session_state["history_page"] = min(total_pages, page_num + 1)
                st.rerun()
                
        st.write("")
        
        # Render lists as cards/expanders for interactive detailed views and delete capability
        for idx, row in page_df.iterrows():
            ts = row.get("timestamp")
            dt_display = ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts, "strftime") else str(ts)[:19]
            source = row.get("source", "Web")
            sentiment = row.get("sentiment", "N/A")
            confidence = row.get("confidence", 0.0)
            emotion = row.get("emotion", "N/A").title()
            domain = row.get("domain", "general").title()
            
            # Sentiment styling
            if sentiment == "Positive":
                tag = f"<span class='sentiment-badge badge-positive'>Positive ({confidence}%)</span>"
            elif sentiment == "Negative":
                tag = f"<span class='sentiment-badge badge-negative'>Negative ({confidence}%)</span>"
            else:
                tag = f"<span class='sentiment-badge badge-neutral'>Neutral ({confidence}%)</span>"
                
            header = f"📅 {dt_display} | Source: {source} | Domain: {domain} | Emotion: {emotion} | {tag}"
            
            with st.expander(header, expanded=False):
                # Text detail
                st.markdown(f"✍️ **Original Review:**")
                st.write(row.get("review"))
                
                # Check for corrected voice transcript
                if source == "Voice":
                    st.markdown(f"🎙️ **Polished Grammar Review:**")
                    st.write(row.get("corrected_review"))
                    
                st.markdown(f"🧹 **Cleaned Input Token:** `{row.get('cleaned_review')}`")
                
                # Model outputs
                st.markdown("---")
                st.markdown(f"💬 **Model Explanation:**")
                st.write(row.get("explanation", "No analysis explanation stored."))
                
                st.markdown(f"💡 **Action Suggestions:**")
                for rec in row.get("recommendation", []):
                    st.markdown(f"• {rec}")
                    
                # Delete control
                st.write("")
                d_c1, d_c2 = st.columns([5, 1])
                with d_c2:
                    del_btn = st.button("🗑️ Delete Review", key=f"del_{idx}_{ts}", type="secondary", use_container_width=True)
                    if del_btn:
                        deleted = repository.delete(ts)
                        if deleted:
                            st.toast("Feedback review deleted.", icon="🗑️")
                            st.rerun()
                        else:
                            st.error("Failed to delete review record.")


if __name__ == "__main__":
    main()