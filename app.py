"""
Streamlit Web Application: Real-Time Text Emotion Recognition
Interactive NLP interface powered by TF-IDF & Logistic Regression / Naive Bayes.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure src module is importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from src.preprocessing import clean_text

# Page Configuration
st.set_page_config(
    page_title="EmoSense | Real-Time Emotion Classifier",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #4F46E5, #9333EA, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #6B7280;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 16px;
        background: #F3F4F6;
        border: 1px solid #E5E7EB;
        text-align: center;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .emotion-emoji {
        font-size: 4rem;
        margin-bottom: 0.5rem;
    }
    .emotion-label {
        font-size: 1.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .confidence-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-top: 8px;
    }
    .sample-btn {
        margin-right: 5px;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Color and Emoji mappings for emotions
EMOTION_CONFIG = {
    'joy': {'emoji': '😊', 'color': '#10B981', 'card_bg': '#ECFDF5', 'text_color': '#065F46'},
    'sadness': {'emoji': '😢', 'color': '#3B82F6', 'card_bg': '#EFF6FF', 'text_color': '#1E40AF'},
    'anger': {'emoji': '😡', 'color': '#EF4444', 'card_bg': '#FEF2F2', 'text_color': '#991B1B'},
    'fear': {'emoji': '😨', 'color': '#8B5CF6', 'card_bg': '#F5F3FF', 'text_color': '#5B21B6'},
    'love': {'emoji': '❤️', 'color': '#EC4899', 'card_bg': '#FDF2F8', 'text_color': '#9D174D'},
    'surprise': {'emoji': '😲', 'color': '#F59E0B', 'card_bg': '#FFFBEB', 'text_color': '#92400E'}
}

SAMPLE_QUOTES = {
    "😊 Joy": "I just received the job offer I have worked toward for years, absolutely thrilled!",
    "😢 Sadness": "I feel completely lost, heartbroken and lonely today, missing my old home.",
    "😡 Anger": "This is completely intolerable and unfair, they lied directly to our faces!",
    "😨 Fear": "I have an overwhelming sense of dread about tomorrow's presentation and outcome.",
    "❤️ Love": "Being around you makes every single day brighter, I cherish you so much.",
    "😲 Surprise": "I was totally not expecting my entire team to throw me a surprise celebration!"
}


@st.cache_resource
def load_pipeline():
    model_path = os.path.join(CURRENT_DIR, "models", "emotion_model.joblib")
    vectorizer_path = os.path.join(CURRENT_DIR, "models", "tfidf_vectorizer.joblib")
    meta_path = os.path.join(CURRENT_DIR, "models", "metadata.json")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None, None

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

    return model, vectorizer, metadata


# Load model components
model, vectorizer, metadata = load_pipeline()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/brain.png", width=110)
    st.title("EmoSense NLP")
    st.caption("AI-Powered Fine-Grained Emotion Detection")
    st.markdown("---")

    st.subheader("📌 Project Overview")
    st.markdown("""
    This application analyses natural language text to detect human emotional undertones across **6 distinct emotions**:
    - **Joy** 😊 | **Sadness** 😢
    - **Anger** 😡 | **Fear** 😨
    - **Love** ❤️ | **Surprise** 😲
    """)

    st.markdown("---")
    st.subheader("⚙️ Model Pipeline")
    if metadata and "metrics" in metadata:
        st.markdown(f"**Model:** `{metadata.get('best_model', 'Logistic Regression')}`")
        st.markdown(f"**Accuracy:** `{metadata.get('best_accuracy', 0.89) * 100:.2f}%`")
        st.markdown(f"**Vectorization:** `TF-IDF (Sublinear scaling)`")
        st.markdown(f"**Dataset Samples:** `{metadata.get('dataset_stats', {}).get('total_samples', '16,000+')}`")
    else:
        st.info("Train the model via `python src/train.py` to inspect real-time metrics.")

    st.markdown("---")
    st.markdown("👨‍💻 **Developer Links**")
    st.markdown("⭐ [GitHub Repository](https://github.com) | 💼 [LinkedIn Profile](https://linkedin.com)")

# Main Header
st.markdown('<div class="main-title">🎭 EmoSense: Text Emotion Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Type any statement, tweet, or thought below to uncover its underlying emotional tone in real-time.</div>', unsafe_allow_html=True)

if model is None or vectorizer is None:
    st.warning("⚠️ Model artifacts not found! Please run the training script: `python src/train.py`.")
    st.stop()

# Tab Layout
tab_live, tab_batch, tab_analytics = st.tabs([
    "🎯 Live Predictor", "📂 Batch Analysis", "📊 Model Metrics & Benchmark"
])

with tab_live:
    st.markdown("##### 💡 Try an Instant Example:")
    col_samples = st.columns(len(SAMPLE_QUOTES))
    
    # Session state for input text
    if "input_text" not in st.session_state:
        st.session_state["input_text"] = "I am incredibly grateful and excited to share this new milestone with everyone!"

    for i, (label, sample_text) in enumerate(SAMPLE_QUOTES.items()):
        if col_samples[i].button(label, use_container_width=True):
            st.session_state["input_text"] = sample_text
            st.rerun()

    user_input = st.text_area(
        "Enter text to analyze:",
        value=st.session_state["input_text"],
        height=110,
        placeholder="e.g., I've been feeling exhausted lately and wondering if things will get better..."
    )

    col_btn, col_clear = st.columns([1, 6])
    analyze_clicked = col_btn.button("🔍 Analyze Emotion", type="primary", use_container_width=True)

    if user_input.strip():
        # Preprocess text
        cleaned_input = clean_text(user_input)
        
        # Predict probabilities
        features = vectorizer.transform([cleaned_input])
        probabilities = model.predict_proba(features)[0]
        pred_idx = np.argmax(probabilities)
        
        classes = metadata.get("classes", list(model.classes_))
        pred_emotion = classes[pred_idx]
        confidence = probabilities[pred_idx] * 100

        cfg = EMOTION_CONFIG.get(pred_emotion, {'emoji': '💬', 'color': '#6366F1', 'card_bg': '#EEF2FF', 'text_color': '#312E81'})

        st.markdown("---")
        
        # Result Layout: 2 Columns (Card on left, Chart on right)
        res_col1, res_col2 = st.columns([1.2, 2.0])

        with res_col1:
            st.markdown(f"""
            <div style="background-color: {cfg['card_bg']}; border: 2px solid {cfg['color']}; border-radius: 16px; padding: 24px; text-align: center;">
                <div style="font-size: 4.5rem; line-height: 1.1;">{cfg['emoji']}</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: {cfg['text_color']}; text-transform: uppercase; margin-top: 8px;">
                    {pred_emotion}
                </div>
                <div style="background-color: {cfg['color']}; color: white; display: inline-block; padding: 6px 18px; border-radius: 20px; font-weight: 700; font-size: 1rem; margin-top: 10px;">
                    {confidence:.1f}% Confidence
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🔍 Cleaned NLP Tokens", expanded=False):
                st.write("**Original Text:**", user_input)
                st.write("**After Normalization:**", cleaned_input if cleaned_input else "(empty tokens)")

        with res_col2:
            st.markdown("##### 📈 Emotion Probability Distribution")
            df_prob = pd.DataFrame({
                'Emotion': [c.capitalize() for c in classes],
                'Probability (%)': probabilities * 100,
                'RawEmotion': classes
            }).sort_values('Probability (%)', ascending=True)

            colors = [EMOTION_CONFIG.get(e, {}).get('color', '#6366F1') for e in df_prob['RawEmotion']]

            fig = go.Figure(go.Bar(
                x=df_prob['Probability (%)'],
                y=df_prob['Emotion'],
                orientation='h',
                marker=dict(color=colors, line=dict(color='rgba(0,0,0,0.1)', width=1)),
                text=[f"{p:.1f}%" for p in df_prob['Probability (%)']],
                textposition='auto',
            ))
            fig.update_layout(
                margin=dict(l=10, r=20, t=10, b=10),
                height=240,
                xaxis=dict(title="Probability (%)", range=[0, 100]),
                yaxis=dict(title=""),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)

with tab_batch:
    st.markdown("##### 📂 Batch Emotion Processing")
    st.markdown("Paste multiple lines or upload a CSV file containing a `text` column.")

    batch_input = st.text_area(
        "Enter multiple sentences (one per line):",
        value="I love spending time with family and close friends.\nI am anxious about the upcoming exam results.\nShe felt a deep sorrow watching him leave forever.\nWhat an astonishing discovery this was!",
        height=140
    )

    if st.button("⚡ Process Batch", type="primary"):
        lines = [line.strip() for line in batch_input.strip().split("\n") if line.strip()]
        if lines:
            cleaned_lines = [clean_text(line) for line in lines]
            feats = vectorizer.transform(cleaned_lines)
            preds = model.predict(feats)
            probs = model.predict_proba(feats)
            classes = metadata.get("classes", list(model.classes_))

            results = []
            for i, line in enumerate(lines):
                pred_label = classes[preds[i]]
                conf = np.max(probs[i]) * 100
                emoji = EMOTION_CONFIG.get(pred_label, {}).get('emoji', '💬')
                results.append({
                    "Text": line,
                    "Emotion": f"{emoji} {pred_label.capitalize()}",
                    "Confidence (%)": round(conf, 1)
                })

            df_batch = pd.DataFrame(results)
            st.dataframe(df_batch, use_container_width=True)

            # Emotion distribution summary
            st.markdown("##### 📊 Emotion Count Summary")
            counts = df_batch['Emotion'].value_counts().reset_index()
            counts.columns = ['Emotion', 'Count']
            fig_bar = px.bar(counts, x='Emotion', y='Count', color='Emotion', text='Count')
            fig_bar.update_layout(height=260, plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

with tab_analytics:
    st.markdown("##### 📊 Model Benchmarks & Performance Metrics")
    
    if metadata and "metrics" in metadata:
        m1, m2, m3 = st.columns(3)
        metrics = metadata["metrics"]
        m1.metric("Naive Bayes (BoW)", f"{metrics.get('MultinomialNB_BoW', 0)*100:.2f}%")
        m2.metric("Naive Bayes (TF-IDF)", f"{metrics.get('MultinomialNB_TFIDF', 0)*100:.2f}%")
        m3.metric("Logistic Regression (TF-IDF)", f"{metrics.get('LogisticRegression_TFIDF', 0)*100:.2f}%", delta="Best Model")

        st.markdown("---")
        c1, c2 = st.columns([1, 1])

        with c1:
            st.markdown("###### 📊 Model Comparison")
            bench_df = pd.DataFrame({
                "Algorithm": ["Naive Bayes (BoW)", "Naive Bayes (TF-IDF)", "Logistic Regression (TF-IDF)"],
                "Accuracy (%)": [
                    metrics.get('MultinomialNB_BoW', 0)*100,
                    metrics.get('MultinomialNB_TFIDF', 0)*100,
                    metrics.get('LogisticRegression_TFIDF', 0)*100
                ]
            })
            fig_comp = px.bar(bench_df, x="Algorithm", y="Accuracy (%)", text="Accuracy (%)",
                              color="Algorithm", color_discrete_sequence=['#94A3B8', '#818CF8', '#10B981'])
            fig_comp.update_layout(height=300, yaxis_range=[0, 100], plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig_comp, use_container_width=True)

        with c2:
            st.markdown("###### 🏷️ Dataset Class Distribution")
            class_counts = metadata.get("dataset_stats", {}).get("class_counts", {})
            if class_counts:
                df_dist = pd.DataFrame({
                    "Emotion": [c.capitalize() for c in class_counts.keys()],
                    "Samples": list(class_counts.values())
                })
                fig_pie = px.pie(df_dist, names="Emotion", values="Samples", hole=0.4,
                                 color_discrete_sequence=['#3B82F6', '#10B981', '#EF4444', '#8B5CF6', '#EC4899', '#F59E0B'])
                fig_pie.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10))
                st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Run `python src/train.py` to populate performance metrics.")
