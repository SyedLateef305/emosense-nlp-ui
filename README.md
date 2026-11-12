# 🎭 EmoSense: Real-Time Text Emotion Recognition

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning and Natural Language Processing (NLP) web application that detects human emotional tone from text in real time. Powered by an optimized **TF-IDF + Logistic Regression / Naive Bayes** pipeline and wrapped in an interactive **Streamlit** dashboard.

---

## 🌟 Key Features

- **⚡ Real-Time Emotion Classification**: Detects 6 fine-grained emotions:
  - 😊 **Joy** | 😢 **Sadness** | 😡 **Anger** | 😨 **Fear** | ❤️ **Love** | 😲 **Surprise**
- **📊 Probability Confidence Breakdown**: Interactive horizontal distribution charts rendered with Plotly.
- **📂 Batch Inference**: Upload or paste multiple sentences to analyze customer feedback, survey responses, or tweets simultaneously.
- **📈 Model Benchmarks**: Side-by-side performance evaluation comparing Bag-of-Words (BoW) Multinomial Naive Bayes, TF-IDF Naive Bayes, and Logistic Regression.
- **🧹 Transparent NLP Pipeline**: Live preview of raw text normalization, tokenization, emoji stripping, and stopword filtering.

---

## 🏗️ Architecture & Pipeline

```text
Raw Text Input
      │
      ▼
1. Text Preprocessing
   ├── Lowercasing & ASCII Normalization
   ├── URL, HTML Tag & Emoji Removal
   ├── Punctuation & Digit Stripping
   └── Stopword Filtering
      │
      ▼
2. Feature Extraction (TF-IDF Vectorization)
   ├── Sublinear TF scaling
   └── Unigrams & Bigrams
      │
      ▼
3. Machine Learning Classifier
   └── Multi-Class Logistic Regression (L-BFGS)
      │
      ▼
4. Output Presentation
   ├── Predicted Emotion & Confidence Score
   └── Plotly Probability Distribution
```

---

## 📊 Model Performance Benchmarks

Trained on **16,000+** annotated text instances:

| Model Architecture | Feature Representation | Accuracy |
| :--- | :--- | :--- |
| **Multinomial Naive Bayes** | Bag-of-Words (CountVectorizer) | ~80.2% |
| **Multinomial Naive Bayes** | TF-IDF Vectorizer | ~77.8% |
| **Logistic Regression** (Production) | **TF-IDF Vectorizer** | **~89.1%** |

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/<YOUR-USERNAME>/text-emotion-recognition.git
cd text-emotion-recognition
```

### 2. Create and Activate Virtual Environment
Using standard Python `venv`:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```
*(Or with `uv`)*:
```bash
uv venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Models
```bash
python src/train.py
```
This processes the dataset, trains the candidate models, and exports the serialized model and vectorizer into the `models/` directory.

### 5. Launch the Streamlit Web App
```bash
streamlit run app.py
# or
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501` to test the live application!

---

## 📁 Project Structure

```
text-emotion-recognition/
├── data/
│   └── train.txt                 # Semicolon-separated annotated training dataset
├── models/
│   ├── emotion_model.joblib      # Serialized trained Logistic Regression model
│   ├── tfidf_vectorizer.joblib   # Serialized TF-IDF vectorizer
│   └── metadata.json             # Labels, evaluation metrics & dataset stats
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # Modular text cleaning and normalization
│   └── train.py                  # Training pipeline, evaluation & model export
├── app.py                        # Streamlit web application
├── requirements.txt              # Project dependencies
├── .gitignore                    # Git ignore file
├── README.md                     # Project documentation
└── LINKEDIN_POST.md              # Ready-to-publish LinkedIn showcase post
```

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com).

## 📄 License
This project is licensed under the [MIT License](LICENSE).
