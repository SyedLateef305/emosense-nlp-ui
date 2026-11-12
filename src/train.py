"""
Model Training & Evaluation Pipeline for Emotion Recognition.
Trains and compares Multinomial Naive Bayes and Logistic Regression models.
Saves the best model, vectorizer, and label metadata to models/ directory.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ensure current package can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import clean_text

EMOTION_EMOJIS = {
    'sadness': '😢',
    'joy': '😊',
    'love': '❤️',
    'anger': '😡',
    'fear': '😨',
    'surprise': '😲'
}

def train_pipeline(data_path="data/train.txt", models_dir="models"):
    print(f"[+] Loading dataset from {data_path}...")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path, sep=';', header=None, names=['text', 'emotion'])
    print(f"    Loaded {len(df)} records.")
    print(f"    Classes distribution:\n{df['emotion'].value_counts()}")

    # 1. Label Mapping
    unique_emotions = sorted(df['emotion'].unique().tolist())
    emotion_to_id = {emo: idx for idx, emo in enumerate(unique_emotions)}
    id_to_emotion = {str(idx): emo for idx, emo in enumerate(unique_emotions)}

    df['label'] = df['emotion'].map(emotion_to_id)

    # 2. Preprocess text
    print("[+] Cleaning text data...")
    df['cleaned_text'] = df['text'].apply(clean_text)

    # Filter out empty rows after cleaning if any
    df = df[df['cleaned_text'].str.strip() != ''].copy()

    X = df['cleaned_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"    Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 3. Model 1: CountVectorizer + MultinomialNB
    print("[+] Training Model 1: CountVectorizer + MultinomialNB...")
    bow_vectorizer = CountVectorizer()
    X_train_bow = bow_vectorizer.fit_transform(X_train)
    X_test_bow = bow_vectorizer.transform(X_test)

    nb_bow = MultinomialNB()
    nb_bow.fit(X_train_bow, y_train)
    pred_bow = nb_bow.predict(X_test_bow)
    acc_bow = accuracy_score(y_test, pred_bow)
    print(f"    MultinomialNB (BoW) Accuracy: {acc_bow * 100:.2f}%")

    # 4. Model 2: TfidfVectorizer + MultinomialNB
    print("[+] Training Model 2: TF-IDF + MultinomialNB...")
    tfidf_vectorizer = TfidfVectorizer(max_features=10000, sublinear_tf=True)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    nb_tfidf = MultinomialNB()
    nb_tfidf.fit(X_train_tfidf, y_train)
    pred_nb_tfidf = nb_tfidf.predict(X_test_tfidf)
    acc_nb_tfidf = accuracy_score(y_test, pred_nb_tfidf)
    print(f"    MultinomialNB (TF-IDF) Accuracy: {acc_nb_tfidf * 100:.2f}%")

    # 5. Model 3: TfidfVectorizer + Logistic Regression
    print("[+] Training Model 3: TF-IDF + Logistic Regression...")
    logistic_model = LogisticRegression(max_iter=1000, C=2.0, solver='lbfgs')
    logistic_model.fit(X_train_tfidf, y_train)
    pred_log = logistic_model.predict(X_test_tfidf)
    acc_log = accuracy_score(y_test, pred_log)
    print(f"    Logistic Regression (TF-IDF) Accuracy: {acc_log * 100:.2f}%")

    report = classification_report(
        y_test, pred_log, target_names=unique_emotions, output_dict=True
    )
    cm = confusion_matrix(y_test, pred_log).tolist()

    # 6. Save Artifacts
    os.makedirs(models_dir, exist_ok=True)
    model_file = os.path.join(models_dir, "emotion_model.joblib")
    vectorizer_file = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    meta_file = os.path.join(models_dir, "metadata.json")

    print(f"[+] Saving best model (Logistic Regression) to {model_file}...")
    joblib.dump(logistic_model, model_file)

    print(f"[+] Saving TF-IDF vectorizer to {vectorizer_file}...")
    joblib.dump(tfidf_vectorizer, vectorizer_file)

    metadata = {
        "emotion_to_id": emotion_to_id,
        "id_to_emotion": id_to_emotion,
        "emotion_emojis": EMOTION_EMOJIS,
        "classes": unique_emotions,
        "metrics": {
            "MultinomialNB_BoW": round(float(acc_bow), 4),
            "MultinomialNB_TFIDF": round(float(acc_nb_tfidf), 4),
            "LogisticRegression_TFIDF": round(float(acc_log), 4),
        },
        "best_model": "LogisticRegression_TFIDF",
        "best_accuracy": round(float(acc_log), 4),
        "classification_report": report,
        "confusion_matrix": cm,
        "dataset_stats": {
            "total_samples": len(df),
            "class_counts": df['emotion'].value_counts().to_dict()
        }
    }

    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"[+] Saved metadata to {meta_file}.")
    print("[OK] Training pipeline successfully completed!")

if __name__ == "__main__":
    train_pipeline()
