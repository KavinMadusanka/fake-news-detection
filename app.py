
import streamlit as st
st.set_page_config(page_title="Fake News Detection", page_icon="📰")

import re, joblib, numpy as np, tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------- CONFIG ----------
MODEL_DIR = "/content/drive/MyDrive/fake_news_project/results/models/models"
MAX_LEN = 300
USE_LSTM = True     # set to False if you want only TF-IDF + SVM
# ---------------------------

# ---------- TEXT CLEANER ----------
def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"[^a-z\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ---------- LOAD MODELS ----------
@st.cache_resource
def load_models():
    tfidf = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer.joblib")
    tfidf_lr = joblib.load(f"{MODEL_DIR}/tfidf_logreg.joblib")
    svm = joblib.load(f"{MODEL_DIR}/svm.joblib")

    lstm, tokenizer = None, None
    if USE_LSTM:
        lstm = tf.keras.models.load_model(f"{MODEL_DIR}/lstm.h5", compile=False)
        try:
            tokenizer = joblib.load(f"{MODEL_DIR}/lstm_tokenizer.joblib")
        except:
            tokenizer = None
            st.warning("Tokenizer for LSTM not found — skipping LSTM model.")
    return tfidf, tfidf_lr, svm, lstm, tokenizer


tfidf, tfidf_lr, svm, lstm, tokenizer = load_models()

# ---------- PAGE HEADER ----------
st.title("📰 Fake News Detection Dashboard")
st.markdown(
    "This tool analyzes a given news headline or short paragraph "
    "using multiple machine-learning models to determine whether it’s **real or fake news**."
)

user_text = st.text_area(
    "Enter a news headline or paragraph:",
    height=160,
    placeholder="e.g. Sri Lanka central bank announces new monetary policy adjustments for 2025"
)
run = st.button("🔍 Detect")

# ---------- HELPER ----------
def pseudo_confidence_from_decision(score: float) -> float:
    """Normalize SVM decision score to 0–1."""
    return float(1 / (1 + np.exp(-abs(score))))

# ---------- PREDICTION ----------
if run and user_text.strip():
    text_c = clean_text(user_text)
    results = []

    # --- TF-IDF + Logistic Regression ---
    X = tfidf.transform([text_c])
    pred_lr = int(tfidf_lr.predict(X)[0])
    prob_lr = float(tfidf_lr.predict_proba(X)[0][pred_lr])
    label_lr = "Fake" if pred_lr == 1 else "Real"
    results.append(("TFIDF + LogReg", label_lr, prob_lr))

    # --- SVM ---
    pred_svm = int(svm.predict(X)[0])
    score = float(svm.decision_function(X)[0])
    conf_svm = pseudo_confidence_from_decision(score)
    label_svm = "Fake" if pred_svm == 1 else "Real"
    results.append(("SVM", label_svm, conf_svm))

    # --- LSTM ---
    if USE_LSTM and tokenizer is not None and lstm is not None:
        seq = tokenizer.texts_to_sequences([text_c])
        pad = pad_sequences(seq, maxlen=MAX_LEN)
        prob_fake = float(lstm.predict(pad, verbose=0)[0][0])
        pred_lstm = 1 if prob_fake >= 0.5 else 0
        label_lstm = "Fake" if pred_lstm == 1 else "Real"
        results.append(("LSTM", label_lstm, prob_fake))

    # --- Majority Vote ---
    votes = [r[1] for r in results]
    final_label = "Fake" if votes.count("Fake") >= (len(votes)//2 + 1) else "Real"

    # ---------- DISPLAY RESULTS ----------
    st.subheader("Prediction Results")
    for name, label, conf in results:
        st.write(f"**{name}:** {label}  |  Confidence = `{conf:.3f}`")

    st.markdown("---")
    if final_label == "Fake":
        st.error(f"🧾 **Final Decision:** {final_label}")
    else:
        st.success(f"🧾 **Final Decision:** {final_label}")

    st.caption("*(Note: SVM confidence is a normalized margin, not a true probability.)*")

else:
    st.info("💡 Enter text above and click **Detect** to begin.")
