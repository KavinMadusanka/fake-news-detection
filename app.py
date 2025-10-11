
import utils.session
import streamlit as st
st.set_page_config(page_title="Fake News Detection (Deep Learning)", page_icon="📰")

import re, os, joblib, numpy as np, tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------- CONFIG ----------
MODEL_DIR = "results/models/models"
MAX_LEN = 300
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
def safe_load_model(names):
    for n in names:
        path = os.path.join(MODEL_DIR, n)
        if os.path.exists(path):
            return tf.keras.models.load_model(path, compile=False)
    return None

@st.cache_resource
def safe_joblib(name):
    path = os.path.join(MODEL_DIR, name)
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_resource
def load_models():
    lstm_model   = safe_load_model(["lstm_model.keras", "lstm.h5"])
    bilstm_model = safe_load_model(["bilstm_model.keras"])
    rnn_model    = safe_load_model(["simplernn_model.keras"])
    cnn_model    = safe_load_model(["cnn_model.keras"])
    tokenizer    = safe_joblib("tokenizer.joblib")

    return lstm_model, bilstm_model, rnn_model, cnn_model, tokenizer

lstm_model, bilstm_model, rnn_model, cnn_model, tokenizer = load_models()

# ---------- PAGE HEADER ----------
st.title("📰 Fake News Detection Dashboard (Deep Learning)")
st.markdown(
    "This tool analyzes a given news headline or short paragraph "
    "using **multiple deep learning models (LSTM, BiLSTM, RNN, CNN)** "
    "to determine whether it’s **real or fake news**."
)

user_text = st.text_area(
    "Enter a news headline or paragraph:",
    height=160,
    placeholder="e.g. Sri Lanka central bank announces new monetary policy adjustments for 2025"
)
run = st.button("🔍 Detect")

# ---------- PREDICTION ----------
def predict_fake_news(text):
    results = {}
    text_cleaned = clean_text(text)

    def run_dl_model(model, name):
        if model is None or tokenizer is None:
            return
        seq = tokenizer.texts_to_sequences([text_cleaned])
        pad = pad_sequences(seq, maxlen=MAX_LEN)
        prob_fake = float(model.predict(pad, verbose=0)[0][0])
        pred = 1 if prob_fake >= 0.5 else 0
        results[name] = ("Fake" if pred else "Real", round(prob_fake, 3))

    # --- Run all available models ---
    run_dl_model(lstm_model, "LSTM")
    run_dl_model(bilstm_model, "BiLSTM")
    run_dl_model(rnn_model, "RNN")
    run_dl_model(cnn_model, "CNN")

    # --- Majority Vote ---
    votes = [label for label, _ in results.values()]
    final = "Fake" if votes.count("Fake") >= (len(votes)//2 + 1) else "Real"
    results["Majority_Vote"] = (final, None)
    return results

# ---------- DISPLAY RESULTS ----------
if run and user_text.strip():
    preds = predict_fake_news(user_text)

    st.subheader("Prediction Results")
    for model, (label, prob) in preds.items():
        if model != "Majority_Vote":
            st.write(f"**{model}:** {label}  |  Confidence = `{prob:.3f}`")

    st.markdown("---")
    final_label = preds["Majority_Vote"][0]
    if final_label == "Fake":
        st.error(f"🧾 **Final Decision:** {final_label}")
    else:
        st.success(f"🧾 **Final Decision:** {final_label}")

else:
    st.info("💡 Enter text above and click **Detect** to begin.")
