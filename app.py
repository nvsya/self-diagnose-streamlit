
import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(["Negatif", "Netral", "Positif"])

# Load models and vectorizer
with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)
with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# UI Setup
st.set_page_config(page_title="Self-Diagnose Sentiment Analyzer", layout="centered")
st.markdown("## 🧠 Analisis Sentimen Komentar TikTok")
st.markdown("Prediksi kecenderungan self-diagnose berdasarkan komentar terkait kesehatan mental.")

# User input
user_input = st.text_area("📝 Masukkan komentar TikTok di bawah ini:")

# Model selector
model_choice = st.radio("🔍 Pilih Algoritma:", ["Support Vector Machine (SVM)", "Extreme Gradient Boosting (XGBoost)"])

# Predict button
if st.button("🚀 Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("Harap masukkan komentar terlebih dahulu.")
    else:
        vec = vectorizer.transform([user_input])
        if model_choice == "Support Vector Machine (SVM)":
            prediction = svm_model.predict(vec)[0]
        else:
            prediction = xgb_model.predict(vec)[0]
            
        label = label_encoder.inverse_transform([prediction])[0]
        st.success(f"💡 Hasil Prediksi Sentimen: **{label.upper()}**")

# Footer
st.markdown("---")
st.markdown("Dibuat untuk keperluan skripsi oleh Navsya Nitisara 🌸")
