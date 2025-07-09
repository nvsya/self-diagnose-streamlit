
import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(["Negative", "Neutral", "Positive"])

# Load models and vectorizer
with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)
with open("xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# UI Setup
st.set_page_config(page_title="Self-Diagnose Sentiment Analyzer", layout="centered")
st.markdown("## 🧠 TikTok Comment Sentiment Analysis")
st.markdown("Prediction of self-diagnosis tendencies based on mental health-related comments")

# User input
user_input = st.text_area("📝 Enter your TikTok comments below:")

# Model selector
model_choice = st.radio("🔍 Select Algorithm:", ["Support Vector Machine (SVM)", "Extreme Gradient Boosting (XGBoost)"])

# Predict button
if st.button("🚀 Sentiment Prediction"):
    if user_input.strip() == "":
        st.warning("Please enter a comment first.")
    else:
        vec = vectorizer.transform([user_input])
        if model_choice == "Support Vector Machine (SVM)":
            prediction = svm_model.predict(vec)[0]
        else:
            prediction = xgb_model.predict(vec)[0]
            
        label = label_encoder.inverse_transform([prediction])[0]
        st.success(f"💡 Sentiment Prediction Results: **{label.upper()}**")

# Footer
st.markdown("---")
st.markdown("Created for thesis purposes by Navsya Nitisara 🌸")
