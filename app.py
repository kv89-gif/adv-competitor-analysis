import streamlit as st
import pandas as pd
import joblib
from enhanced_feature_extractor import extract_enhanced_features

st.title("Backlink Spam Classifier v4")

@st.cache_resource
def load_model():
    return joblib.load("spam_model_v4.pkl")

model = load_model()
model_features = getattr(model, "feature_names_in_", None)

uploaded_file = st.file_uploader("📤 Upload CSV of URLs", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Extract features with or without feature name alignment
        features = extract_enhanced_features(df, model_feature_names=model_features)

        # Predict
        preds = model.predict(features)
        df["Prediction"] = ["Spam" if p == 1 else "Not Spam" for p in preds]

        # Display results
        st.dataframe(df)

        # Offer download
        st.download_button(
