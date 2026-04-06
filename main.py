# main.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import InterviewPerformanceModel
from utils import generate_sample_data, get_feedback

# Page config
st.set_page_config(page_title="Interview Analysis", layout="wide")

st.title("🎯 Interview Performance Analysis System")

# Generate data
df = generate_sample_data(200)

# Train model
model = InterviewPerformanceModel()
model.train(df)

# Sidebar input
st.sidebar.header("Enter Candidate Details")

aptitude = st.sidebar.slider("Aptitude", 0, 100, 60)
communication = st.sidebar.slider("Communication", 0, 100, 60)
technical = st.sidebar.slider("Technical", 0, 100, 60)
confidence = st.sidebar.slider("Confidence", 0, 100, 60)

# Prediction
if st.sidebar.button("Analyze Performance"):
    input_data = [aptitude, communication, technical, confidence]
    prediction, cluster = model.predict_performance(input_data)

    st.subheader("📊 Result")
    st.write(f"**Predicted Performance:** {prediction}")
    st.write(f"**Cluster Group:** {cluster}")
    st.success(get_feedback(prediction))

# ------------------ Visualization ------------------

st.subheader("📈 Data Visualization")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    df["Performance"].value_counts().plot(kind='bar', ax=ax)
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    ax.scatter(df["Aptitude"], df["Technical"])
    ax.set_xlabel("Aptitude")
    ax.set_ylabel("Technical")
    st.pyplot(fig)

# ------------------ Data Table ------------------

st.subheader("📋 Dataset Preview")
st.dataframe(df.head())