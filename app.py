import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("myModel.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="📘 Student Performance Predictor", page_icon="📘")
st.title("📘 Student Performance Predictor")
st.write("Enter the details below to predict the student's performance index.")

# Input features
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        hours_studied = st.number_input("Hours Studied", 0.0, 24.0, step=0.5)
        previous_scores = st.number_input("Previous Scores", 0.0, 100.0, step=1.0)
        extracurricular = st.selectbox("Participates in Extracurricular Activities?", ['Yes', 'No'])
    with col2:
        sleep_hours = st.number_input("Sleep Hours", 0.0, 24.0, step=0.5)
        sample_papers = st.number_input("Sample Question Papers Practiced", 0, 100, step=1)

    submit = st.form_submit_button("🔍 Predict")

if submit:
    # Convert categorical input to numerical
    extracurricular_num = 1 if extracurricular == 'Yes' else 0

    input_data = np.array([[hours_studied, previous_scores, extracurricular_num, sleep_hours, sample_papers]])
    df_input = pd.DataFrame(input_data, columns=["Hours Studied", "Previous Scores", "Extracurricular Activities", "Sleep Hours", "Sample Question Papers Practiced"])
    
    prediction = model.predict(df_input)[0]
    st.success(f"📈 Predicted Performance Index: **{prediction:.2f}**")
