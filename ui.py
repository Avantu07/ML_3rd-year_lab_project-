
import streamlit as st
import pandas as pd

from app import load_and_train_models, predict_heart_disease

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Risk Prediction")
st.write("Machine Learning Based Prediction System")


@st.cache_resource
def get_models():
    return load_and_train_models()


with st.spinner("Loading dataset and training models..."):
    df, X, models, scores = get_models()

st.success("Dataset loaded and models trained successfully!")

col1, col2 = st.columns(2)
col1.metric("Patient Records", len(df))
col2.metric("Models Compared", len(models))

st.subheader("Model Accuracy Comparison")
st.bar_chart(pd.Series(scores, name="Accuracy (%)"))

st.dataframe(
    pd.DataFrame(
        list(scores.items()),
        columns=["Model", "Accuracy (%)"]
    ),
    hide_index=True
)

st.divider()
st.subheader("Enter Patient Details")

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 1, 120, 45)
        sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
        cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4])
        trestbps = st.number_input(
            "Resting Blood Pressure", 50, 250, 120
        )
        chol = st.number_input("Cholesterol", 50, 700, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
        restecg = st.selectbox("Resting ECG", [0, 1, 2])

    with col2:
        thalach = st.number_input(
            "Maximum Heart Rate", 50, 250, 150
        )
        exang = st.selectbox("Exercise-Induced Angina", [0, 1])
        oldpeak = st.number_input(
            "ST Depression", 0.0, 10.0, 1.0
        )
        slope = st.selectbox("ST Slope", [1, 2, 3])
        ca = st.selectbox("Major Vessels", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia", [3, 6, 7])

    model_name = st.selectbox("Choose Model", list(models.keys()))

    predict_button = st.form_submit_button("Predict Heart Disease")

if predict_button:
    patient = pd.DataFrame([[
        age, sex, cp, trestbps, chol, fbs, restecg,
        thalach, exang, oldpeak, slope, ca, thal
    ]], columns=X.columns)

    result = predict_heart_disease(patient, models[model_name])

    st.subheader("Prediction Result")
    st.write("Model used:", model_name)

    if result == 1:
        st.warning(
            "The model indicates a possible heart disease signal."
        )
    else:
        st.success(
            "The model did not indicate heart disease in this prediction."
        )

    st.caption("Educational project only — not a medical diagnosis.")

st.caption("Dataset: UCI Machine Learning Repository")

