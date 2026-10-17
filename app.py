
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_model.pkl"

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
)

st.title("🎓 Student Performance Predictor")
st.write(
    "Predict a student's final mathematics grade "
    "using their earlier grades and school information."
)

if not MODEL_PATH.exists():
    st.error("Model not found. Run train_model.py first.")
    st.stop()

saved = joblib.load(MODEL_PATH)
model = saved["model"]
features = saved["features"]

st.subheader("Student Information")

with st.form("student_form"):
    studytime = st.selectbox(
        "Weekly study time",
        [1, 2, 3, 4],
        format_func=lambda x: {
            1: "Less than 2 hours",
            2: "2–5 hours",
            3: "5–10 hours",
            4: "More than 10 hours",
        }[x],
    )

    failures = st.number_input(
        "Past class failures",
        min_value=0,
        max_value=4,
        value=0,
        step=1,
    )

    absences = st.number_input(
        "Number of absences",
        min_value=0,
        max_value=93,
        value=0,
        step=1,
    )

    g1 = st.slider("First-period grade (G1)", 0, 20, 10)
    g2 = st.slider("Second-period grade (G2)", 0, 20, 10)

    submitted = st.form_submit_button("Predict final grade")

if submitted:
    valid_inputs = (
        studytime in [1, 2, 3, 4]
        and 0 <= failures <= 4
        and 0 <= absences <= 93
        and 0 <= g1 <= 20
        and 0 <= g2 <= 20
    )

    if not valid_inputs:
        st.error("Please check your inputs and try again.")

    else:
        inputs = pd.DataFrame(
            [[studytime, failures, absences, g1, g2]],
            columns=features,
        )




    prediction = float(model.predict(inputs)[0])
    prediction = max(0, min(20, prediction))

    st.metric("Predicted final grade", f"{prediction:.1f} / 20")

    st.caption(
        "This is an estimate, not a guarantee. "
        "The model was trained on historical data from two "
        "Portuguese schools and may not generalize elsewhere."
    )
