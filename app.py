import joblib
import pandas as pd
import streamlit as st
from features import add_features  # needed so joblib can load the pipeline

st.set_page_config(page_title="Student Success Predictor", page_icon="🎓")


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


model = load_model()

st.title("🎓 Student Success Predictor")
st.write(
    "Enter a student's learning-platform activity to estimate whether their "
    "performance rating is likely to be **High** or **Low**."
)

with st.form("student_form"):
    c1, c2 = st.columns(2)
    with c1:
        days = st.number_input("Days logged in", 0, 100, 14)
        read = st.number_input("Read actions", 0, 500000, 130)
        submission = st.number_input("Submission actions", 0, 500, 3)
        create = st.number_input("Create actions", 0, 500, 7)
    with c2:
        quiz = st.number_input("Quiz actions", 0, 1000, 0)
        forum = st.number_input("Forum actions", 0, 500, 0)
        update = st.number_input("Update actions", 0, 500, 3)
        delete = st.number_input("Delete actions", 0, 50, 0)
    submitted = st.form_submit_button("Predict")

if submitted:
    X = pd.DataFrame([{
        "Submission.Actions": submission,
        "Quiz.Actions": quiz,
        "Forum.Actions": forum,
        "Create.Actions": create,
        "Read.Actions": read,
        "Update.Actions": update,
        "Delete.Actions": delete,
        "Days.Logged.In": days,
    }])
    p_high = model.predict_proba(X)[0][1]
    if p_high >= 0.5:
        st.success(f"Likely **High** rating (probability {p_high:.0%})")
    else:
        st.error(f"At risk of a **Low** rating (probability {1 - p_high:.0%})")
    st.progress(float(p_high), text=f"P(High) = {p_high:.0%}")
    st.caption(
        "This is a risk indicator to support advising, not a final judgment. "
        "Model accuracy is roughly 82% on held-out data."
    )
