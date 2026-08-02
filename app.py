import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Diabetes Risk Estimator", page_icon="🩺")

@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("model.joblib")
    schema = joblib.load("input_schema.joblib")


    # TreeExplainer just wraps the already-fitted forest.
    # (no retraining), so it's fine to build it here and cache it once.
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    return pipeline, schema, explainer

pipeline, schema, explainer = load_artifacts()
feature_cols = schema["feature_cols"]
options = schema["options"]

# Readable labels for each survey code, shown in the UI
LABELS = {
    "GENHLTH": "General health",
    "X_BMI5CAT": "BMI category",
    "CHECKUP1": "Last routine checkup",
    "INCOME2": "Household income",
    "X_RACE": "Race",
    "MSCODE": "Metro status",
    "FLUSHOT6": "Flu shot in last 12 months",
    "EMPLOY1": "Employment status",
    "SEX": "Sex",
    "MARITAL": "Marital status",
    "X_EDUCAG": "Education level",
    "CVDCRHD4": "History of coronary heart disease",
    "HLTHCVR1": "Primary health coverage",
    "CHCKIDNY": "History of kidney disease",
    "USEEQUIP": "Uses mobility equipment",
    "X_TOTINDA": "Physical activity in last 30 days",
    "ADDEPEV2": "Ever told had a depressive disorder",
    "RENTHOM1": "Home ownership",
    "EXERANY2": "Exercise in last 30 days",
    "BLIND": "Serious vision difficulty",
    "DECIDE": "Serious difficulty concentrating/remembering",
    "X_SMOKER3": "Smoking status",
    "X_AGEG10YR": "Age group",
    "SLEPTIM1G3": "Average sleep",
    "MENTHLTHG3": "Poor mental health days (last 30)",
}

st.title("🩺 Diabetes Risk Estimator")
st.caption(
    "Predicts the likelihood of a diabetes diagnosis from BRFSS-style survey "
    "responses. Trained on BRFSS 2014 data. For education only, not medical advice."
)

with st.form("risk_form"):
    st.subheader("Respondent profile")
    inputs = {}
    cols = st.columns(2)
    for i, col in enumerate(feature_cols):
        with cols[i % 2]:
            inputs[col] = st.selectbox(LABELS.get(col, col), options[col])
    submitted = st.form_submit_button("Estimate risk")

if submitted:
    row = pd.DataFrame([inputs])[feature_cols]
    proba = pipeline.predict_proba(row)[0, 1]
    pred = "Yes" if proba >= 0.5 else "No"

    st.subheader("Result")
    st.metric("Estimated diabetes probability", f"{proba:.1%}")
    st.progress(min(int(proba * 100), 100))

    if pred == "Yes":
        st.warning("Model prediction: higher-risk profile")
    else:
        st.success("Model prediction: lower-risk profile")

    st.caption(
        "This is a statistical estimate from a survey-trained model, "
        "not a diagnosis. Consult a healthcare professional for medical advice."
    )

    st.subheader("Why the model said this")
    preprocess = pipeline.named_steps["preprocess"]
    encoded_names = preprocess.named_transformers_["cat"].get_feature_names_out(feature_cols)
    row_encoded = preprocess.transform(row).toarray()

    sv = explainer.shap_values(row_encoded)
    sv_yes = sv[0, :, 1]

    active_idx = np.where(row_encoded[0] == 1)[0]
    contributions = [(encoded_names[i], sv_yes[i]) for i in active_idx]
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top = contributions[:8]

    labels = [f"{LABELS.get(n.split('_')[0], n.split('_')[0])}: {inputs[n.split('_')[0]]}" for n, _ in top]
    values = [v for _, v in top]
    colors = ["#C62828" if v > 0 else "#1565C0" for v in values]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel("Impact on risk estimate (SHAP value)")
    ax.axvline(0, color="black", linewidth=0.8)
    st.pyplot(fig)
    st.caption("Red = pushed the estimate toward higher risk. Blue = pushed it toward lower risk.")

st.divider()
st.caption("Data source: CDC BRFSS 2014 · Model: Random Forest (scikit-learn)")
