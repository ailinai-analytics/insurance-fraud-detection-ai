import os
from PIL import Image
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Insurance Claim Risk Dashboard", layout="wide")

st.title("Insurance Claim Risk Scoring Dashboard")

# -----------------------------
# Load files
# -----------------------------
pred_df = pd.read_csv("outputs/model_predictions.csv")
feat_df = pd.read_csv("outputs/feature_importance.csv")

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")

risk_threshold = st.sidebar.slider(
    "Minimum Fraud Risk Score",
    min_value=0.0,
    max_value=1.0,
    value=0.20,
    step=0.01
)

top_n = st.sidebar.slider(
    "Number of Top Suspicious Claims",
    min_value=5,
    max_value=50,
    value=20,
    step=5
)

filtered_df = pred_df[pred_df["fraud_risk_score"] >= risk_threshold].copy()

# -----------------------------
# KPI section
# -----------------------------
total_claims = len(pred_df)
filtered_claims = len(filtered_df)
predicted_suspicious = pred_df["predicted_target"].sum()
actual_suspicious = pred_df["actual_target"].sum()
avg_risk_score = pred_df["fraud_risk_score"].mean()

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Test Claims", total_claims)
col2.metric("Filtered Claims", filtered_claims)
col3.metric("Predicted Suspicious", int(predicted_suspicious))
col4.metric("Actual Suspicious", int(actual_suspicious))
col5.metric("Average Risk Score", round(avg_risk_score, 3))

st.markdown("---")

# -----------------------------
# Risk score distribution
# -----------------------------
st.subheader("Fraud Risk Score Distribution")

fig, ax = plt.subplots(figsize=(8, 4))
pred_df["fraud_risk_score"].hist(bins=30, ax=ax)
ax.set_title("Distribution of Fraud Risk Scores")
ax.set_xlabel("Fraud Risk Score")
ax.set_ylabel("Count")
st.pyplot(fig)
plt.close(fig)

st.markdown("---")

# -----------------------------
# Top suspicious cases
# -----------------------------
st.subheader("Top Suspicious Claims")

top_risky = filtered_df.sort_values("fraud_risk_score", ascending=False).head(top_n)

display_cols = [
    col for col in [
        "TRANSACTION_ID",
        "INSURANCE_TYPE",
        "PREMIUM_AMOUNT",
        "CLAIM_AMOUNT",
        "CITY_x",
        "STATE_x",
        "AGE",
        "TENURE",
        "INCIDENT_SEVERITY",
        "fraud_risk_score",
        "actual_target",
        "predicted_target"
    ] if col in top_risky.columns
]

st.dataframe(top_risky[display_cols], use_container_width=True)

csv_data = top_risky.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Top Suspicious Claims CSV",
    data=csv_data,
    file_name="top_suspicious_claims.csv",
    mime="text/csv"
)

st.markdown("---")

# -----------------------------
# Claim investigation panel
# -----------------------------
# -----------------------------
# -----------------------------
# Claim investigation panel
# -----------------------------
st.subheader("Claim Investigation Panel")

if len(top_risky) > 0:
    # Pick an identifier column if available
    candidate_id_cols = ["TRANSACTION_ID", "CUSTOMER_ID", "POLICY_NUMBER"]
    id_col = next((col for col in candidate_id_cols if col in top_risky.columns), None)

    if id_col is None:
        top_risky = top_risky.copy()
        top_risky["ROW_ID"] = top_risky.index.astype(str)
        id_col = "ROW_ID"

    selected_claim = st.selectbox(
        "Select a claim to investigate",
        top_risky[id_col].astype(str).tolist()
    )

    claim_row = top_risky[
        top_risky[id_col].astype(str) == str(selected_claim)
    ].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Claim ID", str(claim_row.get(id_col, "N/A")))
    c2.metric("Fraud Risk Score", round(float(claim_row.get("fraud_risk_score", 0)), 3))
    c3.metric("Predicted Label", int(claim_row.get("predicted_target", 0)))

    detail_cols = [
        "INSURANCE_TYPE",
        "PREMIUM_AMOUNT",
        "CLAIM_AMOUNT",
        "CITY_x",
        "STATE_x",
        "AGE",
        "TENURE",
        "INCIDENT_SEVERITY",
        "actual_target"
    ]

    available_details = {
        col: claim_row[col]
        for col in detail_cols
        if col in claim_row.index
    }

    st.write("### Claim Details")
    if available_details:
        st.json(available_details)
    else:
        st.info("Detailed claim fields are not present in model_predictions.csv.")

    st.write("### Top Risk Signals")

    risk_signals = []

    try:
        claim_amount = float(claim_row.get("CLAIM_AMOUNT", 0))
    except:
        claim_amount = 0

    try:
        premium_amount = float(claim_row.get("PREMIUM_AMOUNT", 0))
    except:
        premium_amount = 0

    try:
        tenure = float(claim_row.get("TENURE", 0))
    except:
        tenure = 0

    severity = str(claim_row.get("INCIDENT_SEVERITY", ""))

    if claim_amount > 5000:
        risk_signals.append("High claim amount")

    if premium_amount > 0 and claim_amount / premium_amount > 20:
        risk_signals.append("Very high claim-to-premium ratio")

    if tenure < 200:
        risk_signals.append("Relatively new customer or policy")

    if severity in ["Major Loss", "Total Loss"]:
        risk_signals.append("High incident severity")

    if risk_signals:
        for signal in risk_signals:
            st.write(f"- {signal}")
    else:
        st.write("- No major rule-based risk signals detected")

    st.write("### Risk Interpretation")
    risk_score = float(claim_row.get("fraud_risk_score", 0))

    if risk_score >= 0.7:
        st.error("This claim is very high risk and should be prioritized for investigation.")
    elif risk_score >= 0.4:
        st.warning("This claim shows moderate fraud risk and should be reviewed.")
    else:
        st.success("This claim has lower fraud risk compared with the top suspicious set.")
else:
    st.info("No claims available for investigation at the selected threshold.")
# -----------------------------
# Feature importance
# -----------------------------
st.subheader("Top Feature Importance")

top_features = feat_df.head(15).sort_values("importance", ascending=True)

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.barh(top_features["feature"], top_features["importance"])
ax2.set_title("Top 15 Important Features")
ax2.set_xlabel("Importance")
st.pyplot(fig2)
plt.close(fig2)

st.markdown("---")

# -----------------------------
# SHAP explainability
# -----------------------------
st.subheader("Model Explainability (SHAP)")

shap_path = "outputs/shap_summary.png"

if os.path.exists(shap_path):
    shap_img = Image.open(shap_path)
    st.image(
        shap_img,
        caption="SHAP Summary Plot: Global feature impact across many claims",
        use_column_width=True
    )

    st.info(
        "How to read this chart: Each dot represents one claim. "
        "Features near the top are more important overall. "
        "Dots to the right push predictions toward higher fraud risk, "
        "and dots to the left push predictions toward lower fraud risk."
    )
else:
    st.warning("SHAP plot not found. Run: python src/shap_explain.py")

st.markdown("---")

# -----------------------------
# Confusion matrix summary
# -----------------------------
st.subheader("Confusion Matrix Summary")

tn = len(pred_df[(pred_df["actual_target"] == 0) & (pred_df["predicted_target"] == 0)])
fp = len(pred_df[(pred_df["actual_target"] == 0) & (pred_df["predicted_target"] == 1)])
fn = len(pred_df[(pred_df["actual_target"] == 1) & (pred_df["predicted_target"] == 0)])
tp = len(pred_df[(pred_df["actual_target"] == 1) & (pred_df["predicted_target"] == 1)])

cm_df = pd.DataFrame({
    "Metric": ["True Negative", "False Positive", "False Negative", "True Positive"],
    "Value": [tn, fp, fn, tp]
})

st.dataframe(cm_df, use_container_width=True)

st.markdown("---")

# -----------------------------
# Prediction summary table
# -----------------------------
st.subheader("Prediction Summary Table")

summary_df = pred_df.groupby(
    ["actual_target", "predicted_target"]
).size().reset_index(name="count")

st.dataframe(summary_df, use_container_width=True)

st.markdown("---")
# -----------------------------
# Model monitoring
# -----------------------------
st.subheader("Model Monitoring")

predicted_fraud_rate = pred_df["predicted_target"].mean()
actual_fraud_rate = pred_df["actual_target"].mean()
avg_score = pred_df["fraud_risk_score"].mean()

m1, m2, m3 = st.columns(3)
m1.metric("Predicted Fraud Rate", f"{predicted_fraud_rate:.2%}")
m2.metric("Actual Fraud Rate", f"{actual_fraud_rate:.2%}")
m3.metric("Average Risk Score", f"{avg_score:.3f}")
# -----------------------------
# Business insights
# -----------------------------
st.subheader("Business Insights")

high_risk_avg = filtered_df["fraud_risk_score"].mean() if len(filtered_df) > 0 else 0

st.write(f"- Average risk score among filtered claims: **{high_risk_avg:.3f}**")
st.write(f"- Number of claims above selected threshold: **{filtered_claims}**")
st.write(f"- Investigator review list currently shows the top **{min(top_n, filtered_claims)}** claims.")