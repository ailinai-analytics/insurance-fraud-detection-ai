import os
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

print("Loading model and data...")

os.makedirs("outputs", exist_ok=True)

# Load trained model
model = joblib.load("outputs/fraud_risk_model.joblib")

# Load cleaned data
df = pd.read_csv("outputs/insurance_clean.csv")

print("\nColumns in insurance_clean.csv:")
print(df.columns.tolist())


def add_engineered_features(df):
    df = df.copy()

    # Convert date columns
    date_cols = ["TXN_DATE_TIME", "POLICY_EFF_DT", "LOSS_DT", "REPORT_DT", "DATE_OF_JOINING"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Date parts
    for col in date_cols:
        if col in df.columns:
            df[f"{col}_YEAR"] = df[col].dt.year
            df[f"{col}_MONTH"] = df[col].dt.month
            df[f"{col}_DAY"] = df[col].dt.day

    # Time difference features
    if "REPORT_DT" in df.columns and "LOSS_DT" in df.columns:
        df["REPORT_DELAY_DAYS"] = (df["REPORT_DT"] - df["LOSS_DT"]).dt.days

    if "LOSS_DT" in df.columns and "POLICY_EFF_DT" in df.columns:
        df["POLICY_AGE_DAYS"] = (df["LOSS_DT"] - df["POLICY_EFF_DT"]).dt.days

    if "LOSS_DT" in df.columns and "DATE_OF_JOINING" in df.columns:
        df["AGENT_EXPERIENCE_DAYS"] = (df["LOSS_DT"] - df["DATE_OF_JOINING"]).dt.days

    # Ratio features
    if "CLAIM_AMOUNT" in df.columns and "PREMIUM_AMOUNT" in df.columns:
        df["CLAIM_TO_PREMIUM_RATIO"] = (
            df["CLAIM_AMOUNT"] / df["PREMIUM_AMOUNT"].replace(0, np.nan)
        )

    if "CLAIM_AMOUNT" in df.columns and "AGE" in df.columns:
        df["CLAIM_PER_AGE"] = (
            df["CLAIM_AMOUNT"] / df["AGE"].replace(0, np.nan)
        )

    if "CLAIM_AMOUNT" in df.columns and "NO_OF_FAMILY_MEMBERS" in df.columns:
        df["CLAIM_PER_FAMILY"] = (
            df["CLAIM_AMOUNT"] / df["NO_OF_FAMILY_MEMBERS"].replace(0, np.nan)
        )

    # Flags
    if "INCIDENT_HOUR_OF_THE_DAY" in df.columns:
        df["NIGHT_INCIDENT"] = (
            (df["INCIDENT_HOUR_OF_THE_DAY"] < 6) | (df["INCIDENT_HOUR_OF_THE_DAY"] > 22)
        ).astype(int)

    if "CLAIM_AMOUNT" in df.columns:
        threshold = df["CLAIM_AMOUNT"].median()
        df["HIGH_CLAIM_FLAG"] = (df["CLAIM_AMOUNT"] > threshold).astype(int)

    return df


print("\nAdding engineered features...")
df = add_engineered_features(df)

target = "target_label"

# Match training logic
X = df.drop(columns=[target, "CLAIM_STATUS"], errors="ignore")

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

print("\nColumns expected by model:")
print(preprocessor.feature_names_in_)

missing = set(preprocessor.feature_names_in_) - set(X.columns)
print("\nMissing columns after engineering:")
print(missing)

# Reorder exactly as model expects
X = X.reindex(columns=preprocessor.feature_names_in_)

# Small sample
sample_X = X.sample(min(100, len(X)), random_state=42)

print("\nPreparing transformed data...")
X_transformed = preprocessor.transform(sample_X)

feature_names = preprocessor.get_feature_names_out()

print("\nRunning SHAP explainer...")
explainer = shap.TreeExplainer(classifier)

try:
    shap_values = explainer.shap_values(X_transformed)
except Exception:
    shap_values = explainer(X_transformed)

print("\nCreating SHAP summary plot...")
plt.figure()
shap.summary_plot(
    shap_values,
    X_transformed,
    feature_names=feature_names,
    show=False
)
plt.tight_layout()
plt.savefig("outputs/shap_summary.png", bbox_inches="tight")

print("\nDone!")
print("SHAP summary plot saved to outputs/shap_summary.png")