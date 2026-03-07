import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline as SkPipeline

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

print("Loading cleaned dataset...")

df = pd.read_csv("outputs/insurance_clean.csv")

print("Dataset shape:", df.shape)

# -----------------------------
# Target column
# -----------------------------
target = "target_label"

# -----------------------------
# Convert date columns
# -----------------------------
date_cols = ["TXN_DATE_TIME", "POLICY_EFF_DT", "LOSS_DT", "REPORT_DT", "DATE_OF_JOINING"]

for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

# -----------------------------
# Feature engineering
# -----------------------------
if "CLAIM_AMOUNT" in df.columns and "PREMIUM_AMOUNT" in df.columns:
    df["CLAIM_TO_PREMIUM_RATIO"] = df["CLAIM_AMOUNT"] / (df["PREMIUM_AMOUNT"] + 1)

if "LOSS_DT" in df.columns and "REPORT_DT" in df.columns:
    df["REPORT_DELAY_DAYS"] = (df["REPORT_DT"] - df["LOSS_DT"]).dt.days

if "POLICY_EFF_DT" in df.columns and "LOSS_DT" in df.columns:
    df["POLICY_AGE_DAYS"] = (df["LOSS_DT"] - df["POLICY_EFF_DT"]).dt.days

if "DATE_OF_JOINING" in df.columns and "LOSS_DT" in df.columns:
    df["AGENT_EXPERIENCE_DAYS"] = (df["LOSS_DT"] - df["DATE_OF_JOINING"]).dt.days

if "CLAIM_AMOUNT" in df.columns and "AGE" in df.columns:
    df["CLAIM_PER_AGE"] = df["CLAIM_AMOUNT"] / (df["AGE"] + 1)

if "CLAIM_AMOUNT" in df.columns and "NO_OF_FAMILY_MEMBERS" in df.columns:
    df["CLAIM_PER_FAMILY"] = df["CLAIM_AMOUNT"] / (df["NO_OF_FAMILY_MEMBERS"] + 1)

if "INCIDENT_HOUR_OF_THE_DAY" in df.columns:
    df["NIGHT_INCIDENT"] = (df["INCIDENT_HOUR_OF_THE_DAY"] >= 22).astype(int)

if "CLAIM_AMOUNT" in df.columns:
    df["HIGH_CLAIM_FLAG"] = (df["CLAIM_AMOUNT"] > 15000).astype(int)

# Extract date parts
for col in date_cols:
    if col in df.columns:
        df[f"{col}_YEAR"] = df[col].dt.year
        df[f"{col}_MONTH"] = df[col].dt.month
        df[f"{col}_DAY"] = df[col].dt.day

# -----------------------------
# Drop leakage / privacy / weak columns
# -----------------------------
drop_cols = [
    target,
    "CLAIM_STATUS",
    "TRANSACTION_ID",
    "CUSTOMER_ID",
    "POLICY_NUMBER",
    "CUSTOMER_NAME",
    "SSN",
    "ROUTING_NUMBER",
    "ACCT_NUMBER",
    "EMP_ROUTING_NUMBER",
    "EMP_ACCT_NUMBER",
    "ADDRESS_LINE1_x",
    "ADDRESS_LINE2_x",
    "ADDRESS_LINE1_y",
    "ADDRESS_LINE2_y",
    "ADDRESS_LINE1",
    "ADDRESS_LINE2",
    "POSTAL_CODE_x",
    "POSTAL_CODE_y",
    "POSTAL_CODE",
    "TXN_DATE_TIME",
    "POLICY_EFF_DT",
    "LOSS_DT",
    "REPORT_DT",
    "DATE_OF_JOINING",
    "AGENT_ID",
    "VENDOR_ID",
    "AGENT_NAME",
    "VENDOR_NAME"
]

existing_drop_cols = [col for col in drop_cols if col in df.columns]

X = df.drop(columns=existing_drop_cols)
y = df[target]

# -----------------------------
# Clean impossible numeric values
# -----------------------------
for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        X[col] = X[col].replace([np.inf, -np.inf], np.nan)

# -----------------------------
# Identify numeric and categorical columns
# -----------------------------
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

print("\nNumeric features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

# -----------------------------
# Preprocessing
# -----------------------------
numeric_transformer = SkPipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = SkPipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -----------------------------
# XGBoost model pipeline with SMOTE
# -----------------------------
model = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("classifier", XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=18,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False
    ))
])

print("\nTraining model...")
model.fit(X_train, y_train)

# -----------------------------
# Probability predictions
# -----------------------------
y_prob = model.predict_proba(X_test)[:, 1]

# Lower threshold for better suspicious-claim recall
threshold = 0.20
y_pred = (y_prob >= threshold).astype(int)

print(f"\nUsing decision threshold: {threshold}")

# -----------------------------
# Evaluation
# -----------------------------
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_prob))

# -----------------------------
# Save outputs
# -----------------------------
os.makedirs("outputs", exist_ok=True)

results = X_test.copy()
results["actual_target"] = y_test.values
results["predicted_target"] = y_pred
results["fraud_risk_score"] = y_prob

results.to_csv("outputs/model_predictions.csv", index=False)

# Save trained model
joblib.dump(model, "outputs/fraud_risk_model.joblib")

print("\nPrediction file saved to outputs/model_predictions.csv")
print("Trained model saved to outputs/fraud_risk_model.joblib")

# -----------------------------
# Save feature importance
# -----------------------------
try:
    classifier = model.named_steps["classifier"]
    importances = classifier.feature_importances_

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    feat_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    feat_imp.to_csv("outputs/feature_importance.csv", index=False)
    print("Feature importance saved to outputs/feature_importance.csv")
except Exception as e:
    print(f"Could not save feature importance: {e}")