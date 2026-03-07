import os
import pandas as pd
import matplotlib.pyplot as plt

print("Loading datasets...")

insurance = pd.read_csv("data/insurance_data.csv")
employee = pd.read_csv("data/employee_data.csv")
vendor = pd.read_csv("data/vendor_data.csv")

print("\nDataset Shapes")
print("Insurance:", insurance.shape)
print("Employee:", employee.shape)
print("Vendor:", vendor.shape)

print("\nInsurance Columns:")
for col in insurance.columns:
    print(col)

print("\nFirst 5 rows of insurance data:")
print(insurance.head())

# -----------------------------
# Check duplicates
# -----------------------------
print("\nDuplicate rows:")
print(insurance.duplicated().sum())

insurance = insurance.drop_duplicates()

# -----------------------------
# Missing values
# -----------------------------
print("\nMissing Values:")
print(insurance.isnull().sum().sort_values(ascending=False))

# -----------------------------
# Light cleaning
# -----------------------------
if "ADDRESS_LINE2" in insurance.columns:
    insurance = insurance.drop(columns=["ADDRESS_LINE2"])

if "VENDOR_ID" in insurance.columns:
    insurance["VENDOR_ID"] = insurance["VENDOR_ID"].fillna("UNKNOWN")

if "AUTHORITY_CONTACTED" in insurance.columns:
    insurance["AUTHORITY_CONTACTED"] = insurance["AUTHORITY_CONTACTED"].fillna("UNKNOWN")

if "CUSTOMER_EDUCATION_LEVEL" in insurance.columns:
    insurance["CUSTOMER_EDUCATION_LEVEL"] = insurance["CUSTOMER_EDUCATION_LEVEL"].fillna("UNKNOWN")

if "CITY" in insurance.columns:
    insurance["CITY"] = insurance["CITY"].fillna("UNKNOWN")

if "INCIDENT_CITY" in insurance.columns:
    insurance["INCIDENT_CITY"] = insurance["INCIDENT_CITY"].fillna("UNKNOWN")

# -----------------------------
# Target label
# -----------------------------
print("\nCLAIM_STATUS distribution:")
print(insurance["CLAIM_STATUS"].value_counts())

insurance["target_label"] = insurance["CLAIM_STATUS"].map({"A": 0, "D": 1})

print("\nTarget Label Distribution:")
print(insurance["target_label"].value_counts())

# -----------------------------
# Claim status plot
# -----------------------------
insurance["CLAIM_STATUS"].value_counts().plot(kind="bar")
plt.title("Claim Status Distribution")
plt.xlabel("Claim Status")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# -----------------------------
# Claim amount plot
# -----------------------------
insurance["CLAIM_AMOUNT"].hist(bins=40)
plt.title("Claim Amount Distribution")
plt.xlabel("Claim Amount")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# -----------------------------
# Top agents
# -----------------------------
top_agents = insurance["AGENT_ID"].value_counts().head(10)
print("\nTop Agents by Claim Count")
print(top_agents)

top_agents.plot(kind="bar")
plt.title("Top Agents by Claim Count")
plt.xlabel("Agent ID")
plt.ylabel("Number of Claims")
plt.tight_layout()
plt.show()

# -----------------------------
# Top vendors
# -----------------------------
top_vendors = insurance["VENDOR_ID"].value_counts().head(10)
print("\nTop Vendors by Claim Count")
print(top_vendors)

top_vendors.plot(kind="bar")
plt.title("Top Vendors by Claim Count")
plt.xlabel("Vendor ID")
plt.ylabel("Number of Claims")
plt.tight_layout()
plt.show()

# -----------------------------
# Merge employee data
# -----------------------------
merged_df = insurance.merge(employee, on="AGENT_ID", how="left")

# -----------------------------
# Merge vendor data
# -----------------------------
merged_df = merged_df.merge(vendor, on="VENDOR_ID", how="left")

print("\nMerged dataset shape:", merged_df.shape)

# -----------------------------
# Save cleaned dataset
# -----------------------------
os.makedirs("outputs", exist_ok=True)
merged_df.to_csv("outputs/insurance_clean.csv", index=False)

print("\nClean dataset saved to outputs/insurance_clean.csv")


df = pd.read_csv("outputs/insurance_clean.csv")

print("\nColumns in insurance_clean.csv:")
print(df.columns.tolist())