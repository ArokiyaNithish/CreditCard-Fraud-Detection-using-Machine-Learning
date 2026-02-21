# step3_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

# -------------------------
# Step 1: Load dataset
# -------------------------
data_path = "C:/Users/Arokiya Nithish/Downloads/Fraud_Detection_Project/data/creditcard.csv"
data = pd.read_csv(data_path)

print("✅ Original dataset loaded successfully!")
print("Initial shape:", data.shape)
print("First 5 rows:\n", data.head())

# -------------------------
# Step 2: Initialize scaler
# -------------------------
scaler = StandardScaler()

# -------------------------
# Step 3: Scale the 'Amount' column
# -------------------------
data['Amount_scaled'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

# -------------------------
# Step 4: Drop the original 'Amount' and 'Time' columns
# -------------------------
if 'Amount' in data.columns:
    data.drop(['Amount'], axis=1, inplace=True)
if 'Time' in data.columns:
    data.drop(['Time'], axis=1, inplace=True)

# -------------------------
# Step 5: Save the scaler for future use (real-time predictions)
# -------------------------
scaler_path = "C:/Users/Arokiya Nithish/Downloads/Fraud_Detection_Project/scripts/scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler saved successfully at: {scaler_path}")

# -------------------------
# Step 6: Save the cleaned dataset for next steps
# -------------------------
cleaned_data_path = "C:/Users/Arokiya Nithish/Downloads/Fraud_Detection_Project/data/creditcard_clean.csv"
data.to_csv(cleaned_data_path, index=False)
print(f"✅ Cleaned dataset saved successfully at: {cleaned_data_path}")

# -------------------------
# Step 7: Verify changes
# -------------------------
print("\nFirst 5 rows of cleaned data:\n", data.head())
print("\nColumns in cleaned dataset:", data.columns)
print("\n✅ Preprocessing complete! All V1–V28 features + Amount_scaled are included.")
