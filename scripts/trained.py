# train_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# -------------------------
# Step 1: Load cleaned dataset
# -------------------------
data_path = "C:/Users/Arokiya Nithish/Downloads/Fraud_Detection_Project/data/creditcard_clean.csv"
data = pd.read_csv(data_path)
print("✅ Cleaned dataset loaded successfully. Total records:", data.shape[0])

# Separate features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# -------------------------
# Step 2: Split data into train/test sets (stratified)
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Data split into training ({X_train.shape[0]}) and test ({X_test.shape[0]}) sets.")

# -------------------------
# Step 3: Handle imbalanced data using SMOTE
# -------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"✅ After SMOTE, training set size: {X_train_res.shape[0]}")

# -------------------------
# Step 4: Train Random Forest with class_weight
# -------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train_res, y_train_res)
print("✅ Random Forest model trained successfully!")

# -------------------------
# Step 5: Evaluate model on test data
# -------------------------
y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n===== Model Evaluation =====")
print(f"Accuracy: {accuracy*100:.2f}%")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# -------------------------
# Step 6: Save trained model & scaler
# -------------------------
model_path = "C:/Users/Arokiya Nithish/Downloads/Fraud_Detection_Project/scripts/random_forest_model.pkl"
joblib.dump(rf_model, model_path)
print(f"✅ Trained model saved at: {model_path}")

# Save the scaler if needed for real-time predictions
scaler_path = "C:/Users/Arokiya Nithish/Downloads/Fraud_Detection_Project/scripts/scaler.pkl"
scaler = StandardScaler()  # Just in case for new transactions
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler saved at: {scaler_path}")
