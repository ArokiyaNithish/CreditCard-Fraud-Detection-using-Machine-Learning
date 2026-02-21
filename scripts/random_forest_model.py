# step7_random_forest_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load datasets
X_train = pd.read_csv("X_train_balanced.csv")
y_train = pd.read_csv("y_train_balanced.csv")

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

print("Datasets loaded successfully!")
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

# Step 2: Initialize Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=None,          # No limit on depth
    random_state=42,
    n_jobs=-1,               # Use all CPU cores for speed
    class_weight='balanced'  # Helps with imbalanced data
)

# Step 3: Train the model
print("\nTraining Random Forest model...")
rf_model.fit(X_train, y_train.values.ravel())
print("Random Forest training completed!")

# Step 4: Make predictions
y_pred_rf = rf_model.predict(X_test)

# Step 5: Evaluate model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, digits=4))

# Optional: Save the trained model
import joblib
joblib.dump(rf_model, "random_forest_model.pkl")
print("\nModel saved as random_forest_model.pkl")
