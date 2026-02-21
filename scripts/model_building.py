# step6_model_building.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load the datasets
X_train = pd.read_csv("X_train_balanced.csv")
y_train = pd.read_csv("y_train_balanced.csv")

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

print("Datasets loaded successfully!")
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

# Step 2: Initialize and train Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train.values.ravel())

print("\nModel training completed!")

# Step 3: Make predictions
y_pred = model.predict(X_test)

# Step 4: Evaluate model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
