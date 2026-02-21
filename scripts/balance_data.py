# step5_balance_data.py

import pandas as pd
from imblearn.over_sampling import SMOTE

# Step 1: Load training data
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

print("Before SMOTE:")
print(y_train['Class'].value_counts())

# Step 2: Initialize SMOTE
smote = SMOTE(random_state=42)

# Step 3: Apply SMOTE to balance data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 4: Convert back to DataFrame
X_train_resampled = pd.DataFrame(X_train_resampled, columns=X_train.columns)
y_train_resampled = pd.DataFrame(y_train_resampled, columns=['Class'])

# Step 5: Save balanced data
X_train_resampled.to_csv("X_train_balanced.csv", index=False)
y_train_resampled.to_csv("y_train_balanced.csv", index=False)

print("\nAfter SMOTE:")
print(y_train_resampled['Class'].value_counts())

print("\nBalanced training set saved successfully!")
