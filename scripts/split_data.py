# step4_split_data.py

import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Load the cleaned dataset
data = pd.read_csv("creditcard_clean.csv")

# Step 2: Separate features (X) and target (y)
X = data.drop('Class', axis=1)
y = data['Class']

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("First 5 target values:\n", y.head())

# Step 3: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 4: Save the splits to new files
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# Step 5: Verify
print("\nTraining set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)
print("\nData split completed and saved successfully!")
