import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score

import matplotlib.pyplot as plt

# -----------------------------
# Load dataset (FIXED)
# -----------------------------
df = pd.read_csv("bank.csv")   # comma-separated

print(df.head())
print(df.info())

# -----------------------------
# Data Cleaning
# -----------------------------
df.replace("unknown", np.nan, inplace=True)
df.dropna(inplace=True)

# Target column is "deposit"
df['deposit'] = df['deposit'].map({'yes': 1, 'no': 0})

X = df.drop("deposit", axis=1)
y = df["deposit"]

# -----------------------------
# Encoding
# -----------------------------
categorical_cols = X.select_dtypes(include="object").columns
numerical_cols = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# -----------------------------
# Decision Tree Model
# -----------------------------
model = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

model.fit(X_train_transformed, y_train)

# -----------------------------
# Tree Visualization
# -----------------------------
plt.figure(figsize=(22, 10))
plot_tree(
    model,
    filled=True,
    feature_names=preprocessor.get_feature_names_out(),
    class_names=["No", "Yes"]
)
plt.show()

# -----------------------------
# Evaluation
# -----------------------------
y_train_pred = model.predict(X_train_transformed)
y_test_pred = model.predict(X_test_transformed)

print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))
