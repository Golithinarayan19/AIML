# feature_encoding_scaling.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("adult.csv")
print("Dataset Loaded Successfully\n")
print(df.head())

# -------------------------------
# 2. Identify Feature Types
# -------------------------------
categorical_features = df.select_dtypes(include='object').columns
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns

print("\nCategorical Features:")
print(categorical_features)

print("\nNumerical Features:")
print(numerical_features)

# -------------------------------
# 3. Label Encoding (Ordered)
# -------------------------------
le = LabelEncoder()
df['income'] = le.fit_transform(df['income'])

print("\nIncome column after Label Encoding:")
print(df['income'].value_counts())

# -------------------------------
# 4. One-Hot Encoding (No Order)
# -------------------------------
df_encoded = pd.get_dummies(
    df,
    columns=categorical_features.drop('income'),
    drop_first=True
)

print("\nDataset after One-Hot Encoding:")
print(df_encoded.head())

# -------------------------------
# 5. Feature Scaling
# -------------------------------
scaler = StandardScaler()
df_encoded[numerical_features] = scaler.fit_transform(
    df_encoded[numerical_features]
)

print("\nNumerical features after scaling:")
print(df_encoded[numerical_features].describe())

# -------------------------------
# 6. Save Processed Dataset
# -------------------------------
df_encoded.to_csv("adult_preprocessed.csv", index=False)
print("\nPreprocessed dataset saved as adult_preprocessed.csv")

# -------------------------------
# 7. Visualization (Optional)
# -------------------------------
plt.figure()
sns.histplot(df_encoded['age'], kde=True)
plt.title("Age Distribution After Scaling")
plt.show()

print("\n--- Feature Encoding & Scaling Completed Successfully ---")
