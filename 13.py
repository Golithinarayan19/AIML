import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load dataset
digits = load_digits()
X = digits.data          # flattened images (64 features)
y = digits.target

print("Original shape:", X.shape)

# 2. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Apply PCA with different components
components = [2, 10, 30, 50]
explained_variance = []

for n in components:
    pca = PCA(n_components=n)
    pca.fit(X_scaled)
    explained_variance.append(np.sum(pca.explained_variance_ratio_))

# 4. Plot cumulative explained variance
plt.figure()
plt.plot(components, explained_variance, marker='o')
plt.xlabel("Number of PCA Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Explained Variance vs Components")
plt.grid()
plt.show()

# 5. Train Logistic Regression on original data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

lr = LogisticRegression(max_iter=3000)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
original_accuracy = accuracy_score(y_test, y_pred)

# 6. Train Logistic Regression on PCA-reduced data
pca_30 = PCA(n_components=30)
X_pca = pca_30.fit_transform(X_scaled)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)

lr_pca = LogisticRegression(max_iter=3000)
lr_pca.fit(X_train_pca, y_train_pca)
y_pred_pca = lr_pca.predict(X_test_pca)
pca_accuracy = accuracy_score(y_test_pca, y_pred_pca)

# 7. Accuracy comparison
print("\nAccuracy Comparison")
print("------------------")
print(f"Original Data Accuracy : {original_accuracy:.4f}")
print(f"PCA (30 components) Accuracy : {pca_accuracy:.4f}")

# 8. PCA 2D Visualization
pca_2 = PCA(n_components=2)
X_2d = pca_2.fit_transform(X_scaled)

plt.figure()
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, s=15)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("2D PCA Visualization of Digits Dataset")
plt.colorbar(scatter)
plt.show()
