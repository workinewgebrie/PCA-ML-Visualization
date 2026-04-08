# =============================
# PCA + ML PIPELINE (WORKING)
# =============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# -----------------------------
# Load data
# -----------------------------
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# -----------------------------
# Handle missing values
# -----------------------------
numerical_cols = train_df.select_dtypes(include=[np.number]).columns.drop("id")
categorical_cols = train_df.select_dtypes(include=["object"]).columns

for col in numerical_cols:
    train_df[col].fillna(train_df[col].mean(), inplace=True)
    test_df[col].fillna(test_df[col].mean(), inplace=True)

for col in categorical_cols:
    mode_val = train_df[col].mode()[0]
    train_df[col].fillna(mode_val, inplace=True)
    test_df[col].fillna(mode_val, inplace=True)

# -----------------------------
# Encode categorical variables
# -----------------------------
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    label_encoders[col] = le

# -----------------------------
# Features & target
# -----------------------------
X = train_df.drop(columns=["id", "diagnosed_diabetes"])
y = train_df["diagnosed_diabetes"]

# -----------------------------
# Train-validation split
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test_df.drop(columns=["id"]))

# =====================================================
# PCA FOR VISUALIZATION
# =====================================================
pca_vis = PCA(n_components=2)
X_train_pca_vis = pca_vis.fit_transform(X_train_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_train_pca_vis[:, 0],
    y=X_train_pca_vis[:, 1],
    hue=y_train,
    palette="viridis",
    legend="full",
    alpha=0.7
)
plt.title("PCA Visualization of Training Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.savefig("pca_visualization.png")
plt.show()

print("Explained variance ratio (2 PCs):", pca_vis.explained_variance_ratio_)

# =====================================================
# MODEL BEFORE PCA
# =====================================================
model_before = RandomForestClassifier(
    n_estimators=100, random_state=42
)
model_before.fit(X_train_scaled, y_train)

y_pred_before = model_before.predict(X_val_scaled)
acc_before = accuracy_score(y_val, y_pred_before)

print("\n=== Model BEFORE PCA ===")
print("Accuracy:", acc_before)
print(classification_report(y_val, y_pred_before))

# =====================================================
# MODEL AFTER PCA (95% variance)
# =====================================================
pca_model = PCA(n_components=0.95)
X_train_pca = pca_model.fit_transform(X_train_scaled)
X_val_pca = pca_model.transform(X_val_scaled)
X_test_pca = pca_model.transform(X_test_scaled)

model_after = RandomForestClassifier(
    n_estimators=100, random_state=42
)
model_after.fit(X_train_pca, y_train)

y_pred_after = model_after.predict(X_val_pca)
acc_after = accuracy_score(y_val, y_pred_after)

print("\n=== Model AFTER PCA ===")
print("Accuracy:", acc_after)
print(classification_report(y_val, y_pred_after))

# =====================================================
# COMPARISON
# =====================================================
print("\nAccuracy Comparison:")
print(f"Before PCA: {acc_before:.4f}")
print(f"After  PCA: {acc_after:.4f}")

# =====================================================
# TEST SET PREDICTION
# =====================================================
test_predictions = model_after.predict(X_test_pca)

submission = pd.DataFrame({
    "id": test_df["id"],
    "diagnosed_diabetes": test_predictions
})
submission.to_csv("data/submission.csv", index=False)

print("\nSubmission file created: data/submission.csv")