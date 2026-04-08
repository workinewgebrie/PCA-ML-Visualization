import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# Check for missing values
print("Missing values in train:")
print(train_df.isnull().sum())

# Fill missing values
numerical_cols = train_df.select_dtypes(include=[np.number]).columns.drop('id')
categorical_cols = train_df.select_dtypes(include=['object']).columns

for col in numerical_cols:
    if train_df[col].isnull().sum() > 0:
        train_df[col].fillna(train_df[col].mean(), inplace=True)
        test_df[col].fillna(test_df[col].mean(), inplace=True)

for col in categorical_cols:
    if train_df[col].isnull().sum() > 0:
        mode_val = train_df[col].mode()[0]
        train_df[col].fillna(mode_val, inplace=True)
        test_df[col].fillna(mode_val, inplace=True)

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    label_encoders[col] = le

# Separate features and target
X = train_df.drop(['id', 'diagnosed_diabetes'], axis=1)
y = train_df['diagnosed_diabetes']

# Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test_df.drop('id', axis=1))

# Apply PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y_train, palette='viridis')
plt.title('PCA Visualization of Training Data')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig('pca_visualization.png')
plt.show()

print("Explained variance ratio:", pca.explained_variance_ratio_)

# Build the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)
print("Model trained successfully.")

# Evaluate the model
y_pred = model.predict(X_val_scaled)
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Classification Report:")
print(classification_report(y_val, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))

# Predict on test set
test_predictions = model.predict(X_test_scaled)

# Create submission
submission = pd.DataFrame({'id': test_df['id'], 'diagnosed_diabetes': test_predictions})
submission.to_csv('data/submission.csv', index=False)
print("Submission file created.")