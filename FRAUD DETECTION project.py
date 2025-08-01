# Import required libraries
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split

# ---------------------------------------------
# 1. Generate synthetic dataset (you can replace this with real data)
X, y = make_classification(n_samples=1000, n_features=20, 
                           n_informative=2, n_redundant=10, 
                           random_state=42)

# Convert to DataFrame for analysis
feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['Target'] = y

# ---------------------------------------------
# 2. Data Analysis

print("🔍 Basic Info:")
print(df.info())
print("\n📊 Descriptive Statistics:")
print(df.describe())
print("\n📈 Class Distribution:")
print(df['Target'].value_counts())

# Plot class distribution
sns.countplot(data=df, x='Target')
plt.title("Class Distribution")
plt.xlabel("Target (0 = No Fraud, 1 = Fraud)")
plt.ylabel("Count")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("🔗 Feature Correlation Heatmap")
plt.show()

# ---------------------------------------------
# 3. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df[feature_names], df['Target'], test_size=0.3, random_state=42
)

# 4. Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Predict probabilities and classes
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# ---------------------------------------------
# 6. Evaluation Metrics

roc_auc = roc_auc_score(y_test, y_prob)
print(f"\n🎯 ROC AUC Score: {roc_auc:.4f}")

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("🧮 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ---------------------------------------------
# 7. Plot ROC Curve

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", color='blue')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("📈 ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
