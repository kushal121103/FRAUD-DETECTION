import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

# Try importing SMOTE safely
try:
    from imblearn.over_sampling import SMOTE
    smote_available = True
except ModuleNotFoundError:
    print("⚠ imblearn not installed — proceeding without SMOTE balancing.")
    smote_available = False

# Directly load your CSV
csv_path = r"C:\Users\yadav\Documents\dataset\Credit Card Fraud Risk Analysis (1).csv"
df = pd.read_csv(csv_path)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Drop irrelevant columns if they exist
for col in ["Transaction ID", "Customer Name", "Merchant Name"]:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# Convert date and extract features
if "Transaction Date" in df.columns:
    df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce")
    df["Transaction_Year"] = df["Transaction Date"].dt.year
    df["Transaction_Month"] = df["Transaction Date"].dt.month
    df["Transaction_Day"] = df["Transaction Date"].dt.day
    df["Transaction_Weekday"] = df["Transaction Date"].dt.weekday
    df.drop(columns=["Transaction Date"], inplace=True)

# Encode categorical variables
le = LabelEncoder()
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = le.fit_transform(df[col].astype(str))

# Features & target
if "IsFraud" not in df.columns:
    raise ValueError("'IsFraud' column not found in dataset")
X = df.drop(columns=["IsFraud"])
y = df["IsFraud"]

# Apply SMOTE if available
if smote_available:
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
y_pred_prob_log = log_reg.predict_proba(X_test)[:, 1]

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_pred_prob_rf = rf.predict_proba(X_test)[:, 1]

# Function to print metrics
def evaluate_model(name, y_true, y_pred, y_prob):
    print(f"\n{name} Metrics:")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred), 4))
    print("Recall:", round(recall_score(y_true, y_pred), 4))
    print("F1-score:", round(f1_score(y_true, y_pred), 4))
    print("ROC-AUC:", round(roc_auc_score(y_true, y_prob), 4))

# Evaluate
evaluate_model("Logistic Regression", y_test, y_pred_log, y_pred_prob_log)
evaluate_model("Random Forest", y_test, y_pred_rf, y_pred_prob_rf)

# ROC Curve
fpr_log, tpr_log, _ = roc_curve(y_test, y_pred_prob_log)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, label=f"Logistic Regression (AUC = {roc_auc_score(y_test, y_pred_prob_log):.2f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_score(y_test, y_pred_prob_rf):.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Fraud Detection")
plt.legend()
plt.grid()
plt.show()

# Confusion Matrices
cm_log = confusion_matrix(y_test, y_pred_log)
ConfusionMatrixDisplay(cm_log, display_labels=["Not Fraud", "Fraud"]).plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

cm_rf = confusion_matrix(y_test, y_pred_rf)
ConfusionMatrixDisplay(cm_rf, display_labels=["Not Fraud", "Fraud"]).plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Random Forest")
plt.show()
