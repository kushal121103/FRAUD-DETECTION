# Fraud_Detection-Analysis

 ##  **Overview**:
This project provides a comprehensive fraud detection solution combining SQL-based rule detection and machine learning models to identify suspicious transactions. It helps financial institutions, e-commerce platforms, and payment processors detect fraudulent activities in real-time and mitigate risks.

## **Key Features**:
### SQL-Based Fraud Detection Rules
- High-value transactions (> $5,000)
- Multiple transactions in short timeframes
- Device sharing across multiple customers
- Geographical anomalies
- Repeated identical transaction amounts
- Unusual transaction times (midnight to 5 AM)
- Sudden spending spikes (3x above customer average)
- Historical fraud patterns


### Machine Learning Fraud Prediction
- Data preprocessing and feature engineering
- Handles imbalanced data using SMOTE
- Two trained models:
  - Logistic Regression
  - Random Forest
- Performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC


### Risk Analysis Dashboard
- Visualizes transaction patterns
- Shows fraud risk indicators
- Displays historical trends
