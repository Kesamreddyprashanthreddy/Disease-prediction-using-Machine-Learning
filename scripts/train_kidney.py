
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

print("--- Kidney Disease Model Training Started ---")


try:
    df = pd.read_csv('kidney_disease.csv')
    print("Successfully loaded kidney_disease.csv")
except FileNotFoundError:
    print("Error: 'kidney_disease.csv' not found. Make sure it's in the same directory.")
    exit()


for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip()


if 'id' in df.columns:
    df = df.drop('id', axis=1)


if 'classification' in df.columns:
    df['classification'] = df['classification'].astype(str).str.lower().str.strip().map({'ckd': 1, 'notckd': 0})
else:
    print("Error: 'classification' column not found.")
    exit()


for col in df.columns:
    if col != 'classification':
        df[col] = pd.to_numeric(df[col].replace(['?', 'na', 'None', 'none', 'nan'], np.nan), errors='coerce')


for col in df.columns:
    median = df[col].median()
    if np.isnan(median):
        df[col] = df[col].fillna(0)
    else:
        df[col] = df[col].fillna(median)


print("Data cleaning and preprocessing complete.")


X = df.drop('classification', axis=1)
y = df['classification']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Feature scaling complete.")


model = RandomForestClassifier(n_estimators=20, random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")


y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

print("\n--- Saving the trained model and scaler ---")
output_dir = "saved_models"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_path = os.path.join(output_dir, "kidney_model.joblib")
scaler_path = os.path.join(output_dir, "kidney_scaler.joblib")

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

print(f"✅ Model saved to: {model_path}")
print(f"✅ Scaler saved to: {scaler_path}")
print("\n--- Process Finished ---")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

try:
    print("\n--- Generating Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    print("\n--- Generating ROC-AUC Curve ---")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_auc_curve.png')
    plt.show()
    
    print("\n✅ ROC-AUC curve and confusion matrix plots generated successfully.")
    
except Exception as e:
    print(f"An error occurred: {e}")

from sklearn.metrics import classification_report, precision_recall_fscore_support

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=["Not CKD", "CKD"]))
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
metrics_df = pd.DataFrame({
    'Class': ["Not CKD", "CKD"],
    'Precision': precision,
    'Recall': recall,
    'F1-score': f1,
    'Support': support
})

print("\n--- Detailed Metrics Table ---")
print(metrics_df)
metrics_df.set_index('Class')[['Precision', 'Recall', 'F1-score']].plot(kind='bar', figsize=(8, 5))
plt.title("Precision, Recall, and F1-score per Class")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
