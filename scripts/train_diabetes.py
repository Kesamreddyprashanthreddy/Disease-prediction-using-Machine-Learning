

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_fscore_support, roc_curve, roc_auc_score
)
from imblearn.over_sampling import SMOTE

print("\n--- Diabetes Model Training Started ---\n")


try:
    df = pd.read_csv("diabetes_disease.csv")
    print(" Successfully loaded diabetes_disease.csv")
except FileNotFoundError:
    print("❌ Error: 'diabetes_disease.csv' not found!")
    exit()


zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for column in zero_not_accepted:
    df[column] = df[column].replace(0, np.nan)
    mean = int(df[column].mean(skipna=True))
    df[column] = df[column].fillna(mean)
print("Preprocessing complete.")


X = df.iloc[:, 1:9]
y = df.iloc[:, 9]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(" Data split into training and testing sets.")

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print("Feature scaling complete.")


sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)
print(" Applied SMOTE - Balanced Classes:", np.bincount(y_train))


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1
)

print("🔍 Hyperparameter tuning in progress...")
grid_search.fit(X_train, y_train)
classifier = grid_search.best_estimator_
print(f" Best Parameters: {grid_search.best_params_}")


classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy*100:.2f}%")


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=["No Diabetes", "Diabetes"],
            yticklabels=["No Diabetes", "Diabetes"])
plt.title("Confusion Matrix", fontsize=14)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()


fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.show()


print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))


precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
metrics_df = pd.DataFrame({
    'Class': ["No Diabetes", "Diabetes"],
    'Precision': precision,
    'Recall': recall,
    'F1-score': f1,
    'Support': support
})
print("\n--- Detailed Metrics Table ---")
print(metrics_df)

metrics_df.set_index('Class')[['Precision', 'Recall', 'F1-score']].plot(
    kind='bar', figsize=(8, 5))
plt.title("Precision, Recall, and F1-score per Class")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


output_dir = "saved_models"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_path = os.path.join(output_dir, "diabetes_model_optimized.joblib")
scaler_path = os.path.join(output_dir, "diabetes_scaler.joblib")

joblib.dump(classifier, model_path)
joblib.dump(sc_X, scaler_path)

print(f"\nModel saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")
print("\n--- Process Finished ---")








# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
# import os

# print("--- Diabetes Model Training Started ---")


# try:
#     df = pd.read_csv('diabetes_disease.csv')
#     print("Successfully loaded diabetes_disease.csv")
# except FileNotFoundError:
#     print("Error: 'diabetes_disease.csv' not found. Make sure it's in the same directory.")
#     exit()


# zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
# for column in zero_not_accepted:
#     df[column] = df[column].replace(0, np.nan)
#     mean = int(df[column].mean(skipna=True))
#     df[column] = df[column].fillna(mean)
# print("Preprocessing complete.")


# X = df.iloc[:, 1:9]
# y = df.iloc[:, 9]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# print("Data split into training and testing sets.")


# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# print("Feature scaling complete.")


# classifier = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')
# classifier.fit(X_train, y_train)
# print("Model training complete.")


# y_pred = classifier.predict(X_test)
# y_prob = classifier.predict_proba(X_test)[:, 1]  


# print(f"Model Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")


# cm = confusion_matrix(y_test, y_pred)

# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
#             xticklabels=["No Diabetes", "Diabetes"],
#             yticklabels=["No Diabetes", "Diabetes"])
# plt.title("Confusion Matrix", fontsize=14)
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.tight_layout()
# plt.show()


# fpr, tpr, thresholds = roc_curve(y_test, y_prob)
# roc_auc = roc_auc_score(y_test, y_prob)

# plt.figure(figsize=(7, 5))
# plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {roc_auc:.2f})")
# plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend(loc="lower right")
# plt.grid()
# plt.tight_layout()
# plt.show()


# print("\n--- Saving the trained model and scaler ---")
# output_dir = "saved_models"
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# model_path = os.path.join(output_dir, "diabetes_model.joblib")
# scaler_path = os.path.join(output_dir, "diabetes_scaler.joblib")

# joblib.dump(classifier, model_path)
# joblib.dump(sc_X, scaler_path)

# print(f"✅ Model saved to: {model_path}")
# print(f"✅ Scaler saved to: {scaler_path}")
# print("\n--- Process Finished ---")


# from sklearn.metrics import classification_report, precision_recall_fscore_support

# print("\n--- Classification Report ---")
# print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))


# precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)


# metrics_df = pd.DataFrame({
#     'Class': ["No Diabetes", "Diabetes"],
#     'Precision': precision,
#     'Recall': recall,
#     'F1-score': f1,
#     'Support': support
# })

# print("\n--- Detailed Metrics Table ---")
# print(metrics_df)

# metrics_df.set_index('Class')[['Precision', 'Recall', 'F1-score']].plot(kind='bar', figsize=(8, 5))
# plt.title("Precision, Recall, and F1-score per Class")
# plt.ylabel("Score")
# plt.ylim(0, 1)
# plt.grid(axis='y')
# plt.tight_layout()
# plt.show()