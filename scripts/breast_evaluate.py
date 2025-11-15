import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DATASET_PATH = "mias_images_png/"  
MODEL_SAVE_PATH = "saved_models/breast_cancer_image_model.h5"
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 16

model = load_model(MODEL_SAVE_PATH)
print("✅ Model loaded successfully!")


datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)


val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)


y_pred_probs = model.predict(val_generator)      
y_pred = (y_pred_probs > 0.5).astype(int)       
y_true = val_generator.classes                  


cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()


fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()

print(f"✅ ROC-AUC Score: {roc_auc:.4f}")
from sklearn.metrics import classification_report, precision_recall_fscore_support
import pandas as pd
import seaborn as sns

# Define class labels
class_labels = ["Benign", "Malignant"]

# --- Classification Report ---
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_labels))

# --- Precision, Recall, F1-score, Support ---
precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)

metrics_df = pd.DataFrame({
    'Class': class_labels,
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
