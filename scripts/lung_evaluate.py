import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


DATASET_PATH = "datasets/" 
MODEL_SAVE_PATH = "saved_models/lung_disease_model.h5"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

model = load_model(MODEL_SAVE_PATH)
print("✅ Lung disease model loaded successfully!")

datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',  
    subset='validation',
    shuffle=False
)


y_pred_probs = model.predict(val_generator)           
y_pred = np.argmax(y_pred_probs, axis=1)              
y_true = val_generator.classes                         
class_labels = list(val_generator.class_indices.keys())  


cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()


if y_pred_probs.shape[1] == 2:
    y_score = y_pred_probs[:, 1]
else:
    y_score = y_pred_probs.ravel()  

fpr, tpr, thresholds = roc_curve(y_true, y_score)
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


from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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