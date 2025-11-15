# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve,auc
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Flatten, Dense, Dropout
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.optimizers import Adam 
# from tensorflow.keras.callbacks import EarlyStopping 
# import os

# print("--- Breast Cancer Image Model Training Started (with Transfer Learning & Stable Training) ---")


# DATASET_PATH = "mias_images_png/" 
# MODEL_SAVE_PATH = "saved_models/breast_cancer_image_model.h5"
# IMAGE_SIZE = (150, 150)

# datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )


# train_generator = datagen.flow_from_directory(
#     DATASET_PATH,
#     target_size=IMAGE_SIZE,
#     batch_size=16,
#     class_mode='binary',
#     subset='training'
# )

# val_generator = datagen.flow_from_directory(
#     DATASET_PATH,
#     target_size=IMAGE_SIZE,
#     batch_size=16,
#     class_mode='binary',
#     subset='validation'
# )


# base_model = VGG16(
#     weights='imagenet', 
#     include_top=False, 
#     input_shape=IMAGE_SIZE + (3,)
# )


# base_model.trainable = False


# model = Sequential([
#     base_model,
#     Flatten(),
    
#     Dense(256, activation='relu'), 
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')
# ])


# model.compile(
    
#     optimizer=Adam(learning_rate=0.0001), 
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )

# model.summary()


# early_stopping = EarlyStopping(
#     monitor='val_accuracy', 
#     patience=5,             
#     restore_best_weights=True 
# )


# print("\n--- Training the new classifier head... ---")
# history = model.fit(
#     train_generator,
#     validation_data=val_generator,
#     epochs=50, 
#     callbacks=[early_stopping]
# )


# if not os.path.exists("saved_models"):
#     os.makedirs("saved_models")
    
# model.save(MODEL_SAVE_PATH)
# print(f"\n✅ New breast cancer model saved at {MODEL_SAVE_PATH}")


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc,
    classification_report, precision_recall_fscore_support
)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from collections import Counter
import math
import joblib


DATASET_PATH = "mias_images_png/"   
OUTPUT_DIR = "saved_models"
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "breast_cancer_image_model_improved.h5")
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 16
INITIAL_EPOCHS = 12
FINE_TUNE_EPOCHS = 12
SEED = 42
# -------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# 1) Data generators (stronger augmentation + validation split)
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True,
    seed=SEED
)

val_generator = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)


train_counts = Counter(train_generator.classes)
val_counts = Counter(val_generator.classes)
print("Train class distribution:", train_counts)
print("Val class distribution:", val_counts)

#
total = sum(train_counts.values())
class_weight = {}
for cls, cnt in train_counts.items():
    # inverse frequency
    class_weight[cls] = total / (len(train_counts) * cnt)
print("Class weights:", class_weight)


base_model = VGG16(weights='imagenet', include_top=False, input_shape=IMAGE_SIZE + (3,))
base_model.trainable = False 


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.35)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

checkpoint_path = os.path.join(OUTPUT_DIR, "best_breast_model.h5")
callbacks = [
    ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-7),
    EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1)
]


steps_per_epoch = max(1, math.ceil(train_generator.n / BATCH_SIZE))
validation_steps = max(1, math.ceil(val_generator.n / BATCH_SIZE))

print("\n--- Training classifier head (base frozen) ---")
history1 = model.fit(
    train_generator,
    epochs=INITIAL_EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)


unfreeze_from = -8  
for layer in base_model.layers[unfreeze_from:]:
    layer.trainable = True


model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(f"\n--- Fine-tuning from base_model.layers[{unfreeze_from}:] (very low LR) ---")
history2 = model.fit(
    train_generator,
    epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
    initial_epoch=history1.epoch[-1] + 1 if len(history1.epoch) > 0 else 0,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# load best weights saved by ModelCheckpoint (if present)
if os.path.exists(checkpoint_path):
    print("\nLoading best weights from checkpoint.")
    model.load_weights(checkpoint_path)

# 7) Save final model
model.save(MODEL_SAVE_PATH)
print(f"\n✅ Improved model saved to: {MODEL_SAVE_PATH}")

# 8) Evaluate on validation set (detailed metrics)
val_generator.reset()
y_pred_probs = model.predict(val_generator, steps=validation_steps, verbose=1)
# If predict returns shape (N,1), flatten and take first n samples equal to val_generator.n
y_pred_probs = y_pred_probs.ravel()[:val_generator.n]
y_pred = (y_pred_probs > 0.5).astype(int)

y_true = val_generator.classes  # length = val_generator.n


class_labels = [k for k, v in sorted(val_generator.class_indices.items(), key=lambda kv: kv[1])]
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_labels))

precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
import pandas as pd
metrics_df = pd.DataFrame({
    'Class': class_labels,
    'Precision': precision,
    'Recall': recall,
    'F1-score': f1,
    'Support': support
})
print("\n--- Metrics Table ---")
print(metrics_df)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()


fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()
print(f"\n✅ Validation ROC-AUC: {roc_auc:.4f}")


metrics_csv = os.path.join(OUTPUT_DIR, "breast_metrics_table.csv")
metrics_df.to_csv(metrics_csv, index=False)
print(f"Saved metrics table to: {metrics_csv}")



print("\n--- Done ---\n")
