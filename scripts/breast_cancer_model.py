# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# import os

# # Load breast cancer dataset
# data = load_breast_cancer()
# X = data.data
# y = data.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize the data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Convert labels to categorical (for binary classification)
# y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
# y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

# # Define model architecture
# model = Sequential([
#     Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
#     Dropout(0.2),
#     Dense(32, activation='relu'),
#     Dropout(0.2),
#     Dense(2, activation='softmax')  # 2 classes: malignant/benign
# ])

# # Compile the model
# model.compile(
#     optimizer=Adam(learning_rate=0.001),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# # Train the model
# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_test, y_test),
#     epochs=50,
#     batch_size=32,
#     verbose=1
# )

# # Create directory if it doesn't exist
# if not os.path.exists("saved_model"):
#     os.makedirs("saved_model")

# # Save the trained model
# MODEL_SAVE_PATH = "saved_model/breast_cancer_model.h5"
# model.save(MODEL_SAVE_PATH)
# print(f"Model saved at {MODEL_SAVE_PATH}")

# # Evaluate the model
# loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
# print(f"Test Accuracy: {accuracy*100:.2f}%")
# print(f"Test Loss: {loss:.4f}")

# # Print model summary
# model.summary()

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

IMAGE_FOLDER = "D:\\Python\\final_project (3)\\final_project\\final-project\\breast_cancer"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 10
FINE_TUNE_EPOCHS = 20  
FINE_TUNE_AT = 100    

images = []
labels = []

for filename in os.listdir(IMAGE_FOLDER):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        label = 0 if "benign" in filename.lower() else 1
        img_path = os.path.join(IMAGE_FOLDER, filename)
        img = load_img(img_path, target_size=IMAGE_SIZE)
        img_array = img_to_array(img)/255.0
        images.append(img_array)
        labels.append(label)

images = np.array(images)
labels = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])


checkpoint = ModelCheckpoint('breast_cancer_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=7, verbose=1, restore_best_weights=True)

history_initial = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=INITIAL_EPOCHS,
    steps_per_epoch=len(X_train)//BATCH_SIZE,
    validation_steps=len(X_val)//BATCH_SIZE,
    callbacks=[checkpoint, reduce_lr, early_stop]
)


for layer in base_model.layers[FINE_TUNE_AT:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=FINE_TUNE_EPOCHS,
    steps_per_epoch=len(X_train)//BATCH_SIZE,
    validation_steps=len(X_val)//BATCH_SIZE,
    callbacks=[checkpoint, reduce_lr, early_stop]
)


y_pred_probs = model.predict(X_val)
y_pred = (y_pred_probs > 0.5).astype(int)
roc_auc = roc_auc_score(y_val, y_pred_probs)
print(f"ROC-AUC on validation set: {roc_auc:.4f}")


def predict_image(img_path):
    img = load_img(img_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    return "Malignant" if pred[0][0] > 0.5 else "Benign"

# Example usage:
# print(predict_image("all_images/benign_1.jpg"))
