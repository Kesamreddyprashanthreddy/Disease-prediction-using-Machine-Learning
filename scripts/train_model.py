import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model


model = load_model("saved_model/lung_disease_model.h5")


model.summary()
import os

DATASET_PATH = "datasets/"
MODEL_SAVE_PATH = "saved_model/lung_disease_model.h5"


datagen = ImageDataGenerator(
    rescale=1./255, 
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)


model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  
])


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


model.fit(train_generator, validation_data=val_generator, epochs=10)


if not os.path.exists("saved_model"):
    os.makedirs("saved_model")

model.save(MODEL_SAVE_PATH)
print(f"Model saved at {MODEL_SAVE_PATH}")
model.save("models/lung_disease_model.h5")  


