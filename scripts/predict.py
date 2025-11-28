
from tensorflow.keras.models import load_model
MODEL_PATH = "models/lung_disease_model.h5"
model = load_model(MODEL_PATH)
model.summary()