import tensorflow as tf
from tensorflow.keras.models import Model, load_model

model = load_model('모델 경로')

print(model.summary())