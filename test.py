import os
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

data_model = "model"

batch_size = 32
size = 180

class_names = ['daisy','dandelion', 'roses', 'sunflowers', 'tulips']

num_classes = len(class_names)
print(class_names)

model = keras.models.load_model(data_model)

img_pred_dir = 'prediction/daisy.jpg'

img_pred = keras.preprocessing.image.load_img(
    img_pred_dir, target_size=(size, size)
)

img_pred_array = keras.preprocessing.image.img_to_array(img_pred)
img_pred_array = tf.expand_dims(img_pred_array, 0)

prediction = model.predict(img_pred_array)
score = tf.nn.softmax(prediction[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
