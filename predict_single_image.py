import tensorflow as tf
from tqdm import tqdm
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import glob
import os

model = load_model('oct_mendely_artelus_models/model_artmen.h5', compile=False)

# Load input image
img_path = 'NORMAL-1384-1.jpeg'
img_name = img_path.split('/')[-1]  # Get image name from path
img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
x = tf.keras.preprocessing.image.img_to_array(img)
x = tf.keras.applications.efficientnet.preprocess_input(x)

# Predict class probabilities
probs = model.predict(tf.expand_dims(x, axis=0))

# Create dictionary of class probabilities with image name as key
class_names = ['Normal', 'Drusen', 'DME', 'CNV', 'AMD', 'CSR', 'DR', 'MH']
probs_dict = {img_name: {class_names[i]: float(class_prob) for i, class_prob in enumerate(probs[0])}}
print(probs_dict)