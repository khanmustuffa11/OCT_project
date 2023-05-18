import tensorflow as tf
from tqdm import tqdm
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import glob
import os

model = load_model('model_xray_base_8class006--0.523767--0.977798--0.584149--0.961879.h5', compile=False)

# Load input image
img_path = 'C:\\Users\\mkhan\\Desktop\\musabi\\OCT_project\\oct_internal_images_for_testing\\machine_artelus\\cnvm\\P19925_M P Sharma_13-04-2022_MACULA_C_OD_11-35-48_AM_1.tif'
img_name = img_path.split('/')[-1]  # Get image name from path

def get_image(file_path):
    img = Image.open(file_path).convert('RGB')
    img = np.asarray(img.resize((500,500)))
    img = tf.cast(img, tf.float32)
    img = np.expand_dims(img, axis=0)
    return img

# Predict class probabilities
probs = model.predict(get_image(img_path))

# Create dictionary of class probabilities with image name as key
class_names = ['Normal', 'Drusen', 'DME', 'CNV', 'AMD', 'CSR', 'DR', 'MH']
probs_dict = {img_name: {class_names[i]: float(class_prob) for i, class_prob in enumerate(probs[0])}}
print(probs_dict)