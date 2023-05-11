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

input_folder = 'C:\\Users\\mkhan\\Desktop\\musabi\\OCT_project\\data\oct\\oct_artelus\\test\\4\\'


columns = ['filename', 'class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7']
df = pd.DataFrame(columns=columns)

# Load and predict probabilities for each image in input folder
for filename in os.listdir(input_folder):
    # Load input image
    img_path = os.path.join(input_folder, filename)
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = tf.keras.applications.efficientnet.preprocess_input(x)

    # Predict class probabilities
    probs = model.predict(tf.expand_dims(x, axis=0))

    # Append predicted class probabilities to DataFrame
    row = {'filename': filename}
    for i, prob in enumerate(probs[0]):
        row[f'class_{i}'] = prob
    #df = df.append(row, ignore_index=True)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

# Save DataFrame to CSV file
df.to_csv("folder.csv", index=False)








