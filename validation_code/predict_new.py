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
#model = load_model('oct_mendely_artelus_models\model_artmen.h5', compile=False)
#C:\Users\mkhan\Desktop\musabi\OCT_project\data\oct\oct_mendely\test\1
input_folder = 'C:\\Users\\mkhan\\Desktop\\musabi\\OCT_project\\data\\oct\\oct_artelus\\test\\4\\'
#input_folder = 'C:\\Users\\mkhan\\Desktop\\musabi\\OCT_project\\oct_internal_images_for_testing\\validation\\AMD\\'


columns = ['filename', 'Normal', 'Drusen', 'DME', 'CNV', 'AMD', 'CSR', 'DR', 'MH']
df = pd.DataFrame(columns=columns)

# Load and predict probabilities for each image in input folder
for filename in os.listdir(input_folder):
    # Load input image
    img_path = os.path.join(input_folder, filename)
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(500, 500))
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
df.to_csv("AMDcheck.csv", index=False)








