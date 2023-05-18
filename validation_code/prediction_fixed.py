import tensorflow as tf
from tqdm import tqdm
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import glob
import os

model = load_model('gradcam_8class_011--0.528729--0.975070--0.601052--0.953836.h5', compile=False)
#model = load_model('oct_mendely_artelus_models\model_artmen.h5', compile=False)
#C:\Users\mkhan\Desktop\musabi\OCT_project\data\oct\oct_mendely\test\1
input_folder = 'C:\\Users\\mkhan\\Desktop\\musabi\\OCT_project\\data\\oct\\oct_mendely\\test\\1\\'
#input_folder = 'C:\\Users\\mkhan\\Desktop\\musabi\\OCT_project\\oct_internal_images_for_testing\\OCT_artelus_minkahi\\normal\\'


columns = ['filename', 'Normal', 'Drusen', 'DME', 'CNV', 'AMD', 'CSR', 'DR', 'MH']
df = pd.DataFrame(columns=columns)

def get_image(file_path):
    img = Image.open(file_path).convert('RGB')
    img = np.asarray(img.resize((500,500)))
    img = tf.cast(img, tf.float32)
    img = np.expand_dims(img, axis=0)
    return img

# Load and predict probabilities for each image in input folder
for filename in os.listdir(input_folder):
    # Load input image
    img_path = os.path.join(input_folder, filename)

    # Predict class probabilities
    probs = model.predict(get_image(img_path))

    # Append predicted class probabilities to DataFrame
    row = {'filename': filename}
    for i, prob in enumerate(probs[0]):
        row[f'class_{i}'] = prob
    #df = df.append(row, ignore_index=True)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

# Save DataFrame to CSV file
df.to_csv("newmodel_drusen_check.csv", index=False)
