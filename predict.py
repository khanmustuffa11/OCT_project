import tensorflow as tf
from tqdm import tqdm
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
# df = pd.read_csv('final_test.csv')
df = pd.read_csv('data/oct/OCT_100_csv.csv')
#df=df[df['source']=='idrid']
model = load_model('oct_models/oct_model.h5', compile=False)
def get_image(file_path):
    img = Image.open(file_path).convert('RGB')
    img = np.asarray(img.resize((500,500)))
    img = tf.cast(img, tf.float32)
    img = np.expand_dims(img, axis=0)
    return img
predictions = []
confidence = []
for i in tqdm(range(len(df))):
    img_path = df['path'].iloc[i]
    pred = model.predict(get_image(img_path))[0]
    predictions.append(np.argmax(np.asarray(pred)))
    confidence.append(max(pred))

df['predictions'] = predictions
df['confidences'] = confidence
df.to_csv('OCT_100_predictions.csv')
print(classification_report(df['label'],df['predictions']))
print(confusion_matrix(df['label'],df['predictions']))