import tensorflow as tf
from tqdm import tqdm
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import glob
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# df = pd.read_csv('final_test.csv')
#df = pd.read_csv('data/oct/oct_mendely/test_csv.csv')
#df=df[df['source']=='idrid']
model = load_model('oct_models/oct_model.h5', compile=False)
print(model)
def get_image(file_path):
    img = Image.open(file_path).convert('RGB')
    img = np.asarray(img.resize((500,500)))
    img = tf.cast(img, tf.float32)
    img = np.expand_dims(img, axis=0)
    return img
predictions = []
confidence = []
# for i in tqdm(range(len(df))):
#     img_path = df['path'].iloc[i]
#     pred = model.predict(get_image(img_path))[0]
#     predictions.append(np.argmax(np.asarray(pred)))
#     confidence.append(max(pred))
all_pred = []
for i in tqdm(glob.glob('C:\\Users\\mkhan\\Desktop\\musabi\\OCT_project\\data\oct\\oct_artelus\\test\\4\\*.tiff')):
    img_path = i
    pred = model.predict(get_image(img_path))[0]
    all_pred.append([img_path, np.argmax(np.asarray(pred)), max(pred)])
    #confidence.append(max(pred))

df = pd.DataFrame(all_pred, columns=['image','predictions','confidence'])
df.to_csv('folderpredict2.csv')
# print(classification_report(df['level'],df['predictions']))
# print(confusion_matrix(df['level'],df['predictions']))