import tensorflow as tf
from tqdm import tqdm
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import glob
import os
import random
import numpy as np
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D, Flatten, Dropout
#import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard
import tensorflow as tf
from PIL import Image, ImageFile
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy
import pandas as pd
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tensorflow.keras.metrics import top_k_categorical_accuracy
from sklearn.utils import shuffle
#from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0
import os
import sklearn.metrics as sklm
from sklearn.utils.class_weight import compute_sample_weight
#from tensorflow_addons.metrics import F1Score
from convnext import ConvNeXtTiny

# Define the custom layer
class LayerScale(tf.keras.layers.Layer):
    pass
    # Your implementation here...

# Register the custom layer
tf.keras.utils.get_custom_objects()['LayerScale'] = LayerScale

# Load the model with the custom layer
base_model = ConvNeXtTiny(weights='imagenet',include_top=False, pooling = 'avg')
model = Sequential()
model.add(base_model)
model.add(Dropout(.000000000005))
model.add(Dense(8, activation = 'softmax'))
model.load_weights('model_convnext_finetune021--0.492148--0.991566--0.589796--0.962033.h5')

#model = load_model('model_convnext031--0.382818--0.985649--0.420023--0.965179.h5', compile=False, custom_objects={'LayerScale': LayerScale})
#print(model)

def get_image(file_path):
    img = Image.open(file_path).convert('RGB')
    img = np.asarray(img.resize((224,224)))
    img = tf.cast(img, tf.float32)
    img = np.expand_dims(img, axis=0)
    return img

# Load the CSV data
df = pd.read_csv('data/oct/oct_artelus/OCT_final_testwcsv.csv')

# Perform predictions
predictions = []
confidence = []
all_pred = []
for i in tqdm(range(len(df))):
    img_path = df['path'].iloc[i]
    pred = model.predict(get_image(img_path))[0]
    all_pred.append([img_path, np.argmax(np.asarray(pred)), max(pred)])

# Create DataFrame from predictions
df_pred = pd.DataFrame(all_pred, columns=['image', 'predictions', 'confidence'])

# Save predictions to CSV file
df_pred.to_csv('convnext_results_7july.csv')