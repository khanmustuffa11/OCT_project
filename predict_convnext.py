import tensorflow as tf
from tqdm import tqdm
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import glob
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
#df = pd.read_csv('final_test.csv')
df = pd.read_csv('data/oct/oct_artelus/OCT_final_testwcsv.csv')
#df=df[df['source']=='idrid']
#model = load_model('oct_mendely_artelus_models\model_artmen.h5', compile=False)
#my_reloaded_model = tf.keras.models.load_model('model_convnext031--0.382818--0.985649--0.420023--0.965179.h5',custom_objects={'LayerScale':hub.LayerScale})
model = load_model('model_convnext031--0.382818--0.985649--0.420023--0.965179.h5', compile=False)
print(model)
def get_image(file_path):
    img = Image.open(file_path).convert('RGB')
    img = np.asarray(img.resize((500,500)))
    img = tf.cast(img, tf.float32)
    img = np.expand_dims(img, axis=0)
    return img

## CSV Prediction ##
predictions = []
confidence = []
all_pred = []
for i in tqdm(range(len(df))):
    img_path = df['path'].iloc[i]
    pred = model.predict(get_image(img_path))[0]
    all_pred.append([img_path, np.argmax(np.asarray(pred)), max(pred)])
    # predictions.append(np.argmax(np.asarray(pred)))
    # confidence.append(max(pred))
df = pd.DataFrame(all_pred, columns=['image','predictions','confidence'])
df.to_csv('convnext_results_7july.csv')
### END ###

# import tensorflow as tf
# from tqdm import tqdm
# import pandas as pd
# from tensorflow.keras.models import load_model
# from PIL import Image
# import numpy as np
# from sklearn.metrics import classification_report, confusion_matrix
# import glob
# import os

# # Define the custom layer
# class LayerScale(tf.keras.layers.Layer):
#     pass
#     # Your implementation here...

# # Register the custom layer
# tf.keras.utils.get_custom_objects()['LayerScale'] = LayerScale

# # Load the model with the custom layer
# model = load_model('model_convnext031--0.382818--0.985649--0.420023--0.965179.h5', compile=False, custom_objects={'LayerScale': LayerScale})
# print(model)

# def get_image(file_path):
#     img = Image.open(file_path).convert('RGB')
#     img = np.asarray(img.resize((500,500)))
#     img = tf.cast(img, tf.float32)
#     img = np.expand_dims(img, axis=0)
#     return img

# # Load the CSV data
# df = pd.read_csv('data/oct/oct_artelus/OCT_final_testwcsv.csv')

# # Perform predictions
# predictions = []
# confidence = []
# all_pred = []
# for i in tqdm(range(len(df))):
#     img_path = df['path'].iloc[i]
#     pred = model.predict(get_image(img_path))[0]
#     all_pred.append([img_path, np.argmax(np.asarray(pred)), max(pred)])

# # Create DataFrame from predictions
# df_pred = pd.DataFrame(all_pred, columns=['image', 'predictions', 'confidence'])

# # Save predictions to CSV file
# df_pred.to_csv('convnext_results_7july.csv')





