import random
import numpy as np
from tqdm import tqdm
import cv2
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,GlobalMaxPooling2D, Flatten, Dropout, Input
import tensorflow_addons as tfa
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
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0
import os
import sklearn.metrics as sklm
from sklearn.utils.class_weight import compute_sample_weight
from tensorflow_addons.metrics import F1Score
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")
df_train = pd.read_csv('OCT_final_train_csv.csv')
#df_train = df_train[df_train['source']=='kaggle']
#train_kaggle = df_train[df_train['source']=='kaggle'] 
#train_idrid_all = df_train[(df_train['source']=='idrid') & (df_train['drlevel'] > 0)]  
#train_idrid_0 = df_train[(df_train['source']=='idrid') & (df_train['drlevel'] == 0)].sample(len(train_idrid_all),replace=True)
#train_idrid = pd.concat([train_idrid_all,train_idrid_0])

def balance_df(dataframe):
    label_dict = dict(dataframe['level'].value_counts())
    max_sum = list(dataframe['level'].value_counts())[0]
    for level,val in label_dict.items():
        df = dataframe[dataframe['level']==level]
        sampled_df = df.sample(frac=max((max_sum/val - 1),0), replace=True)
        dataframe = pd.concat([dataframe, sampled_df])
    return dataframe

# balanced_kaggle = balance_df(train_kaggle)
#balanced_idrid = balance_df(train_idrid)
#balanced_idrid = balanced_idrid.sample(frac=len(train_kaggle)/(1.5*len(balanced_idrid)),replace=True)
#df_train = balance_df(pd.concat([train_kaggle, balanced_idrid]))
for i in range(100):
    df_train = shuffle(df_train)
df_train = balance_df(df_train)

df_test = pd.read_csv('OCT_final_val_csv.csv')
#df_test = df_test[df_test['source']=='kaggle']
df_test = balance_df(df_test)

print(df_train['level'].value_counts())
print(df_test['level'].value_counts())
df_train= pd.get_dummies(df_train, columns=["level"])
df_test= pd.get_dummies(df_test, columns=["level"])

batch_size = 16
img_width, img_height= 500, 500

import math

class CustomDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_frame, batch_size=10, img_shape=None, augmentation=True, subset = 'training', num_classes=None):
        self.data_frame = data_frame
        self.train_len = len(data_frame)
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.augmentation = augmentation
        self.subset = subset
        print(f"Found {self.data_frame.shape[0]} images belonging to {self.num_classes} classes")

    def __len__(self):
        ''' return total number of batches '''
        self.data_frame = shuffle(self.data_frame)
        return math.ceil(self.train_len / self.batch_size)

    def on_epoch_end(self):
        ''' shuffle data after every epoch '''
        # fix on epoch end it's not working, adding shuffle in len for alternative
        pass

    def __data_augmentation(self, img, mode = 'rgb'):
        ''' function for apply some data augmentation '''
        flip_prob = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if flip_prob > 0.5:
            img = tf.image.transpose(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.5, 1.5)
        img = tf.image.random_saturation(img, 0.5, 1.5)
        if mode == 'rgb':
            img = tf.image.random_hue(img, .2)
        img = tfa.image.rotate(img, random.uniform(-90,90) * math.pi / 180)
        
        return tf.image.random_jpeg_quality(img, 30,100)
    def _apply_clahe(self, image, limit = 0.5):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Convert the image to LAB color space
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Split the LAB image into L, A, and B channels
        l, a, b = cv2.split(lab_image)

        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)

        # Merge the modified L channel with the original A and B channels
        lab_clahe_image = cv2.merge((l_clahe, a, b))

        # Convert the image back to BGR format
        modified_image = cv2.cvtColor(lab_clahe_image, cv2.COLOR_LAB2BGR)

        return cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)

    def __get_image(self, file_id):
        """ open image with file_id path and apply data augmentation """
        if self.subset!= 'training':
            img = Image.open(file_id).convert('RGB')
            img = np.asarray(img.resize((self.img_shape[0], self.img_shape[1])))
            img =self._apply_clahe(img)
        else:
            if random.randint(1,5) >= 4:
                img = Image.open(file_id).convert('L').convert('RGB')
                img = np.asarray(img.resize((self.img_shape[0], self.img_shape[1])))
                if self.augmentation:
                    img = self.__data_augmentation(img, mode = 'gray')
            else:
                img = Image.open(file_id).convert('RGB')
                img = np.asarray(img.resize((self.img_shape[0], self.img_shape[1])))
                img = self._apply_clahe(img)
                if self.augmentation: 
                    img = self.__data_augmentation(img)
        img = tf.cast(img, tf.float32)
        return img

    def __getitem__(self, idx):
        batch_x = self.data_frame["path"][idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_vector= [self.data_frame["level_0"][idx * self.batch_size:(idx + 1) * self.batch_size].values,
                        self.data_frame["level_1"][idx * self.batch_size:(idx + 1) * self.batch_size].values,
                        self.data_frame["level_2"][idx * self.batch_size:(idx + 1) * self.batch_size].values,
                        self.data_frame["level_3"][idx * self.batch_size:(idx + 1) * self.batch_size].values,
                        self.data_frame["level_4"][idx * self.batch_size:(idx + 1) * self.batch_size].values,
                        self.data_frame["level_5"][idx * self.batch_size:(idx + 1) * self.batch_size].values,
                        self.data_frame["level_6"][idx * self.batch_size:(idx + 1) * self.batch_size].values,
                        self.data_frame["level_7"][idx * self.batch_size:(idx + 1) * self.batch_size].values]
                        #self.data_frame["level_4"][idx * self.batch_size:(idx + 1) * self.batch_size].values]
        # batch_weights = [self.data_frame["sample_weight"][idx * self.batch_size:(idx + 1) * self.batch_size].values]
        x = [self.__get_image(file_id) for file_id in batch_x]
        # x = preprocess_input(np.array(x))
        batch_y_vector = np.array(batch_y_vector).T.tolist()
        y_vector = [label_id for label_id in batch_y_vector]
        y_vector = tf.cast(y_vector, tf.int32)

        # batch_weights = np.array(batch_weights).T.tolist()
        # weight_vector = [label_id for label_id in batch_weights]
        # weight_vector = tf.cast(weight_vector, tf.float32)
        return tf.convert_to_tensor(x), tf.convert_to_tensor(y_vector)
        # , tf.convert_to_tensor(weight_vector)

custom_train_generator = CustomDataGenerator(data_frame = df_train, batch_size = batch_size, img_shape = (img_height, img_width, 3))
custom_test_generator = CustomDataGenerator(data_frame = df_test, batch_size = batch_size, img_shape = (img_height, img_width, 3), subset = 'test', augmentation = False)
strategy = tf.distribute.MirroredStrategy()
adm= Adam(learning_rate = 1e-5)
loss = CategoricalCrossentropy(label_smoothing=.1)

with strategy.scope():
    
    input_tensor = Input(shape=(img_width,img_height,3))
    conv_base = EfficientNetV2B0(weights='imagenet',include_top=False,input_tensor=input_tensor, pooling='avg')
    # a_map = layers.Conv2D(512, 1, strides=(1, 1), padding="same", activation='relu')(conv_base.output)
    # a_map = layers.Conv2D(1, 1, strides=(1, 1), padding="same", activation='relu')(a_map)
    # a_map = layers.Conv2D(1280, 1, strides=(1, 1), padding="same", activation='sigmoid')(a_map)
    # res = layers.Multiply()([conv_base.output, a_map])
    #x = GlobalMaxPooling2D(conv_base.output)
    x = Dropout(0.5)(conv_base.output)
    predictions = Dense(8, activation='softmax', name='final_output')(x)
    model = Model(input_tensor, predictions)
    del conv_base
    model.load_weights("base_models/basemodel.h5", by_name=True, skip_mismatch = True)
    model.summary()
    model.compile(optimizer=adm, loss=loss,metrics=['accuracy'])

mcp_save = ModelCheckpoint('clahe_gradcam_8class_{epoch:03d}--{loss:03f}--{accuracy:03f}--{val_loss:03f}--{val_accuracy:03f}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='min')
model.fit(
    custom_train_generator,
    validation_data = custom_test_generator,
    epochs = 100,
    verbose = 1,
    workers = 16,
    max_queue_size = 1,
    callbacks=[mcp_save])