import random
import numpy as np
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D, Flatten, Dropout
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
df_train = pd.read_csv('data/covid_pneu/training_csv.csv')
train_kaggle = df_train[df_train['source']=='covid_pneu'] 
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
df_train = balance_df(df_train)
for i in range(100):
    df_train = shuffle(df_train)

df_test = pd.read_csv('data/covid_pneu/testing_csv.csv')
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
        img = tf.image.random_brightness(img, 0.5)
        img = tf.image.random_contrast(img, 0.5, 1.5)
        img = tf.image.random_saturation(img, 0.5, 1.5)
        if mode == 'rgb':
            img = tf.image.random_hue(img, 0.5)
        img = tfa.image.rotate(img, random.uniform(-90,90) * math.pi / 180)
        
        return tf.image.random_jpeg_quality(img, 30,100)

    def __get_image(self, file_id):
        """ open image with file_id path and apply data augmentation """
        if self.subset!= 'training':
            img = Image.open(file_id).convert('RGB')
            img = np.asarray(img.resize((self.img_shape[0], self.img_shape[1])))
        else:
            if random.randint(0,5) >= 4:
                img = Image.open(file_id).convert('L').convert('RGB')
                img = np.asarray(img.resize((self.img_shape[0], self.img_shape[1])))
                if self.augmentation:
                    img = self.__data_augmentation(img, mode = 'gray')
            else:
                img = Image.open(file_id).convert('RGB')
                img = np.asarray(img.resize((self.img_shape[0], self.img_shape[1])))
                if self.augmentation: 
                    img = self.__data_augmentation(img)
        img = tf.cast(img, tf.float32)
        return img

    def __getitem__(self, idx):
        batch_x = self.data_frame["path"][idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_vector= [self.data_frame["level_0"][idx * self.batch_size:(idx + 1) * self.batch_size].values,
                        self.data_frame["level_1"][idx * self.batch_size:(idx + 1) * self.batch_size].values,
                        self.data_frame["level_2"][idx * self.batch_size:(idx + 1) * self.batch_size].values,
                        self.data_frame["level_3"][idx * self.batch_size:(idx + 1) * self.batch_size].values,]
                        #self.data_frame["drlevel_4"][idx * self.batch_size:(idx + 1) * self.batch_size].values]
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
# print(custom_train_generator.__getitem__(0))
strategy = tf.distribute.MirroredStrategy()
adm= Adam(learning_rate = 1e-4)
loss = CategoricalCrossentropy(label_smoothing=.1)

with strategy.scope():
    base_model = EfficientNetV2B0(weights='imagenet',include_top=False, pooling = 'avg')
    model = Sequential()
    model.add(base_model)
    model.add(Dropout(.75))
    model.add(Dense(4, activation = 'softmax'))
    del base_model
    # model = Sequential()
    # base_model = load_model('stable_effnet.h5', compile = False)
    # for layer in base_model.layers[:-2]:
    #     model.add(layer)
    # model.add(Dropout(.75))
    # model.add(Dense(5, activation = 'softmax'))
    model.summary()
    model.compile(optimizer=adm, loss=loss,metrics=['accuracy'])

mcp_save = ModelCheckpoint('model_{epoch:03d}--{loss:03f}--{accuracy:03f}--{val_loss:03f}--{val_accuracy:03f}.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='min')
model.fit(
    custom_train_generator,
    validation_data = custom_test_generator,
    epochs = 1000,
    verbose = 1,
    workers = 16,
    max_queue_size = 1,
    callbacks=[mcp_save])