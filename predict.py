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
model = load_model('clahe_models/clahe_gradcam_8class_022--0.484554--0.993699--0.606048--0.946598.h5', compile=False)
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
df.to_csv('clahe_results_22_may.csv')
### END ###


# # Folder Predict###
# all_pred = []
# for i in tqdm(glob.glob('C:\\Users\\mkhan\\Desktop\\musabi\\OCT_project\\data\\oct\\oct_artelus\\oct_art_0\\0\\*.tiff')):
#     img_path = i
#     pred = model.predict(get_image(img_path))[0]
#     print(img_path, pred)
#     all_pred.append([img_path, np.argmax(np.asarray(pred)), max(pred)])

# df = pd.DataFrame(all_pred, columns=['image','predictions','confidence'])
# df.to_csv('artelus0check.csv')
# ### End ####  


# print(classification_report(df['level'],df['predictions']))
# print(confusion_matrix(df['level'],df['predictions']))


class CustomDatasetLabeled(torch.utils.data.Dataset):
    def __init__(self, df, split, images_folder, transform = None):
        self.df = df
        self.images_folder = images_folder
        self.transform = transform
        self.split=split
        #self.class2index = {"cat":0, "dog":1}

    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        filename = self.df.loc[index]["path"]
        label = self.df.loc[index]["level"]
        image = PIL.Image.open(os.path.join(self.images_folder, filename))
        image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.split=="unlabeled":
            return image, -1
        return image, label

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_data = CustomDatasetLabeled(df=train, split="train",  images_folder="./", transform=data_transforms['train'])
val_data = CustomDatasetLabeled(df=val, split="val", images_folder="./", transform=data_transforms['val'])