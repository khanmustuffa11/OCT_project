import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image
import numpy as np
import cv2
from cv2 import resize, hconcat,cvtColor,COLOR_GRAY2BGR,vconcat
import os, glob
import matplotlib.cm as cm
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import bisect
DISEASE_LABELS = ['level_0', 'level_1','level_2','level_3','level_4','level_5','level_6','level_7']

DR_LABELS = []

model_path = "vit_torch_0.0624_0.9748_0.0583_0.9757.pth"
model = torch.load(model_path)
def torch_prediction(model_path, image_path, threshold=0.25):
    # Load the trained model
    model = torch.load(model_path)
    model.eval()

    # Image preprocessing
    img = Image.open(image_path).resize((224, 224))
    img = np.asarray(img)
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    img = img.to('cuda')

    # Predict
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.sigmoid(outputs).cpu().numpy()

    # Create prediction dictionary
    pred_dict = {label: float(prob) for label, prob in zip(DR_LABELS + DISEASE_LABELS, probabilities[0]) if prob > threshold}

    return pred_dict
    
def get_image(file_path):
    img_path = file_path
    img = Image.open(img_path).convert('RGB').resize((224,224))
    img = np.asarray(img)
    img = tf.cast(img, tf.float32)
    img = np.expand_dims(img, axis=0)
    return img


def get_prediction_tags(all_pred, threshold = .25):
    prediction_tags = {}
    # DR predictions
    prediction_tags = {label: float(prob) for label, prob in zip(DR_LABELS + DISEASE_LABELS, all_pred[0]) if prob > threshold}
    return prediction_tags

def predict(img_path, return_dr_level = True):
    model.eval()
    prediction_tags = {}
    # Image preprocessing
    img = Image.open(img_path).resize((224, 224)).convert('RGB')
    img = np.asarray(img)
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    img = img.to('cuda')

    # Predict
    with torch.no_grad():
        outputs = model(img)
        all_pred = torch.sigmoid(outputs).cpu().numpy()
    prediction_tags  = get_prediction_tags(all_pred)
    final_pred_tags = prediction_tags.copy()
    # if return_dr_level:
    #     mtm_vtdr,grading_pred,disease_pred= get_dr_level(final_pred_tags)
    return prediction_tags

#def df_predict(df_path, prefix = 'C:\\Users\\mkhan\\Desktop\\model_testing\\zeiss\\zeiss_1500\\', suffix = '.jpg', save = True, gradcam = False, sample = False):
def df_predict(df_path, prefix = 'pathology_dubai/', suffix = '.tiff', save = True, sample = False):
    data = np.genfromtxt(df_path, delimiter=',', skip_header=1, dtype=np.str)
    image_paths = data[:, 0]
    all_prediction_tags = []
    for image_path in tqdm(image_paths):
        image_path = prefix + image_path + suffix
        prediction_tags = predict(image_path)
        all_prediction_tags.append(prediction_tags)
    df = pd.read_csv(df_path)
    df['pred_tags'] = all_prediction_tags
    if save:
        df.to_csv('topcon_50_21april_newmodel.csv')
    else:
        return df

def folder_predict(folder_path):
    img_list = glob.glob(folder_path+'*.jpeg') + glob.glob(folder_path+'*.tiff') + glob.glob(folder_path+'*.png') + glob.glob(folder_path+'*.jfif')
    pred_list = []
    for img in tqdm(img_list):
        predictions = predict(img)
        print(predictions)
        pred_list.append([img, predictions])
    df = pd.DataFrame(pred_list, columns = ['Image', 'pred_tags'])
    df.to_csv('predictions_from_3.csv')

folder_predict('C:\\Users\\mkhan\\Desktop\\musabi\\OCT_project\\data\\oct\\oct_mendely\\test\\3\\')


## for single image
# all_pred=[]
# for image_path in tqdm(glob.glob('pathology_dubai/*.tiff')):
#     prediction = torch_predict(model_path, image_path)
#     all_pred.append([image_path,prediction])
# import pandas as pd
# df = pd.DataFrame(all_pred)
# df.to_csv('vit.csv')