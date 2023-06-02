import os
from PIL import Image
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('vit_torch_0.0624_0.9748_0.0583_0.9757.pth')
import csv

import csv

def predict_folder(model, folder_path, device):
    model.eval()
    predictions = []

    # Get a list of image file names in the folder
    file_names = os.listdir(folder_path)

    with torch.no_grad():
        for file_name in file_names:
            # Load and preprocess the image
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path).convert('RGB')
            image = image.resize((224, 224))
            image = transforms.ToTensor()(image)
            image = image.unsqueeze(0).to(device)

            # Perform forward pass and get predictions
            outputs = model(image)
            probabilities = torch.sigmoid(outputs)
            probabilities = probabilities.squeeze().cpu().numpy()

            # Append the prediction to the list
            prediction = {'File': file_name}
            for i, disease_label in enumerate(DISEASE_LABELS):
                prediction[disease_label] = probabilities[i]
            predictions.append(prediction)

    # Define the path for the output CSV file
    output_csv = 'predict.csv'

    # Save the predictions to the CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['File'] + DISEASE_LABELS
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    return output_csv




# Example usage
DISEASE_LABELS = ['level_0', 'level_1','level_2','level_3','level_4','level_5','level_6','level_7']
folder_path = 'C:\\Users\\mkhan\\Desktop\\musabi\\OCT_project\\data\\oct\\oct_mendely\\test\\1\\'
folder_predictions = predict_folder(model, folder_path, device)
