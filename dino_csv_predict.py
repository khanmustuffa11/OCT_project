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

dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
class DinoVisionTransformerClassifier(torch.nn.Module):
    def __init__(self):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = dinov2_vits14
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 8)  # Update the number of output classes
        )

    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x    

LEARNING_RATE = 0.0001
model = DinoVisionTransformerClassifier()


model = torch.load('C:\\Users\\mkhan\\Desktop\\musabi\\OCT_project\\din_code\\dino_models\\oct_dino_b14_torch0.0181_0.9932_0.1049_0.9761.pth')
#print(model.summary())
if isinstance(model, torch.nn.DataParallel):
           model = model.module
import csv
DISEASE_LABELS = ['level_0', 'level_1','level_2','level_3','level_4','level_5','level_6','level_7']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict_csv(model, csv_path, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        with open(csv_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                image_path = row['path']
                ground_truth = row['level']
                ####### Load and preprocess the image
                # image = Image.open(image_path).convert('RGB')
                # image = image.resize((224, 224))
                # #image = preprocess(image)
                # image = transforms.ToTensor()(image)
                # image = image.unsqueeze(0).to(device)

                img = Image.open(image_path).resize((224, 224)).convert('RGB')
                img = np.asarray(img)
                img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
                img = img.to(device)

                # Perform forward pass and get predictions
                outputs = model(img)
                probabilities = torch.sigmoid(outputs)
                print(probabilities)
                #probabilities = torch.argmax(probabilities, dim=1)

                probabilities = probabilities.squeeze().cpu().numpy()

                # Append the prediction to the list
                prediction = {'image': image_path, 'level': ground_truth}
                for i, disease_label in enumerate(DISEASE_LABELS):
                    prediction[disease_label] = probabilities[i]
                predictions.append(prediction)

        # Define the path for the output CSV file
        output_csv = 'dinoxrayre_predictions.csv'

        # Save the predictions to the CSV file
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = ['image', 'level'] + DISEASE_LABELS
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(predictions)

        return output_csv


# Specify the CSV file path
csv_file = 'C:\\Users\\mkhan\\Desktop\\musabi\\OCT_project\\oct_final_test_vit.csv'

# Perform predictions and save the results
output_csv = predict_csv(model, csv_file, device)

print(f"Predictions saved to: {output_csv}")
