import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import random
import pandas as pd
import PIL
from tqdm import tqdm

def set_seed(no):
    torch.manual_seed(no)
    random.seed(no)
    np.random.seed(no)
    os.environ['PYTHONHASHSEED'] = str()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(100)

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

train = pd.read_csv("data/covid_pneu/training_csv.csv")
val = pd.read_csv("data/covid_pneu/testing_csv.csv")

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
    
train_data = CustomDatasetLabeled(df=train, split="train",  images_folder="./", transform=data_transforms['train'])
val_data = CustomDatasetLabeled(df=val, split="val", images_folder="./", transform=data_transforms['val'])
dataset = torch.utils.data.ConcatDataset([train_data, val_data])

train_loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True, 
                                           num_workers=12)

class_names = [0, 1, 2, 3, 4]

dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = dinov2_vits14
        self.classifier = nn.Sequential(
            nn.Linear(768, 32),
            nn.ReLU(),
            nn.Linear(32, 5)
        )
    
    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.0001
model = DinoVisionTransformerClassifier()
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

model = model.to(device)

def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

for epoch in range(20):  # loop over the dataset multiple times

    running_loss = 0.0
    train_acc = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        train_acc += accuracy_fn(y_true=labels.to(device),
                                 y_pred=outputs.argmax(dim=1)) 
        if i % 50 == 49:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
            running_loss = 0.0
    train_acc /= len(train_loader)
    print(f"Train accuracy: {train_acc:.2f}%")

print('Finished Training')

from pathlib import Path

# Create models directory (if it doesn't already exist), see: https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, # create parent directories if needed
                 exist_ok=True # if models directory already exists, don't error
)

# Create model save path
MODEL_NAME = "x_ray_pretrain_dino_10_epochs.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(), # only saving the state_dict() only saves the learned parameters
           f=MODEL_SAVE_PATH)