import torchvision, torch
from torchvision import transforms
import os
import math
import random
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
device = 'cuda'
img_width, img_height= 224, 224
BATCH_SIZE = 128
DISEASE_LABELS = ['level_0', 'level_1','level_2','level_3']

#DR_LABELS = []
num_classes = len(DISEASE_LABELS)


df_train, df_test = pd.read_csv('data/covid_pneu/vit_training_csv.csv'), pd.read_csv('data/covid_pneu/vit_testing_csv.csv')


def balance_df(dataframe, columns= DISEASE_LABELS):
    label_dict = {}
    max_sum = []
    dataframe_out = dataframe
    for col in columns:
        label_dict[col] = dataframe[col].sum()
        max_sum.append(dataframe[col].sum())
    label_dict = dict(sorted(label_dict.items(), key=lambda item: item[1]))
    for pathology,val in label_dict.items():
        fraction = max((max(max_sum)/val - 3),0)
        sampled_df = dataframe[dataframe[pathology] > 0.3]
        upsampled_df = sampled_df.sample(frac=fraction, replace=True)
        dataframe_out = pd.concat([dataframe_out, upsampled_df])
    label_dict = {}
    for col in columns:
        label_dict[col] = dataframe_out[col].sum()
    print('upsampling')
    print(label_dict)
    return dataframe_out

df_train = balance_df(df_train)
df_test = balance_df(df_test)

def compute_weights(df,class_list=None):
    weights = []
    for each_class in class_list:
        each_weight = list(compute_sample_weight(class_weight='balanced', y = df[each_class]))
        weights.append(each_weight)
    return np.max(np.array(weights),axis = 0)

for i in range(100):
    df_train = shuffle(df_train)

def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
class Engine:
    @staticmethod
    def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device, save_dir="model_checkpoints"):
        model.train()
        results = {'train_losses': [], 'test_losses': [], 'train_accuracies': [], 'test_accuracies': []}

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for epoch in range(epochs):
            train_loss, train_correct, train_total = 0, 0, 0
            for batch_idx, (inputs, labels) in enumerate(train_dataloader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                train_pred = torch.nn.functional.softmax(outputs,dim=1) > 0.5
                train_correct += (train_pred == labels.byte()).sum().item()
                train_total += labels.numel()
                if batch_idx%100==0:
                    print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

            train_loss /= len(train_dataloader)
            train_accuracy = train_correct / train_total
            results['train_losses'].append(train_loss)
            results['train_accuracies'].append(train_accuracy)

            # Evaluate the model
            model.eval()
            test_loss, test_correct, test_total = 0, 0, 0
            with torch.no_grad():
                for inputs, labels in test_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)

                    test_loss += loss.item()

                    test_pred = torch.nn.functional.softmax(outputs, dim=1) > 0.5
                    test_correct += (test_pred == labels.byte()).sum().item()
                    test_total += labels.numel()

            test_loss /= len(test_dataloader)
            test_accuracy = test_correct / test_total
            results['test_losses'].append(test_loss)
            results['test_accuracies'].append(test_accuracy)

            # Save the model
            model_path = f"{save_dir}/vit_torch_xray{train_loss:.4f}_{train_accuracy:.4f}_{test_loss:.4f}_{test_accuracy:.4f}.pth"
            torch.save(model, model_path)

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

        return results

class CustomDataset(Dataset):
    def __init__(self, data_frame, img_shape=None, augmentation=True, subset='training'):
        self.data_frame = data_frame
        self.img_shape = img_shape
        self.augmentation = augmentation
        self.subset = subset

        print(f"Found {self.data_frame.shape[0]} images")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx]['path']
        img = self.__get_image(img_path)
        
        labels = self.data_frame.iloc[idx][DISEASE_LABELS].values
        labels = np.array(labels, dtype=np.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        return img, labels

    def __data_augmentation(self, img):
        img = Image.fromarray(img.astype('uint8'))
        transform_list = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.7, saturation=0.7, hue=0.2),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
        transforms.RandomResizedCrop((img_width, img_height), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomRotation(degrees=15)
    ]
        transform = transforms.Compose(transform_list)
        img = transform(img)
        img = np.asarray(img)
        return img

    def __get_image(self, file_id):
        #file_id = 'pathology_resized/' + str(file_id)
        if self.subset != 'training':
            img = Image.open(file_id).resize(self.img_shape).convert('RGB')
            #img = transforms.ToTensor()(np.array(img))
            img = np.asarray(img)
            img = torch.from_numpy(img).permute(2, 0, 1).float()
        else:
            if random.randint(1, 5) == 5:
                img = Image.open(file_id).convert('L').convert('RGB').resize(self.img_shape)
                #img = transforms.ToTensor()(np.array(img))
                img = np.asarray(img)
                if self.augmentation:
                    img = self.__data_augmentation(img)
                img = torch.from_numpy(img).permute(2, 0, 1).float()
            else:
                img = Image.open(file_id).resize(self.img_shape).convert('RGB')
                #img = transforms.ToTensor()(np.array(img))
                img = np.asarray(img)
                if self.augmentation:
                    img = self.__data_augmentation(img)
                img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img

def create_dataloaders(train_df, test_df, img_shape, batch_size, num_workers=0):
    train_dataset = CustomDataset(train_df, img_shape=img_shape, subset='training')
    test_dataset = CustomDataset(test_df, img_shape=img_shape, subset='testing')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, test_dataloader

train_dataloader, test_dataloader = create_dataloaders(df_train, df_test, img_shape=(img_width, img_height), batch_size=BATCH_SIZE)

pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT

pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights)
pretrained_vit.heads = torch.nn.Linear(in_features=768, out_features=num_classes)

# Check if multiple GPUs are available and wrap the model with DataParallel
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    pretrained_vit = torch.nn.DataParallel(pretrained_vit)

pretrained_vit = pretrained_vit.to('cuda')
pretrained_vit_transforms = pretrained_vit_weights.transforms()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(pretrained_vit.parameters(), lr=0.001)

num_epochs = 100
engine = Engine()
set_seeds()
# Training loop
pretrained_vit_results = engine.train(model=pretrained_vit,
                                      train_dataloader=train_dataloader,
                                      test_dataloader=test_dataloader,
                                      optimizer=optimizer,
                                      loss_fn=criterion,
                                      epochs=100,
                                      device=device)