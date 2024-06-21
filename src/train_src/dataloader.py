import os
import torch
from torch.utils.data import dataset
from PIL import Image

# write a class dataloader that inherits from dataset, and processes data/CUB_200_2011
class dataloader(dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.data = []
        self.labels = []
        self.classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.load_data()

    def load_data(self):
        # get all classes
        self.classes = [d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]
        self.classes.sort()
        for i, c in enumerate(self.classes):
            self.class_to_idx[c] = i
            self.idx_to_class[i] = c
            files = os.listdir(os.path.join(self.data_path, c))
            for f in files:
                if f.endswith('.jpg'):
                    self.data.append(os.path.join(self.data_path, c, f))
                    self.labels.append(i)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label