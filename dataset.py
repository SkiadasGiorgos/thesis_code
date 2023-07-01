import glob
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transforms import create_transforms
import pickle
import torch
import numpy as np

class MyDataset(Dataset):
    def __init__(self, path, transform):
        self.files = path
        self.patch_files = 'landmarks_new.pkl'
        self.transform = create_transforms(transform)
        self.labels = [filepath.split('/')[-2] for filepath in self.files]
        self.encoder = LabelEncoder()
        self.encoder.fit(self.labels)
        self.labels = self.encoder.transform(self.labels)
        with (open(self.patch_files, "rb")) as openfile:
            while True:
                try:
                    self.patch_centers = pickle.load(openfile)
                except EOFError:
                    break


    def __getitem__(self, item):
        image = self.files[item]
        image = Image.open(image)
        image = self.transform(image)
        label = self.labels[item]
        if self.files[item] in self.patch_centers:
            patch_center = self.patch_centers.get(self.files[item])
            patch_center = torch.from_numpy(np.asarray(patch_center))
        else:
            patch_center = [0,0,0,0,0]
            patch_center = torch.from_numpy(np.asarray(patch_center))
            
        return image, label, patch_center
    def __len__(self):
        return len(self.files)