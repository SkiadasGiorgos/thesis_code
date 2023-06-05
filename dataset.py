import glob
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transforms import create_transforms


class MyDataset(Dataset):
    def __init__(self, path, transform):
        self.files = path
        print(type(self.files))
        self.transform = create_transforms(transform)
        self.labels = [filepath.split('/')[-2] for filepath in self.files]
        self.encoder = LabelEncoder()
        self.encoder.fit(self.labels)
        self.labels = self.encoder.transform(self.labels)
    def __getitem__(self, item):
        image = self.files[item]
        image = Image.open(image)
        image = self.transform(image)
        label = self.labels[item]
        return image, label
    def __len__(self):
        return len(self.files)