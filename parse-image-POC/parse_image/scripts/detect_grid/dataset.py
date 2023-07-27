import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
from config import IMAGE_DIR, LABEL_DIR
import os

class GridLabelDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_filenames = os.listdir(image_dir)
        self.transform = ToTensor()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        img = Image.open(os.path.join(self.image_dir, image_filename))
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        with open(os.path.join(self.label_dir, label_filename), 'r') as f:
            labels = torch.tensor([
                [int(cell) for cell in line.split()]
                for line in f.read().split('\n')
                if line
            ])

        return self.transform(img), labels
