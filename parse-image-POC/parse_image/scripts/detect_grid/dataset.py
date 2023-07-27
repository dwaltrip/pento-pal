import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor

IMAGE_EXTS = ['.png']

def is_image(filename):
    return any([file.endswith(ext) for ext in exts])

class GridLabelDataset(Dataset):
    def __init__(self, image_dir, label_dir, exts=IMAGE_EXTS):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_filenames = [
            file for file in os.listdir(image_dir) if is_image(file)
        ]
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
