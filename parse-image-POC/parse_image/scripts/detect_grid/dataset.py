import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from parse_image.scripts.detect_grid.config import CLASS_NAMES, CLASS_MAPS


IMAGE_EXTS = ['.png']

def is_image(filename):
    return any([filename.endswith(ext) for ext in IMAGE_EXTS])

class GridLabelDataset(Dataset):
    def __init__(self, image_dir, label_dir):
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
            content = f.read().strip().split('\n')
            lines = [line.strip() for line in content if line.strip()]
            labels = torch.tensor([
                [CLASS_MAPS.name_to_label[name] for name in line.split(' ')]
                for line in lines 
            ])

        return self.transform(img), labels
