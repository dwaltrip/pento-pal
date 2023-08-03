from itertools import zip_longest
import os

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from parse_image.scripts.detect_grid.config import (
    CLASS_NAMES,
    CLASS_MAPS,
    IMAGE_SIDE_LEN,
)
from parse_image.scripts.detect_grid.prep_images import resize_and_pad
from parse_image.scripts.detect_grid.annotation_tool_v2 import (
    normalize_label_file_content,
    validate_labels,
)
from parse_image.scripts.detect_grid.utils import is_image
from parse_image.scripts.detect_grid.augment import get_augmentations


IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

class GridLabelDataset(Dataset):

    def __init__(self, image_dir, label_dir, augment=False):
        self.augment = augment

        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_filenames = [
            file for file in os.listdir(image_dir) if is_image(file)
        ]
        self.label_filenames = [
            file for file in os.listdir(label_dir) if file.endswith('.txt')
        ]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
        ])

        files = zip_longest(self.image_filenames, self.label_filenames)
        self.prepped_data = self._prep_data(files)

    
    def __getitem__(self, idx):
        return self.prepped_images[idx], self.prepped_labels[idx]

    def __len__(self):
        return min(len(self.prepped_images), len(self.prepped_labels))
    
    def _prep_data(self, files):
        def prep_from_files(image_filename, label_filename):
            assert image_filename or label_filename
            if not image_filename:
                image_filename = os.path.splitext(label_filename)[0] + '.png'
            elif not label_filename:
                label_filename = os.path.splitext(image_filename)[0] + '.txt'
            return (
                self._prep_image(os.path.join(self.image_dir, image_filename)),
                self._prep_label(os.path.join(self.label_dir, label_filename)),
            )

        prepped_data = [prep_from_files(*item) for item in files]

        if self.augment:
            return [
                augmented_item 
                for item in prepped_data
                for augmented_item in get_augmentations(*item)
            ]
        else:
            return prepped_data

    def _prep_image(self, image_path):
        raw_img = Image.open(image_path)
        img = resize_and_pad(img, target_size=IMAGE_SIDE_LEN)
        # remove alpha channel, if there is one
        img = img[:3]
        return img
        # return self.transform(img)
    
    def _prep_label(self, label_path):
        with open(label_path, 'r') as f:
            content = f.read()
        cleaned_labels = normalize_label_file_content(content)

        if not validate_labels(cleaned_labels):
            raise ValueError('Invalid label file:', label_path)

        lines = cleaned_labels.split('\n')
        return torch.tensor([
            [CLASS_MAPS.name_to_label[name] for name in line.split(' ')]
            for line in lines if line 
        ])
