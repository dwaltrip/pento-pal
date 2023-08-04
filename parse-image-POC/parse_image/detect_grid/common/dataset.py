from itertools import zip_longest
import os

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from settings import CLASS_NAMES, CLASS_MAPS

from parse_image.utils.misc import is_image
from parse_image.utils.resize_and_pad import resize_and_pad
from parse_image.detect_grid.common.augment import get_augmentations
from parse_image.detect_grid.hard_method.config import (
    IMAGE_SIDE_LEN,
)
from parse_image.detect_grid.common.annotation_tool_v2 import (
    normalize_label_file_content,
    validate_labels,
)


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
            # transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD),
        ])

        # Track which image files are used, for debugging
        self._idx_to_image_filename = {}
        self._image_files_used = set()

        self.filepaths = self._prep_filepaths(self.image_filenames, self.label_filenames)
        self.prepped_data = self._prep_data(self.filepaths)

        print('GridLabelDataset', '(augmented)' if augment else '')
        print(
            f'\t{len(self.label_filenames)} labels',
            f'({len(self.filepaths)} with matching images)',
        )
        print(f'\t{len(self.prepped_data)} total training examples')
 
    def __getitem__(self, idx):
        self._image_files_used.add(self._idx_to_image_filename[idx])
        return self.prepped_data[idx]

    def __len__(self):
        return len(self.prepped_data)
     
    def _prep_data(self, filepaths):
        # def prep_from_files(image_filename, label_filename):
        #     assert image_filename or label_filename
        #     if not image_filename:
        #         image_filename = os.path.splitext(label_filename)[0] + '.png'
        #     elif not label_filename:
        #         label_filename = os.path.splitext(image_filename)[0] + '.txt'
        #     return (
        #         self._prep_image(os.path.join(self.image_dir, image_filename)),
        #         self._prep_label(os.path.join(self.label_dir, label_filename)),
        #     )
        prepped_data = [(
                self._prep_image(img_path),
                self._prep_label(label_path),
            )
            for img_path, label_path in filepaths
        ]
        for i, (img_path, label_path) in enumerate(filepaths):
            self._idx_to_image_filename[i] = (img_path, label_path)

        if self.augment:
            return [
                augmented_item 
                for item in prepped_data
                for augmented_item in get_augmentations(*item)
            ]
        else:
            return prepped_data

    def _prep_image(self, image_path):
        img = Image.open(image_path)
        img = resize_and_pad(img, target_size=IMAGE_SIDE_LEN)
        img =  self.transform(img)
        # remove alpha channel, if there is one
        img = img[:3]
        return img
    
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

    def _prep_filepaths(self, image_filenames, label_filenames):
        image_filename_set = set(image_filenames)
        skipped_labels = []
        prepped_filepaths = []

        for label_filename in label_filenames:
            image_filename = os.path.splitext(label_filename)[0] + '.png'

            if image_filename not in image_filename_set:
                skipped_labels.append(label_filename)
                continue

            prepped_filepaths.append((
                os.path.join(self.image_dir, image_filename),
                os.path.join(self.label_dir, label_filename)
            ))

        print(f'Images not found for {len(skipped_labels)} labels')
        return prepped_filepaths 
