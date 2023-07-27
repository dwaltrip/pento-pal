import os
from types import SimpleNamespace

from settings import AI_DATA_DIR, PROJECT_ROOT


CLASS_NAMES = ['f', 'i', 'l', 'n', 'p', 't', 'u', 'v', 'w', 'x', 'y', 'z']
CLASS_MAPS = SimpleNamespace(
    name_to_label= { name: i for i, name in enumerate(CLASS_NAMES) },
    label_to_name= { i: name for i, name in enumerate(CLASS_NAMES) },
    name_to_color= {
        'f' : '#f196f1',
        'i' : '#000',
        'l' : '#87bce8',
        'n' : '#f53f2a',
        'p' : '#e4d6b2',
        't' : '#457ddf',
        'u' : '#efab2c',
        'v' : '#6b43c8',
        'w' : '#d7d9db',
        'x' : '#3aae45',
        'y' : '#e7ee40',
        'z' : '#9fdd92',
    },
)

RUN_NAME = 'pento-custom-model-EXPERIMENT-1'

DATA_PATH = os.path.join(AI_DATA_DIR, RUN_NAME)
IMAGE_DIR = os.path.join(DATA_PATH, 'images')
LABEL_DIR = os.path.join(DATA_PATH, 'labels')

WEIGHTS_DIR = os.path.join(PROJECT_ROOT, 'weights')
MODEL_SAVE_PATH = os.path.join(WEIGHTS_DIR, 'resnet50_modified.pth')
TRAINED_MODEL_SAVE_PATH = os.path.join('finetuned_grid_detector.pth')

NUM_CLASSES = len(CLASS_NAMES)
HIDDEN_LAYER = 256
NUM_EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 0.001
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'
