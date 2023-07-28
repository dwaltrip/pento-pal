import os
from types import SimpleNamespace

from settings import AI_DATA_DIR, PROJECT_ROOT


RUN_NAME = 'pento-custom-model-EXPERIMENT-1'

IMAGE_SIDE_LEN = 224
IMAGE_SIZE = (3, IMAGE_SIDE_LEN, IMAGE_SIDE_LEN)

NUM_EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 0.001
HIDDEN_LAYER_SIZE = 256
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'

CLASS_NAMES = ['f', 'i', 'l', 'n', 'p', 't', 'u', 'v', 'w', 'x', 'y', 'z']
NUM_CLASSES = len(CLASS_NAMES)
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

DATA_PATH = os.path.join(AI_DATA_DIR, RUN_NAME)
IMAGE_DIR = os.path.join(DATA_PATH, 'images')
LABEL_DIR = os.path.join(DATA_PATH, 'labels')

WEIGHTS_DIR = os.path.join(PROJECT_ROOT, 'weights')
PRETRAINED_MODEL_SAVE_PATH = os.path.join(WEIGHTS_DIR, 'resnet50_modified.pth')
TRAINED_MODEL_SAVE_PATH = os.path.join(WEIGHTS_DIR, 'finetuned_grid_detector.pth')
