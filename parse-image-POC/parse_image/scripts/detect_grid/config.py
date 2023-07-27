import os

from settings import AI_DATA_DIR, PROJECT_ROOT


RUN_NAME = 'pento-custom-model-EXPERIMENT-1'

DATA_PATH = os.path.join(AI_DATA_DIR, RUN_NAME)
IMAGE_DIR = os.path.join(DATA_PATH, 'images')
LABEL_DIR = os.path.join(DATA_PATH, 'labels')

WEIGHTS_DIR = os.path.join(PROJECT_ROOT, 'weights')
MODEL_SAVE_PATH = os.path.join(WEIGHTS_DIR, 'resnet50_modified.pth')
TRAINED_MODEL_SAVE_PATH = os.path.join('finetuned_grid_detector.pth')

NUM_CLASSES = 12
HIDDEN_LAYER = 256
NUM_EPOCHS = 10
BATCH_SIZE = 4
LEARNING_RATE = 0.001
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'
