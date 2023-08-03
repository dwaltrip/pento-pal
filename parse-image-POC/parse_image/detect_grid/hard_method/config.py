import os

from settings import AI_DATA_DIR, PROJECT_ROOT


# RUN_NAME = 'pento-custom-model-EXPERIMENT-1'
RUN_NAME = 'pento-detect-grid-2023-08-01'

IMAGE_SIDE_LEN = 224
IMAGE_SIZE = (3, IMAGE_SIDE_LEN, IMAGE_SIDE_LEN)

NUM_EPOCHS = 50
# BATCH_SIZE = 8
BATCH_SIZE = 64
# LEARNING_RATE = 0.001
LEARNING_RATE = 0.003
# LEARNING_RATE = 0.01
HIDDEN_LAYER_SIZE = 256
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'

DATA_PATH = os.path.join(AI_DATA_DIR, RUN_NAME)
IMAGE_DIR = os.path.join(DATA_PATH, 'images')
LABEL_DIR = os.path.join(DATA_PATH, 'labels')

WEIGHTS_DIR = os.path.join(PROJECT_ROOT, 'weights')
PRETRAINED_MODEL_SAVE_PATH = os.path.join(WEIGHTS_DIR, 'resnet50_modified.pth')
TRAINED_MODEL_SAVE_PATH = os.path.join(WEIGHTS_DIR, 'finetuned_grid_detector.pth')
