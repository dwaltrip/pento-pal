from functools import cache
import os

from ultralytics import YOLO

from settings import PROJECT_ROOT, RESULTS_ROOT


CORNER_PRED_TRAINING_RUN = 'detect-puzzle-box--2023-08-19--ts257--small'
PIECE_DETECT_TRAINING_RUN = 'detect-pieces--08-21--small'


def get_weights_path(training_run_name):
    return os.path.join(
        PROJECT_ROOT,
        RESULTS_ROOT,
        training_run_name,
        'weights',
        'best.pt',
    )


@cache
def load_corner_prediction_model():
    return YOLO(get_weights_path(CORNER_PRED_TRAINING_RUN))


@cache
def load_piece_detection_model():
    return YOLO(get_weights_path(PIECE_DETECT_TRAINING_RUN))
