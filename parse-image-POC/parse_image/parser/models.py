from functools import cache
import os

from ultralytics import YOLO

from settings import PROJECT_ROOT, RESULTS_ROOT
from parse_image.parser.logger import logger


# TODO: make these more configurable
CORNER_PRED_TRAINING_RUN = 'detect-puzzle-box--2023-08-19--ts257--small'
# PIECE_DETECT_TRAINING_RUN = 'detect-pieces--08-21--small'
# PIECE_DETECT_TRAINING_RUN = 'detect-pieces--08-23--ds100-small'
# PIECE_DETECT_TRAINING_RUN = 'detect-pieces--08-23--ds100-small-e60'
PIECE_DETECT_TRAINING_RUN = 'detect-pieces--08-24--ds104-small-e90'
# PIECE_DETECT_TRAINING_RUN = 'detect-pieces--08-23--ds100-med-e50'


logger.log(
    '[Models]',
    f'- corner detection : {CORNER_PRED_TRAINING_RUN}',
    f'- piece detection  : {PIECE_DETECT_TRAINING_RUN}',
    sep='\n\t',
)

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
