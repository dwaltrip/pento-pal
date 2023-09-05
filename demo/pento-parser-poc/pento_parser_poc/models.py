from functools import cache

from ultralytics import YOLO

from pento_parser_poc.settings import WEIGHT_FILES


@cache
def load_corner_prediction_model():
    return YOLO(WEIGHT_FILES.CORNER_PRED)


@cache
def load_piece_detection_model():
    return YOLO(WEIGHT_FILES.PIECE_DETECT)
