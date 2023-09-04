from collections import namedtuple

from pento_parser_poc.errors import CornerDetectionError
from pento_parser_poc.models import load_corner_prediction_model
from pento_parser_poc.logger import logger


CornerPred = namedtuple('CornerPred', ['box', 'conf', 'keypoints'])

def make_corner_pred(box_pred):
    box = box_pred.boxes[0]
    return CornerPred(box, box.conf.item(), box_pred.keypoints)


def get_puzzle_box_corners(image, conf_threshold=None):
    model = load_corner_prediction_model()
    results = model.predict(image, verbose=False)
    result = results[0]

    preds = [make_corner_pred(box_pred) for box_pred in result]

    if conf_threshold:
        preds = [pred for pred in preds if pred.conf >= conf_threshold]
    
    if len(preds) != 1:
        logger.log(
            f'Corner detection found multiple boxes: ({len(preds)}).',
            'Using box w/ highest confidence.',
        )

    # Take the prediction with the highest confidence
    best_pred = sorted(preds, key=lambda x: x.conf, reverse=True)[0]
    corners_from_keypoints = best_pred.keypoints.tolist()

    if len(corners_from_keypoints) != 4:
        raise CornerDetectionError(
            f'Expected 4 corners, got {len(corners_from_keypoints)}.',
        )

    return corners_from_keypoints
