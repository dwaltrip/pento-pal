from settings import NUM_CLASSES
from parse_image.data.bounding_box import PieceBoundingBox
from parse_image.parser.errors import PieceDetectionError
from parse_image.parser.models import load_piece_detection_model
from parse_image.parser.logging import logger


DEBUG = False
VERBOSE = True

def get_piece_bounding_boxes(image, conf_threshold=None):
    model = load_piece_detection_model()
    results = model.predict(image, verbose=False)
    result = results[0]
    boxes = result.boxes

    if conf_threshold:
        boxes = [
            box for box in result.boxes
            if box.conf.item() > conf_threshold
        ]
        num_dropped = len(result.boxes) - len(boxes)
        if VERBOSE and num_dropped > 0:
            logger.log(
                f'Filtering by conf... {num_dropped} boxes were dropped',
                f'(box.conf < {conf_threshold})',
            )

    # Take the boxes with the highest confidence for each piece type.
    boxes_in_conf_order = sorted(boxes, key=lambda box: box.conf.item(), reverse=True)
    boxes = []
    seen = set()
    for box in boxes_in_conf_order:
        if box.cls.item() not in seen:
            boxes.append(box)
            seen.add(box.cls.item())
    
    if len(boxes_in_conf_order) != len(boxes):
        print(f'\tNOTE: {len(boxes_in_conf_order) - len(boxes)} bounding boxes were discarded.')

    if DEBUG:
        return len(result.boxes), len(boxes)

    # TODO: There are fallbacks we can attempt to do here.
    # E.g. For each piece type, take the one with the highest confidence.
    # Need to investigate these cases.
    if len(boxes) != NUM_CLASSES:
        msg = ' '.join([
            f'Incorrect numer of pieces detected.',
            f'Expected {NUM_CLASSES}, got {len(boxes)}.',
        ])
        data = dict(count=len(boxes), raw_count=len(result.boxes))
        raise PieceDetectionError(msg, data=data)

    return [PieceBoundingBox.from_prediction(box) for box in boxes]
