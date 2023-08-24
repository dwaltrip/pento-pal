from parse_image.parser.errors import CornerDetectionError
from parse_image.parser.models import load_corner_prediction_model


def get_puzzle_box_corners(image, conf_threshold=None):
    model = load_corner_prediction_model()

    results = model.predict(image, verbose=False)
    result = results[0]
    boxes = result.boxes
    if conf_threshold:
        boxes = [
            box for box in boxes
            if box.conf.item() > conf_threshold
        ]

    if len(boxes) != 1:
        raise CornerDetectionError(f'Expected 1 box, got {len(boxes)}.')

    box = boxes[0]
    # TODO: we don't actually use puzzle_box... get rid of it?
    puzzle_box = box.xyxy[0].tolist()
    corners_from_keypoints = result.keypoints.squeeze(0).tolist()

    if len(corners_from_keypoints) != 4:
        raise CornerDetectionError(
            f'Expected 4 corners, got {len(corners_from_keypoints)}.',
        )

    return corners_from_keypoints
