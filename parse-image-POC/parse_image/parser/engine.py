
from settings import NUM_CLASSES
from parse_image.parser.models import (
    load_corner_prediction_model,
    load_piece_detection_model,
)


DETECTION_THRESHOLD = 0.5


def get_puzzle_box_corners(image):
    # TODO: Cache this?
    model = load_corner_prediction_model()

    results = model.predict(image)
    result = results[0]
    boxes = [
        box for box in result.boxes
        if box.conf.item() > DETECTION_THRESHOLD
    ]
    # TODO: throw a custom error here
    assert len(boxes) == 1, 'Expected exactly 1 box, got: {}'.format(len(boxes))

    box = boxes[0]
    # TODO: we don't actually use puzzle_box... get rid of it?
    puzzle_box = box.xyxy[0].tolist()
    corners_from_keypoints = result.keypoints.squeeze(0).tolist()

    return corners_from_keypoints


def dewarp_rectangle(pil_image, corners, aspect_ratio):
    """
    De-warps a rectangular region in the given image.
    Params:
        pil_image: Input image (PIL Image)
        corners: List of four corner points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        aspect_ratio: Target aspect ratio (width/height)
    Returns:
        De-warped, cropped image
    """
    image = np.array(pil_image)
    # Detected corners from the image
    src_pts = np.array(corners, dtype=np.float32)
    # Idealized rectangle points
    dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    width_dist = max(
        cv2.norm(src_pts[0] - src_pts[1]),
        cv2.norm(src_pts[2] - src_pts[3]),
    )
    height_dist = max(
        cv2.norm(src_pts[0] - src_pts[3]),
        cv2.norm(src_pts[1] - src_pts[2]),
    )

    # ----------------------------------------------------------------------
    # TODO: we aren't using the height_dist at all... not sure about that...
    # Need to ask GPT-4 to explain this code
    # ----------------------------------------------------------------------
    width = int(width_dist)
    height = int(width / aspect_ratio)

    # Find homography and apply to the image
    H, _ = cv2.findHomography(src_pts, dst_pts)
    dewarped_image = cv2.warpPerspective(image, H, (width, height))

    return Image.fromarray(dewarped_image)


def get_piece_bounding_boxes(image):
    # TODO: Cache this?
    model = load_piece_detection_model()
    results = model.predict(image)

    boxes = [
        box for box in result.boxes
        if box.conf.item() > DETECTION_THRESHOLD
    ]
    # TODO: custom error
    assert len(boxes) == NUM_CLASSES, 'Incorrect numer of pieces detected

    def make_bounding_box(box):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = box.cls.item()
        class_name = result.names[class_id]
        return BoundingBox(
            class_id=class_id,
            top_left=Point(y=y1, x=x1),
            bot_right=Point(y=y2, x=x2),
        )
    return [make_bounding_box(box) for box in boxes]


def simple_map_boxes_to_grid(*args, **kwargs):
    raise NotImplementedError()
