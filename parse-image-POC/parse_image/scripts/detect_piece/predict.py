import os
import sys

from ultralytics import YOLO

from tasks.detect_piece.predict import predict_piece_detection

test_images = [
    'example-images/pento-1.png',
    'example-images/pento-2.png',
    'example-images/pento-3.png',
    'example-images/pento-4.png',
    'example-images/pento-5.png',
]

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Incorrect args.")
        sys.exit(1)

    # weights_path = 'runs/detect/train8/weights/best.pt'
    weights_path = sys.argv[1]

    model = YOLO(weights_path)
    predict_piece_detection(model, test_images, conf_threshold=None)
    # predict_piece_detection(model, test_images, conf_threshold=0.5)
