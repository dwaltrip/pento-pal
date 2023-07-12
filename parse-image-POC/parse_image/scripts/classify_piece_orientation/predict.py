import os
import sys

from ultralytics import YOLO

from tasks.classify_piece_orientation import predict_classify_piece_orientation


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Incorrect args.")
        sys.exit(1)

    # weights_path = 'runs/detect/classify-train-test/weights/best.pt'
    weights_path = sys.argv[1]
    if not os.path.isfile(weights_path):
        print('weight file:', weights_path)
        raise Exception('weights not found')
    
    # data_dir = f'{AI_DIR}/pento-orientation-classify--TEST-images/CROPPED'
    data_dir = sys.argv[2]

    model = YOLO(weights_path)
    predict_classify_piece_orientation(model, data_dir)
