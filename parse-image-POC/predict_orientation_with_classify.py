import os
from pathlib import Path

import torch
from ultralytics import YOLO

# previous_weights_path = 'runs/classify-train-test/weights/best.pt'
previous_weights_path = 'runs/classify-train-test2/weights/best.pt'

DATA_DIR = '/Users/danielwaltrip/all-files/projects/ai-data'


def get_child_items(folder):
    return [
        child for child in Path(folder).iterdir()
        if 'DS_Store' not in child.stem
    ]

def fmt_image_path(img):
    return os.path.join(
        os.path.basename(os.path.dirname(img)),
        os.path.basename(img),
    )


if __name__ == '__main__':
    if not os.path.isfile(previous_weights_path):
        print('weight file:', previous_weights_path)
        raise Exception('weights not found')

    model = YOLO(previous_weights_path)

    test_img_root_dir = os.path.join(
        DATA_DIR,
        'pento-orientation-classify--TEST-images/CROPPED',
    )

    class_dirs = get_child_items(test_img_root_dir)
    # class_dirs = [get_child_items(test_img_root_dir)[0]]

    test_image_paths = []
    for class_dir in class_dirs:
        test_image_paths.extend(get_child_items(class_dir))

    results = model.predict(test_image_paths)
    print('num results:', len(results))

    for result in results:
        sorted_probs, prob_indices = torch.sort(result.probs)
        TOP_N_PREDICTIONS = 3

        top_probs_and_classes = list(reversed(list(zip(
            [model.names[i.item()] for i in prob_indices],
            [p.item() for p in sorted_probs],
        ))))[:TOP_N_PREDICTIONS]

        cls_name, pred_percent = top_probs_and_classes[0]

        print(
            fmt_image_path(result.path),
            '--',
            f'{cls_name}: {pred_percent:>3.0%}',
        )
