import itertools
import json
import os

from settings import AI_DATA_DIR
from parse_image.utils.misc import parse_script_args


PUZZLE_BOX_CLASS_ID = 0

KEYPOINT_NAMES = ['top-left', 'top-right', 'bot-left', 'bot-right']

LABEL_STUDIO_JSON_FILENAME = 'label-studio-export-with-keypoints.json'


def flatten_list_of_lists(list_of_lists):
    return itertools.chain.from_iterable(list_of_lists)

def extract_annotations(label_studio_json_data):
    annotations = []
    for image in label_studio_json_data:
        annotation_results = image['annotations'][0]['result']

        label_data_for_image = dict(
            image={
                'filename': image['file_upload'],
                'width': annotation_results[0]['original_width'],
                'height': annotation_results[0]['original_height'],
            },
            puzzle_bounding_box=None,
            keypoints={},
        )
        for annotation in annotation_results:
            annot_val = annotation['value']
            if annotation['type'] == 'rectanglelabels':
                label_data_for_image['puzzle_bounding_box'] = {
                    'x': annot_val['x'],
                    'y': annot_val['y'],
                    'width': annot_val['width'],
                    'height': annot_val['height'],
                    'name': annot_val['rectanglelabels'][0],
                }
            elif annotation['type'] == 'keypointlabels':
                keypoint_name = annot_val['keypointlabels'][0]
                label_data_for_image['keypoints'][keypoint_name] = {
                    'name': keypoint_name,
                    'x': annotation['value']['x'],
                    'y': annotation['value']['y'],
                }
        annotations.append(label_data_for_image)
    return annotations


def convert_ls_bbox_to_yolo(bbox):
    """ normalize Label Studio coordinates into YOLOv8 format """
    x_center = bbox['x'] + (bbox['width'] / 2)
    y_center = bbox['y'] + (bbox['height'] / 2)
    # Label Studio gives percentages, YOLO needs values between 0 and 1
    return dict(
        x = x_center / 100.0,
        y = y_center / 100.0,
        height = bbox['height'] / 100.0,
        width = bbox['width'] / 100.0,
    )


def convert_ls_keypoint_to_yolo(keypoint):
    """ normalize Label Studio coordinates into YOLOv8 format """
    # Label Studio gives percentages, YOLO needs values between 0 and 1
    return dict(
        x = keypoint['x'] / 100.0,
        y = keypoint['y'] / 100.0,
    )


def create_label_file_with_bbox_and_keypoints(image_file, parsed_label_data):
    print()
    print(f'-- {image_file} --', end=' ')

    pbb = convert_ls_bbox_to_yolo(parsed_label_data['puzzle_bounding_box'])
    keypoints = {
        name: convert_ls_keypoint_to_yolo(keypoint)
        for name, keypoint in parsed_label_data['keypoints'].items()
    }

    values = [
        PUZZLE_BOX_CLASS_ID,
        pbb['x'],
        pbb['y'],
        pbb['width'],
        pbb['height'],

        *itertools.chain.from_iterable(
            [(keypoints[name]['x'], keypoints[name]['y']) for name in KEYPOINT_NAMES],
        )
    ]

    values_str = ' '.join([str(v) for v in values])
    label_file = os.path.join(dest_label_dir, image_file.replace('.jpg', '.txt'))
    with open(label_file, 'w') as f:
        f.write(values_str)


if __name__ == '__main__':
    args = parse_script_args([
        'dataset_dir',
    ])
    data_dir = os.path.join(AI_DATA_DIR, args.dataset_dir)

    label_studio_json_file = os.path.join(
        data_dir,
        LABEL_STUDIO_JSON_FILENAME,
    )
    with open(label_studio_json_file, 'r') as f:
        raw_label_studio_data = json.load(f)

    labels = extract_annotations(raw_label_studio_data)

    for data in labels:
        create_label_file_with_bbox_and_keypoints(data['image']['filename'], data)

    # print('-----------------------')
    # print(json.dumps(labels, indent=2))

    # image_dir = os.path.join(data_dir, 'images')
    # label_dir = os.path.join(data_dir, 'labels')

