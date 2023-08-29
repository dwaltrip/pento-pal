import itertools
import json
import os
from pathlib import Path
import sys

from settings import AI_DATA_DIR
from parse_image.utils.misc import parse_script_args
from parse_image.data import Point
from parse_image.detect_puzzle_box.labels import KEYPOINT_NAMES
# TODO: Folder structure should be inverted:
#   `parse_image.detect_puzzle_box.validation`
from parse_image.validation.puzzle_box_detection import (
    are_corner_keypoints_valid,
)


PUZZLE_BOX_CLASS_ID = 0

LABEL_STUDIO_JSON_FILENAME = 'label-studio-export-with-keypoints.json'


class LabelValidationError(Exception):
    pass


def flatten_list_of_lists(list_of_lists):
    return itertools.chain.from_iterable(list_of_lists)

def extract_annotations(label_studio_json_data):
    annotations = []
    errors = []

    for image in label_studio_json_data:
        image_annotations = image['annotations'][0]['result']
        filename = image['file_upload']

        try:
            annotations.append(dict(
                image={
                    'filename': filename,
                    'width': image_annotations[0]['original_width'],
                    'height': image_annotations[0]['original_height'],
                },
                puzzle_bounding_box=extract_puzzle_bbox(image_annotations),
                keypoints=extract_keypoints(image_annotations),
            ))
        except LabelValidationError as err:
            errors.append((filename, err))
            continue

    return annotations, errors


def extract_keypoints(image_annotations):
    keypoint_data = [
        extract_kp_data(annot['value'])
        for annot in image_annotations
        if annot['type'] == 'keypointlabels'
    ]
    kps_by_name = { kp['name']: kp for kp in keypoint_data }

    kp_as_point = lambda kp: Point(kp['x'], kp['y'])
    is_valid, err_msg = are_corner_keypoints_valid(
        [kp_as_point(kps_by_name[name]) for name in KEYPOINT_NAMES],
    )
    if not is_valid:
        raise LabelValidationError(err_msg)

    return kps_by_name

def extract_kp_data(annotation_value):
    val = annotation_value
    name = val['keypointlabels'][0]
    return dict(name=name, x=val['x'], y=val['y'])


def extract_puzzle_bbox(image_annotations):
    results = []

    for annotation in image_annotations:
        if annotation['type'] != 'rectanglelabels':
            continue

        val = annotation['value']
        results.append(dict(
            x=val['x'],
            y=val['y'],
            width=val['width'],
            height=val['height'],
            name=val['rectanglelabels'][0],
        ))
    
    if len(results) != 1:
        raise LabelValidationError(f'Expected 1 box, got {len(results)}')

    return results[0]


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


def construct_label_values_with_bbox_and_keypoints(parsed_label_data):
    pbb = convert_ls_bbox_to_yolo(parsed_label_data['puzzle_bounding_box'])
    keypoints = {
        name: convert_ls_keypoint_to_yolo(keypoint)
        for name, keypoint in parsed_label_data['keypoints'].items()
    }
    return [
        PUZZLE_BOX_CLASS_ID,
        pbb['x'],
        pbb['y'],
        pbb['width'],
        pbb['height'],

        *itertools.chain.from_iterable(
            [(keypoints[name]['x'], keypoints[name]['y']) for name in KEYPOINT_NAMES],
        )
    ]


# ---------------------------------------------
# TODO: move most of this into a function!
# ---------------------------------------------
if __name__ == '__main__':
    args = parse_script_args([
        'dataset_dir',
    ])
    data_dir = os.path.join(AI_DATA_DIR, args.dataset_dir)
    # image_dir = os.path.join(data_dir, 'images')
    # label_src_dir = os.path.join(data_dir, 'labels_original')
    label_dest_dir = os.path.join(data_dir, 'labels')

    label_studio_json_file = os.path.join(
        data_dir,
        LABEL_STUDIO_JSON_FILENAME,
    )
    with open(label_studio_json_file, 'r') as f:
        raw_label_studio_data = json.load(f)

    labels_per_image, errors = extract_annotations(raw_label_studio_data)

    if errors and len(errors) > 0:
        print('Validation errors while extracting data:')
        for filename, err in errors:
            print(f'  {filename}: {err}')
        sys.exit(1)
    else:
        print('No validation errors!')

    for data in labels_per_image:
        image_filename = data['image']['filename']
        label_path = os.path.join(
            label_dest_dir,
            Path(image_filename).with_suffix('.txt'),
        )

        label_values = construct_label_values_with_bbox_and_keypoints(data)
        label_file_content = ' '.join([str(val) for val in label_values])
        with open(label_path, 'w') as f:
            f.write(label_file_content)
