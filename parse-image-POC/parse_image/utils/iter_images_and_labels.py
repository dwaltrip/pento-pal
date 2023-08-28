import os

    
def iter_images_and_labels(image_dir, labels_dir, verbose=False):
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

    if verbose:
        print(
            f'Skipped {len(os.listdir(labels_dir)) - len(label_files)}',
            'non-txt files in labels_dir.',
        )

    for label_filename in label_files:
        image_filename = label_filename.replace('.txt', '.png')

        image_path = os.path.join(image_dir, image_filename)
        label_path = os.path.join(labels_dir, label_filename)

        if not os.path.exists(image_path):
            if verbose:
                print(f'Skipping {label_filename}: matching image not found.')
            continue

        yield image_path, label_path
