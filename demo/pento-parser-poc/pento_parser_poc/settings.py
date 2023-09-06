import os
from pathlib import Path
from types import SimpleNamespace


CLASS_NAMES = ['f', 'i', 'l', 'n', 'p', 't', 'u', 'v', 'w', 'x', 'y', 'z']

NUM_CLASSES = len(CLASS_NAMES)

CLASS_MAPS = SimpleNamespace(
    name_to_class_id = { name: i for i, name in enumerate(CLASS_NAMES) },
    class_id_to_name= { i: name for i, name in enumerate(CLASS_NAMES) },
    name_to_color= {
        'f' : '#f196f1',
        'i' : '#000',
        'l' : '#87bce8',
        'n' : '#f53f2a',
        'p' : '#e4d6b2',
        't' : '#457ddf',
        'u' : '#efab2c',
        'v' : '#6b43c8',
        'w' : '#d7d9db',
        'x' : '#3aae45',
        'y' : '#e7ee40',
        'z' : '#9fdd92',
    },
)

GRID = SimpleNamespace(height=10, width=6)

PACKAGE_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = PACKAGE_DIR.parent.absolute()

WEIGHTS_DIR = os.path.join(PACKAGE_DIR, 'weights')

def make_weight_path(model_name):
    return os.path.join(WEIGHTS_DIR, Path(model_name).with_suffix('.pt'))

WEIGHT_FILES = SimpleNamespace(
    CORNER_PRED=make_weight_path('detect-puzzle-box--08-30--ds367-small-e80'),
    PIECE_DETECT=make_weight_path('detect-pieces--08-31--ds147-small-e120'),
)
