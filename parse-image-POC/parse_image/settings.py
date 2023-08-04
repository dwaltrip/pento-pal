from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).parent.parent.absolute()

AI_DATA_DIR = '/Users/danielwaltrip/all-files/projects/ai-data/pentominoes'

RESULTS_ROOT = 'runs'

CLASS_NAMES = ['f', 'i', 'l', 'n', 'p', 't', 'u', 'v', 'w', 'x', 'y', 'z']
NUM_CLASSES = len(CLASS_NAMES)
CLASS_MAPS = SimpleNamespace(
    name_to_label= { name: i for i, name in enumerate(CLASS_NAMES) },
    label_to_name= { i: name for i, name in enumerate(CLASS_NAMES) },
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