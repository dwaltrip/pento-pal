import os
import sys
from types import SimpleNamespace
import yaml


# TODO: use stdlib to improve this
def parse_script_args(arg_names):
    num_args = len(arg_names)

    if len(sys.argv) != num_args + 1:
        print('Incorrect number of args.')
        print(f'Provide {num_args} args:', ', '.join(arg_names))
        sys.exit(1)

    return SimpleNamespace(**dict(zip(arg_names, sys.argv[1:])))


def write_yaml_file(filepath, data):
    if os.path.exists(filepath):
        raise FileExistsError(f'{filepath} already exists.')

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        yaml.dump(data, f)
