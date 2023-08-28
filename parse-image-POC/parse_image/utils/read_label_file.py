
def read_puzzle_grid_label(label_path):
    with open(label_path) as f:
        lines = f.read().strip().split('\n')
    label_rows = [line.strip().split(' ') for line in lines]
    return label_rows
