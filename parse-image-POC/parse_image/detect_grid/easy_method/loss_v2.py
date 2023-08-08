import my_ml_utils as mlu
import torch
from torch import nn
import torch.nn.functional as F


def print_lines(*args, **kwargs):
    print(*args, **kwargs, sep='\n')


# each cell in the grid should have 1 to 4 matching neighbors
VALID_MATCHING_NEIGHBORS_RANGE = (1, 4)

def base_penalty_fn(x):
    c1, c2 = VALID_MATCHING_NEIGHBORS_RANGE
    return ((x - c1)*(x - c2))

def zero_penalty(x):
    return 0

penalize_lt_1_or_gt_4_neighbors = mlu.build_smooth_piecewise_fn(
    [
        base_penalty_fn, # x <  1
        zero_penalty,    # 1 <= x <= 4
        base_penalty_fn, # 4 <  x
    ],
    points=VALID_MATCHING_NEIGHBORS_RANGE,
)

# kernel = torch.ones(1, 1, 3, 3)
# kernel[:, :, 1, 1] = 0 # set center to 0
kernel = torch.tensor([
    [0.5, 1.0, 0.5],
    [1.0, 0.0, 1.0],
    [0.5, 1.0, 0.5],
]).unsqueeze(0).unsqueeze(0)


# def calc_disconnected_penalty(preds):
def calc_disconnected_penalty(grid_one_hot):
    # --------------------------
    # TODO: fix this!! 
    nc = 3
    preds_softmax = grid_one_hot
    # --------------------------

    neighbor_counts = torch.zeros(preds_softmax.shape)

    for i in range(nc):
        ith_class_slice = preds_softmax[i, :, :]
        # add batch, channel dimensions to make `F.conv2d` happy
        ith_class_slice = ith_class_slice.unsqueeze(0).unsqueeze(0)

        conv_result = F.conv2d(
            F.pad(ith_class_slice, (1, 1, 1, 1)),
            kernel,
        )
        print()
        print_lines('conv_result:', conv_result)
        # get rid of superfluous channel dim with squeeze
        neighbor_counts[i, :, :] = conv_result.squeeze(0)

    print()
    print('---------------------------------------------')
    print_lines('neighbor_counts:', neighbor_counts)

    penalties_per_cell = penalize_lt_1_or_gt_4_neighbors(neighbor_counts)
    print()
    print_lines('penalties_per_cell:', penalties_per_cell)
    print()
    print_lines('penalties_per_cell * grid_one_hot:', penalties_per_cell * grid_one_hot)
    disconnected_penalty = penalties_per_cell.float().mean()
    return disconnected_penalty
    

def calc_class_count_penalty(preds):
    preds_softmax = F.softmax(preds, dim=1)
    # count number of cells predicted as each class
    preds_softmax_class_sums = preds_softmax.sum(dim=(2,3))
    # calculate absolute deviation from ideal count, then average
    class_count_penalty = torch.abs(preds_softmax_class_sums - 5).float().mean()
    return class_count_penalty


CLASS_COUNT_PENALTY_WEIGHT = 1
DISCONNECTED_PENALTY_WEIGHT = 1

def calculate_loss(preds, targets, debug=False):
    # Basic CrossEntropyLoss
    base_loss = F.cross_entropy(preds, targets)

    class_count_penalty = calc_class_count_penalty(preds)
    disconnectedness_penalty = calc_disconnectedness_penalty(preds)

    # Combined loss
    loss_parts = [
        base_loss,
        class_count_penalty * CLASS_COUNT_PENALTY_WEIGHT,
        disconnected_penalty * DISCONNECTED_PENALTY_WEIGHT,
    ]
    if debug or True:
        print('\t', 'loss_parts:', [round(l.item(), 3) for l in loss_parts])

    return sum(loss_parts)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():
    grid = torch.tensor([
        [0, 1, 1],
        [2, 1, 0],
        [2, 0, 2],
    ])
    indices_ranked_by_connectedness = [
        (0, 1), # score: 2
        (0, 2), (1, 0), (1, 1), (2, 0), # score: 1
        (0, 0), (1, 2), (2, 1), (2, 2), # score: 0
    ]
    # ---------

    grid_one_hot = F.one_hot(grid, 3).permute(2,0,1)
    print('--------')
    print_lines('grid:', grid)
    print('--------')
    print_lines('grid_one_hot:', grid_one_hot)

    dc_penalty = calc_disconnected_penalty(grid_one_hot.float())

    print()
    print('dc_penalty:', dc_penalty)


if __name__ ==  '__main__':
    main()
