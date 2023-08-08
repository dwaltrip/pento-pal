import my_ml_utils as mlu
import torch
from torch import nn
import torch.nn.functional as F


# This gives `nan` loss during training....
TEMPERATURE = 32

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
    temperature=TEMPERATURE,
)


NEIGHBOR_KERNEL = torch.tensor([
    [0.5, 1.0, 0.5],
    [1.0, 0.0, 1.0],
    [0.5, 1.0, 0.5],
]).unsqueeze(0).unsqueeze(0)

def calc_connectedness_penalty(preds):
    nc = preds.shape[1]
    preds_softmax = mlu.super_softmax_1d(preds, dim=1, b=TEMPERATURE)

    neighbor_sums = torch.zeros_like(preds_softmax)
    for i in range(nc):
        # unsqueeze(1) to add "channel" dimension, after batch dim
        preds_softmax_slice = preds_softmax[:, i, :, :].unsqueeze(1)

        conv_result = F.conv2d(
            F.pad(preds_softmax_slice, (1, 1, 1, 1)),
            NEIGHBOR_KERNEL,
        )
        # squeeze to remove "channel" dim we added at the top of the looop
        neighbor_sums[:, i, :, :] = conv_result.squeeze(1)


    # Penalize if less than 1 or more than 4 neighbors have the same class
    penalties = penalize_lt_1_or_gt_4_neighbors(neighbor_sums)
    # Calculate connectedness penalty
    connectedness_penalty = penalties.float().mean()

    return connectedness_penalty
    

def calc_class_count_penalty(preds):
    preds_softmax = F.softmax(preds, dim=1)

    # count number of cells predicted as each class
    preds_softmax_class_sums = preds_softmax.sum(dim=(2,3))
    # calculate absolute deviation from ideal count, then average
    class_count_penalty = torch.abs(preds_softmax_class_sums - 5).float().mean()
    return class_count_penalty


CLASS_COUNT_PENALTY_WEIGHT = 0.5
CONNECTEDNESS_PENALTY_WEIGHT = 1
def calculate_loss(preds, targets, debug=False):
    # Basic CrossEntropyLoss
    base_loss = F.cross_entropy(preds, targets)

    class_count_penalty = calc_class_count_penalty(preds)
    connectedness_penalty = calc_connectedness_penalty(preds)

    # Combined loss
    loss_parts = [
        base_loss,
        class_count_penalty * CLASS_COUNT_PENALTY_WEIGHT,
        connectedness_penalty * CONNECTEDNESS_PENALTY_WEIGHT,
    ]

    if debug or True:
        # print('\tpreds.shape', preds.shape)
        # print('\tpreds_softmax.shape', preds_softmax.shape)
        # print('\tpreds_softmax_class_sums.shape', preds_softmax_class_sums.shape)
        # print('\t', 'base_loss:', loss_parts[0].item())
        # print('\t', 'class_count_penalty:', loss_parts[1].item())
        # print('\t', 'connectedness_penalty:', loss_parts[2].item())
        # print('\t', 'loss_parts:', [l.item() for l in loss_parts])
        print('\t\tloss_parts: ', [round(l.item(), 3) for l in loss_parts])

    # import pdb; pdb.set_trace()

    return sum(loss_parts)
