import torch
import torch.nn.functional as F


def penalize_lt_1_or_gt_4(neighbor_sums):
    # shift neighbor_sums so that values between 1 and 4 are close to zero
    shifted_sums = neighbor_sums - 2.5
    # apply a smooth function that is close to zero between -1.5 and 1.5
    penalties = torch.sigmoid(shifted_sums) * torch.sigmoid(-shifted_sums)
    return penalties

import pdb;
def calc_connectedness_penalty(preds):
    nc = preds.shape[1]
    print('nc:', nc)
    preds_softmax = F.softmax(preds, dim=1)

    # Create 3x3 convolution kernel
    # kernel = torch.ones(nc, nc, 3, 3, device=preds.device, dtype=torch.float32)
    # kernel[:, :, 1, 1] = 0
    # Calculate neighbor sum for each class
    # neighbor_sums = F.conv2d(F.pad(preds_softmax, (1, 1, 1, 1)), kernel)

    kernel = torch.ones(1, 1, 3, 3)
    kernel[:, :, 1, 1] = 0 # set center to 0

    neighbor_sums = torch.zeros_like(preds_softmax)
    for i in range(nc):
        preds_softmax_slice = preds_softmax[:, i, :, :]
        neighbor_sums[:, i, :, :] = F.conv2d(
            F.pad(preds_softmax_slice, (1, 1, 1, 1)),
            kernel,
        )

    # Penalize if less than 1 or more than 4 neighbors have the same class
    penalties = penalize_lt_1_or_gt_4(neighbor_sums)
    # Calculate connectedness penalty
    connectedness_penalty = penalties.float().mean()

    pdb.set_trace()

    return connectedness_penalty
    

def calc_class_count_penalty(preds):
    preds_softmax = F.softmax(preds, dim=1)

    # count number of cells predicted as each class
    preds_softmax_class_sums = preds_softmax.sum(dim=(2,3))
    # calculate absolute deviation from ideal count, then average
    class_count_penalty = torch.abs(preds_softmax_class_sums - 5).float().mean()
    return class_count_penalty


CLASS_COUNT_PENALTY_WEIGHT = 1
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
        print('\t', 'loss_parts:', [round(l.item(), 3) for l in loss_parts])

    return sum(loss_parts)