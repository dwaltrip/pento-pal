import torch
import torch.nn.functional as F


def foo_1():
    nc = 2
    input = torch.randn(1, nc, 4, 4) # batch_size = 1, num_classes = 2
    input[:, 0, :, :] = 1. # set first class to 1
    input[:, 1, :, :] = 2. # set second class to 2

    # create dummy kernel
    kernel = torch.ones(nc, nc, 3, 3)
    kernel[:, :, 1, 1] = 0 # set center to 0

    # apply convolution
    output = F.conv2d(F.pad(input, (1, 1, 1, 1)), kernel)

    print("Input:", input.int())
    print("Output:", output.int())


def foo_2():
    nc = 2
    # create dummy input
    input = torch.randn(1, nc, 4, 4) # batch_size = 1
    input[:, 0, :, :] = 1. # set first class to 1
    input[:, 1, :, :] = 2. # set second class to 2

    output = torch.zeros_like(input)

    kernel = torch.ones(1, 1, 3, 3)
    kernel[:, :, 1, 1] = 0 # set center to 0

    # create dummy kernel
    for i in range(nc):

        # apply convolution
        output[:, i:i+1, :, :] = F.conv2d(
            F.pad(
                input[:, i:i+1, :, :],
                (1, 1, 1, 1)
            ),
            kernel,
        )

    print("Input:", input.int())
    print("Output:", output.int())

foo_2()