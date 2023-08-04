from collections import namedtuple
import os
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, random_split, SubsetRandomSampler

from settings import AI_DATA_DIR, PROJECT_ROOT
from parse_image.detect_grid.common.dataset import GridLabelDataset
from parse_image.detect_grid.easy_method.model import get_custom_model

from parse_image.detect_grid.easy_method.draw_grid_lines import add_grid_lines_and_show


IS_MPS_AVAILABLE = torch.backends.mps.is_available()

TRAIN_PERCENT = 0.8

Hyperparams = namedtuple('Hyperparams', [
    'epochs',
    'batch_size',
    'lr',
    'augment',
    'subset',
    'debug',
])
def build_hyperparams(**kwargs):
    subset = kwargs.get('subset', None)
    batch_size = kwargs.get('batch_size', None)
    assert subset or batch_size, 'Must specify either subset or batch_size'
    return Hyperparams(
        epochs=kwargs['epochs'],
        batch_size=(subset if subset else batch_size),
        lr=kwargs['lr'],
        augment=kwargs.get('augment', False),
        subset=subset,
        debug=kwargs.get('debug', False),
    )


def train_model(model, device, data_dir, save_path, hyp):
    image_dir = os.path.join(data_dir, 'images')
    label_dir = os.path.join(data_dir, 'labels')
    full_dataset = GridLabelDataset(image_dir, label_dir, augment=hyp.augment)

    # add_grid_lines_and_show([image for i, (image, label) in enumerate(full_dataset) if i < 8])
    # assert False, 'stop here'

    subset = hyp.subset
    epochs = hyp.epochs
    batch_size = hyp.batch_size
    lr = hyp.lr
    augment = hyp.augment

    if subset:
        indices = list(x for x in SubsetRandomSampler(range(len(full_dataset))))[:subset]
        dataset = Subset(full_dataset, indices)
    else:
        dataset = full_dataset

    train_size = int(TRAIN_PERCENT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    if subset:
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = []
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    skip_validation = len(val_dataloader) == 0

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    print('------------------------------------------')
    print(f'Model head ({model.head.__class__.__name__}):')
    print(model.head)
    print('------------------------------------------')

    # Train the model
    print(f'Run info / hyperparams:')
    print(
        '\t' + f'Number of frozen layers:',
        len(list(name for name, p in model.named_parameters() if not p.requires_grad))
    )
    print('\t' + f'Dataset size =', len(dataset))
    print('\t' + f'Train data size =', len(train_dataloader.dataset))
    print('\t' + f'Val data size =', len(getattr(val_dataloader, 'dataset', [])))
    print('\t' + f'-------------------')
    print('\t' + f'Augment = {augment}')
    print('\t' + f'Epochs = {epochs}')
    print('\t' + f'lr = {lr}')
    print('\t' + f'batch_size = {batch_size}')
    print()

    training_t0 = time.time()

    print(f'Starting the training loop...')
    for epoch in range(epochs):
        t0 = time.time()
        print(f'Epoch [{epoch+1}/{epochs}]')

        epoch_training_loss = 0.0
        epoch_validation_loss = 0.0

        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            # print('outputs.shape', outputs.shape)

            # CrossEntropyLoss expects shape: [N, C, H, W]
            outputs_permuted = outputs.permute(0, 3, 1, 2)
            loss = loss_fn(outputs_permuted, labels)
            epoch_training_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_dataloader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                N, H, W, C = outputs.shape
                outputs_reshape = outputs.reshape(N * H * W, C)
                labels_reshape = labels.reshape(N * H * W)
                loss = loss_fn(outputs_reshape, labels_reshape)
                epoch_validation_loss += loss.item()

        t1 = time.time()
        print('\t' + f'time: {t1-t0:.2f} seconds')
        print('\t' + f'train Loss: {epoch_training_loss / len(train_dataloader):.4f}')
        if(not skip_validation):
            print('\t' + f'val Loss: {epoch_validation_loss / len(val_dataloader):.4f}')
        
        EXIT_EARLY_THRESHOLD = 0.01
        if epoch_training_loss < EXIT_EARLY_THRESHOLD:
            print(f'Training loss is below {EXIT_EARLY_THRESHOLD}. Stopping early.')
            break
    
    # only print image filenames when doing small debugging runs
    if len(full_dataset._image_files_used) < 10:
        print('Images used:')
        for f in full_dataset._image_files_used:
            print('\t' + str(f))
        print('------------')

    print('Finished training.')
    print('Total training time:', time.time() - training_t0, 'seconds')

    # TODO: If the file exists, modify the name to not overwrite the old file
    #       Maybe timestamp it or something? Or give it a name?
    torch.save(model.state_dict(), save_path)
    print('Model saved to:', save_path)

if __name__ == '__main__':
    device = 'cpu'
    # Getting an error with MPS on my machine :(
    # device = torch.device('mps') if IS_MPS_AVAILABLE else 'cpu'
    print('device:', device)

    hyp = build_hyperparams(
        # epochs=50,
        epochs=30,
        batch_size=32,
        # lr=0.005,
        lr=0.003,
        # augment=True,
        subset=None,
    )
    hyp_try_to_overfit = build_hyperparams(
        subset=2,
        epochs=100,
        lr=0.005,
        augment=False,
        # debug=True,
    )
    backbone_weight_path = os.path.join(PROJECT_ROOT, 'weights', 'grid-easy-backbone.pth')
    model = get_custom_model(backbone_weight_path)
    model.to(device)

    import torchinfo
    torchinfo.summary(model, input_size=(2, 3, 224, 224))
    # assert False

    data_dir = os.path.join(AI_DATA_DIR, 'detect-grid-easy--2023-08-03')
    save_path = os.path.join(PROJECT_ROOT, 'weights', 'grid-easy-model.pth')

    train_model(model, data_dir=data_dir, device=device, save_path=save_path, hyp=hyp) 
    # train_model(model, data_dir=data_dir, device=device, save_path=save_path, hyp=hyp_try_to_overfit)
