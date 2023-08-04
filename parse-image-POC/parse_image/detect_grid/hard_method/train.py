import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, random_split, SubsetRandomSampler

from parse_image.detect_grid.common.dataset import GridLabelDataset
from parse_image.detect_grid.hard_method.model import get_custom_model
from parse_image.detect_grid.hard_method.config import *


IS_MPS_AVAILABLE = torch.backends.mps.is_available()

TRAIN_PERCENT = 0.8

def train_model(model, device):
    DEBUG = False

    batch_size = BATCH_SIZE
    epochs = NUM_EPOCHS
    # epochs = 200
    lr = LEARNING_RATE
    # lr = 0.005

    augment = False
    # augment = True

    # subset = None
    subset = 2
    batch_size = subset

    full_dataset = GridLabelDataset(IMAGE_DIR, LABEL_DIR, augment=augment)
    if subset:
        indices = list(x for x in SubsetRandomSampler(range(len(full_dataset))))[:subset]
        dataset = Subset(full_dataset, indices)
    else:
        dataset = full_dataset

    train_size = int(TRAIN_PERCENT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = []

    skip_validation = len(val_dataloader) == 0

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    print('------------------------------------------')
    print('Grid Predictor (head):')
    print(model.grid_predictor)
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

            # N, H, W, C = outputs.shape
            # # shape: [N*H*W, C]
            # outputs_reshape = outputs.reshape(N * H * W, C)
            # # shape: [N*H*W]
            # labels_reshape = labels.reshape(N * H * W)
            # loss = loss_fn(outputs_reshape, labels_reshape)

            # CrossEntropyLoss expects shape: [N, C, H, W]
            outputs_permuted = outputs.permute(0, 3, 1, 2)
            if DEBUG:
                print('outputs.shape:', outputs.shape)
                print('outputs_permuted.shape:', outputs_permuted.shape)
                print('labels.shape:', labels.shape)

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
    torch.save(model.state_dict(), TRAINED_MODEL_SAVE_PATH)
    print('Model saved to:', TRAINED_MODEL_SAVE_PATH)

if __name__ == '__main__':
    # device = 'cpu'
    device = torch.device('mps') if IS_MPS_AVAILABLE else 'cpu'
    print('device:', device)

    model = get_custom_model().to(device)

    train_model(model, device=device)
