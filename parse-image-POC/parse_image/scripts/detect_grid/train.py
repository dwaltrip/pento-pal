import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset, random_split

from parse_image.scripts.detect_grid.model import get_custom_model
from parse_image.scripts.detect_grid.dataset import GridLabelDataset
from parse_image.scripts.detect_grid.config import *


IS_MPS_AVAILABLE = torch.backends.mps.is_available()

TRAIN_PERCENT = 0.8

def train_model(model, device):
    full_dataset = GridLabelDataset(IMAGE_DIR, LABEL_DIR)
    # dataset = Subset(full_dataset, range(20))
    dataset = full_dataset

    train_size = int(TRAIN_PERCENT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    print('------------------------------------------')
    print('Grid Predictor (head):')
    print(model.grid_predictor)
    print('------------------------------------------')

    # Train the model
    print('')
    print(f'Starting the training loop...')
    print('\n\t'.join([
        f'Epochs = {NUM_EPOCHS}',
        f'Number of training examples: {len(dataset)}',
    ]))

    training_loss_per_epoch = []
    # validation_loss_per_epoch = []

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}]')

        epoch_training_loss = 0.0
        epoch_validation_loss = 0.0

        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            N, H, W, C = outputs.shape
            # shape: [N*H*W, C]
            outputs_reshape = outputs.reshape(N * H * W, C)
            # shape: [N*H*W]
            labels_reshape = labels.reshape(N * H * W)

            loss = loss_fn(outputs_reshape, labels_reshape)
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

        print('\t', f'train Loss: {epoch_training_loss / len(train_dataloader):.4f}')
        print('\t', f'val Loss: {epoch_validation_loss / len(val_dataloader):.4f}')

    print('Finished training.')
    # TODO: If the file exists, modify the name to not overwrite the old file
    #       Maybe timestamp it or something? Or give it a name?
    torch.save(model.state_dict(), TRAINED_MODEL_SAVE_PATH)
    print('Model saved to:', TRAINED_MODEL_SAVE_PATH)

if __name__ == '__main__':
    device = 'cpu'
    # device = torch.device('mps') if IS_MPS_AVAILABLE else 'cpu'
    print('device:', device)

    model = get_custom_model().to(device)

    train_model(model, device=device)
