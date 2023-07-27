import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from parse_image.scripts.detect_grid.model import get_custom_model
from parse_image.scripts.detect_grid.dataset import GridLabelDataset
from parse_image.scripts.detect_grid.config import *


def train_model(model):
    dataset = GridLabelDataset(IMAGE_DIR, LABEL_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    print('')
    print(f'Starting the training loop...')
    print('\n\t'.join([
        f'Epochs = {NUM_EPOCHS}',
        f'Number of training examples: {len(dataset)}',
    ]))

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}]')

        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(inputs)

            N, H, W, C = outputs.shape
            # shape: [N*H*W, C]
            outputs_reshape = outputs.permute(0, 2, 3, 1).reshape(N * H * W, C)
            # shape: [N*H*W]
            labels_reshape = labels.reshape(N * H * W)

            loss = loss_fn(outputs_reshape, labels_reshape)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('\t', f'train Loss: {loss.item():.4f}')

    print('Finished Training')
    torch.save(model.state_dict(), TRAINED_MODEL_SAVE_PATH)
    print('Model saved to:', TRAINED_MODEL_SAVE_PATH)

if __name__ == '__main__':
    print('device:', DEVICE)
    model = get_custom_model(NUM_CLASSES, HIDDEN_LAYER).to(DEVICE)

    train_model(model)
