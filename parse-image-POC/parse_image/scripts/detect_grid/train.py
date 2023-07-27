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
    for epoch in range(NUM_EPOCHS):
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss every 10 batches
            if (i+1) % 10 == 0:
                print (
                    f'Epoch [{epoch+1}/{NUM_EPOCHS}],',
                    f'Step [{i+1}/{len(dataloader)}],',
                    f'Loss: {loss.item():.4f}'
                )

    print('Finished Training')
    torch.save(model.state_dict(), TRAINED_MODEL_SAVE_PATH)

if __name__ == '__main__':
    print('device:', DEVICE)
    model = get_custom_model(NUM_CLASSES, HIDDEN_LAYER).to(DEVICE)

    train_model(model)
