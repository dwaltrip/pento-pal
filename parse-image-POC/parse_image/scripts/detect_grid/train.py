import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from prep_custom_model import GridPredictor  # import the custom head class from the first script

# Load your prepared data
# TODO: Implement this!

# Convert the data to PyTorch tensors and normalize to [0,1]
transform = Compose([
    ToTensor()
])

# Set up your data loader
# TODO: Implement this!

# Initialize the model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

# Get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Define the custom head
num_classes = 12
model.roi_heads.box_predictor = GridPredictor(in_features, hidden_layer=256, num_classes=num_classes)

# Load the modified pre-trained weights
model.load_state_dict(torch.load('modified_model.pth'))

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Set up the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print loss every 10 batches
        if (i+1) % 10 == 0:
            print (f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
print('Finished Training')

# Save the trained model
torch.save(model.state_dict(), 'custom_trained_model.pth')
