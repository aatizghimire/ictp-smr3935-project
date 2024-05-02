import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset
import time

# Define constants
IMAGE_SIZE = 28 * 28
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10
LEARNING_RATE = 0.1
EPOCHS = 10
BATCH_SIZE = 64

# Define neural network model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(IMAGE_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='.', train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root='.', train=False, download=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize neural network model
model = NeuralNetwork()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Training
start_time = time.time()
for epoch in range(EPOCHS):
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.view(-1, IMAGE_SIZE)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
end_time = time.time()
training_time = end_time - start_time
print(f"Training time: {training_time} seconds")

# Testing
correct = 0
total = 0
start_time = time.time()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.view(-1, IMAGE_SIZE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

end_time = time.time()
testing_time = end_time - start_time
print(f"Testing time: {testing_time} seconds")

print(f"Accuracy on test dataset: {100 * correct / total}%")