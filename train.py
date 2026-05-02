import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.autoencoder import Autoencoder
from utils.noise import AddNoise

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    AddNoise()
])

# Dataset
train_data = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model
model = Autoencoder().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 65

for epoch in range(epochs):
    for noisy_imgs, _ in train_loader:

        noisy_imgs = noisy_imgs.to(device)

        # ⚠️ FIX: you need clean images separately
        clean_imgs = noisy_imgs.clone().detach()

        outputs = model(noisy_imgs)
        loss = criterion(outputs, clean_imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "autoencoder.pth")
print("Model saved!")
