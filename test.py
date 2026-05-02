import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from models.autoencoder import Autoencoder
from utils.noise import AddNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = Autoencoder().to(device)
model.load_state_dict(torch.load("autoencoder.pth", map_location=device))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    AddNoise()
])

# Data
test_data = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

# Get batch
dataiter = iter(test_loader)
noisy_imgs, _ = next(dataiter)

noisy_imgs = noisy_imgs.to(device)

with torch.no_grad():
    outputs = model(noisy_imgs).cpu()

# Plot
fig, axes = plt.subplots(2, 5, figsize=(10, 4))

for i in range(5):
    axes[0, i].imshow(noisy_imgs[i][0].cpu(), cmap='gray')
    axes[0, i].set_title("Noisy")
    axes[0, i].axis('off')

    axes[1, i].imshow(outputs[i][0], cmap='gray')
    axes[1, i].set_title("Denoised")
    axes[1, i].axis('off')

plt.show()
