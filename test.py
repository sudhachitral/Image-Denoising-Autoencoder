import torch

model = Autoencoder()

# load from Google Drive (runtime access)
model.load_state_dict(torch.load(
    "/content/drive/MyDrive/autoencoder.pth",
    map_location=torch.device('cpu')
))

model.eval()
print("Model loaded successfully")
