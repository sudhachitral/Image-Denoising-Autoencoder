import torch

class AddNoise(object):
    def __call__(self, img):
        noise = torch.randn_like(img) * 0.5
        return img + noise
