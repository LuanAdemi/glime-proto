import matplotlib.pyplot as plt

import torch

from torchvision.datasets import MNIST
from torchvision import transforms

from torch.utils.data import DataLoader

from glow import GLOW

import normflows as nf

import torchvision as tv
import numpy as np


transform = transforms.Compose([
    transforms.ToTensor(),
    nf.utils.Scale(255. / 256.),
    nf.utils.Jitter(1 / 256.)
])

train_data = MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=256, shuffle=True, drop_last=True)

flow = GLOW(1, 32, (1, 28, 28), 10, 256)
flow.fit(train_loader, 50000)

flow.model.save('models/glow_mnist_1_32_256_50000.pt')
