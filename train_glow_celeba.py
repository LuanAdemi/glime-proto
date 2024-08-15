import matplotlib.pyplot as plt

import torch

from torchvision.datasets import CelebA
from torchvision import transforms

from torch.utils.data import DataLoader

from glow import GLOW

import normflows as nf

import torchvision as tv
import numpy as np


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(96),
    transforms.CenterCrop(96),
    nf.utils.Scale(255. / 256.),
    nf.utils.Jitter(1 / 256.)
])

train_data = CelebA(root='./data', transform=transform, download=True)

train_loader = DataLoader(train_data, batch_size=48, shuffle=True, drop_last=True)

flow = GLOW(3, 32, (3, 96, 96), 40, 512)
flow.fit(train_loader, 50000)

flow.model.save('models/glow_celeba_3_32_256_50000.pt')
