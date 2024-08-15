from collections import defaultdict
import pickle
import matplotlib.pyplot as plt

import torch

from torchvision.datasets import CelebA
from torchvision import transforms

from torch.utils.data import DataLoader
from tqdm import tqdm

from glow import GLOW

import normflows as nf

import torchvision as tv
import numpy as np


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(96),
    transforms.CenterCrop(96),
])

train_data = CelebA(root='./data', transform=transform, download=True)

flow = GLOW(3, 32, (3, 96, 96), 40, 256)
flow.model.load('models/glow/glow_celeba_3_32_256_50000.pt')



"""
For each class, we want to find the mean of the latent vectors that correspond to that class.
"""

latent_means_pos = defaultdict(list)
latent_means_neg = defaultdict(list)

N = 100000 #len(train_data)

for j, (x, y) in enumerate(tqdm(train_data)):
    x = x.to('cuda:1')
    with torch.no_grad():
        z = flow.to_latent(x.unsqueeze(0))[0]

        # collect latent vectors for each class
        for i, v in enumerate(y):
            if v.item() == 1:
                latent_means_pos[i].append(z)
            else:
                latent_means_neg[i].append(z)
    if j >= N:
        break
    
latent_manipulators = {}

for i in range(40):
    latent_means_pos[i] = torch.stack(latent_means_pos[i]).mean(axis=0).cpu()
    latent_means_neg[i] = torch.stack(latent_means_neg[i]).mean(axis=0).cpu()
    latent_manipulators[i] = latent_means_pos[i] - latent_means_neg[i]

pickle.dump(latent_manipulators, open('latent_manipulators.pkl', 'wb'))
