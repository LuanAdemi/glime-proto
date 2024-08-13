import torchvision.models as models
from PIL import Image
import torch
from torchvision import transforms

from torchvision.datasets import MNIST

from torch.utils.data import DataLoader

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

model = models.resnet18(num_classes=10).to(device)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False).to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
    # rotate
    transforms.RandomRotation(10)
])

train_data = MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=512, shuffle=True, drop_last=True)

epochs = 20

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}')

torch.save(model.state_dict(), 'models/resnet_mnist.pt')