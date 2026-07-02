import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -----------------
# Configurações
# -----------------

IMG_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 10

device = torch.device("cpu")

# -----------------
# Transformações
# -----------------

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# -----------------
# Dataset
# -----------------

train_dataset = datasets.ImageFolder(
    root="dataset/train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    root="dataset/val",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE
)

# -----------------
# CNN
# -----------------

class TomatoCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128 * 32 * 32, 512),
            nn.ReLU(),

            nn.Linear(512, 4)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


model = TomatoCNN().to(device)

# -----------------
# Loss + Optimizer
# -----------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# -----------------
# Treinamento
# -----------------

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Época {epoch+1}/{EPOCHS} | "
        f"Loss: {running_loss:.4f}"
    )

print("Treinamento finalizado!")

torch.save(
    model.state_dict(),
    "models/tomato_cnn.pth"
)

print("Modelo salvo!")