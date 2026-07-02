import torch
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# -------------------
# Configurações
# -------------------

IMG_SIZE = 256
BATCH_SIZE = 32

device = torch.device("cpu")

# -------------------
# Transformações
# -------------------

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# -------------------
# Dataset
# -------------------

val_dataset = datasets.ImageFolder(
    root="dataset/val",
    transform=transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -------------------
# Modelo
# -------------------

class TomatoCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*32*32,512),
            nn.ReLU(),
            nn.Linear(512,4)
        )

    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = TomatoCNN()

model.load_state_dict(
    torch.load("models/tomato_cnn.pth")
)

model.eval()

# -------------------
# Avaliação
# -------------------

correct = 0
total = 0

with torch.no_grad():

    for images, labels in val_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"Acurácia: {accuracy:.2f}%")