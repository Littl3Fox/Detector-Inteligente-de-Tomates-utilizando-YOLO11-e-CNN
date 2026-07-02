import torch
import torch.nn as nn

from torchvision import transforms
from PIL import Image

# ==========================
# CONFIGURAÇÕES
# ==========================

IMAGE_PATH = "imagens_teste/tomate_estragado.jpg"

CLASSES = [
    "Damaged",
    "Old",
    "Ripe",
    "Unripe"
]

IMG_SIZE = 256

# ==========================
# MODELO
# ==========================

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


# ==========================
# CARREGA MODELO
# ==========================

model = TomatoCNN()

model.load_state_dict(
    torch.load(
        "models/tomato_cnn.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()

# ==========================
# TRANSFORMAÇÃO
# ==========================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ==========================
# CARREGA IMAGEM
# ==========================

image = Image.open(IMAGE_PATH)

image = image.convert("RGB")

image_tensor = transform(image)

image_tensor = image_tensor.unsqueeze(0)

# ==========================
# PREVISÃO
# ==========================

with torch.no_grad():

    output = model(image_tensor)

    probabilities = torch.softmax(
        output,
        dim=1
    )

    confidence, predicted = torch.max(
        probabilities,
        1
    )

classe = CLASSES[predicted.item()]

print()
print("Imagem:", IMAGE_PATH)
print("Classe:", classe)
print(
    f"Confiança: {confidence.item()*100:.2f}%"
)
print()