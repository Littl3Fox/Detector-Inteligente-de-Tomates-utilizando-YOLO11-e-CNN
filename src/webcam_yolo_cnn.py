import cv2
import torch
import torch.nn as nn

from ultralytics import YOLO
from torchvision import transforms
from PIL import Image

# ==========================
# YOLO
# ==========================

yolo = YOLO("runs/detect/runs/tomato_detector/weights/best.pt")

# ==========================
# CNN
# ==========================

CLASSES = [
    "Damaged",
    "Old",
    "Ripe",
    "Unripe"
]

IMG_SIZE = 256


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

        x=self.features(x)
        x=self.classifier(x)

        return x


cnn = TomatoCNN()

cnn.load_state_dict(
    torch.load(
        "models/tomato_cnn.pth",
        map_location="cpu"
    )
)

cnn.eval()
torch.set_grad_enabled(False)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = yolo(
    frame,
    verbose=False,
    imgsz=640,
    conf = 0.6
    )
    for result in results:

        for box in result.boxes:
            

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )
            h, w = frame.shape[:2]

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            image = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(image)

            tensor = transform(image)

            tensor = tensor.unsqueeze(0)

            with torch.no_grad():

                output = cnn(tensor)

                probs = torch.softmax(output, dim=1)

                conf, pred = torch.max(probs, 1)

            classe = CLASSES[pred.item()]

            porcentagem = conf.item()*100

            cv2.rectangle(
                frame,
                (x1,y1),
                (x2,y2),
                (0,255,0),
                2
            )
            det_conf = box.conf.item()

            texto = f"{classe} CNN:{porcentagem:.1f}% YOLO:{det_conf*100:.1f}%"

            cv2.putText(
                frame,
                texto,
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

    cv2.imshow("Tomato Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()