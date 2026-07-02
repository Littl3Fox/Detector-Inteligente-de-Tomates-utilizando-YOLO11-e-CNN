import os
import random
import shutil

# ======================
# CONFIGURAÇÃO
# ======================

random.seed(42)

IMAGE_DIR = "dataset_yolo/images"
LABEL_DIR = "dataset_yolo/labels"

TRAIN_RATIO = 0.8

# ======================

os.makedirs(os.path.join(IMAGE_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(IMAGE_DIR, "val"), exist_ok=True)

os.makedirs(os.path.join(LABEL_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(LABEL_DIR, "val"), exist_ok=True)

# procura imagens

images = []

for arquivo in os.listdir(IMAGE_DIR):
    if arquivo.lower().endswith((".jpg", ".jpeg", ".png", ".jfif")):
        images.append(arquivo)

random.shuffle(images)

train_size = int(len(images) * TRAIN_RATIO)

train = images[:train_size]
val = images[train_size:]

def mover(lista, pasta):

    for img in lista:

        nome = os.path.splitext(img)[0]

        origem_img = os.path.join(IMAGE_DIR, img)
        destino_img = os.path.join(IMAGE_DIR, pasta, img)

        shutil.move(origem_img, destino_img)

        txt = nome + ".txt"

        origem_label = os.path.join(LABEL_DIR, txt)

        if os.path.exists(origem_label):
            destino_label = os.path.join(LABEL_DIR, pasta, txt)
            shutil.move(origem_label, destino_label)

mover(train, "train")
mover(val, "val")

print("--------------------")
print("Treino:", len(train))
print("Validação:", len(val))
print("--------------------")