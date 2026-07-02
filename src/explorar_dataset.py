from pathlib import Path

from PIL import Image
# Aqui busco ver quantas imagens tenho de cada categoria na pasta de treino
# Busco ver se o dataset de treino está balanceado
print("TREINO")
base = Path("dataset/train")

for pasta in base.iterdir():
    if pasta.is_dir():
        quantidade = len(list(pasta.glob("*")))
        print(f"{pasta.name}: {quantidade}")


print("VALIDACAO")
base = Path("dataset/val")

for pasta in base.iterdir():
    if pasta.is_dir():
        quantidade = len(list(pasta.glob("*")))
        print(f"{pasta.name}: {quantidade}")

#Vejo o tamanho da imagem
img = Image.open(next(Path("dataset/train/Ripe").glob("*")))

print(img.size)