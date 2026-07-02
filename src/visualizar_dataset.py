from pathlib import Path
import cv2

# Aqui vizualizo algumas imagens de cada classe

classe = "Damaged"
#classe = "Old"
#classe = "Ripe"
#classe = "Unripe"

pasta = Path(f"dataset/train/{classe}")

imagem = next(pasta.glob("*"))

img = cv2.imread(str(imagem))

print("Arquivo:", imagem.name)
print("Dimensões:", img.shape)

cv2.imshow(classe, img)

cv2.waitKey(0)
cv2.destroyAllWindows()