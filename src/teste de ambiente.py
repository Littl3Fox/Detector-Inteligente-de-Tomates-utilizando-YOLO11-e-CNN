import cv2
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import torch


print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-Learn:", sklearn.__version__)
print("Matplotlib:", matplotlib.__version__)
print(torch.__version__)
print("CUDA:", torch.cuda.is_available()) # Não tenho placa de vídeo dedicada ;-;

print("Ambiente configurado com sucesso!")