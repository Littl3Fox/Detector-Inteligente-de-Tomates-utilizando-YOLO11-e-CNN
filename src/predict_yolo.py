from ultralytics import YOLO
import cv2


def main():

    # Modelo treinado
    model = YOLO(
        "runs/detect/runs/tomato_detector/weights/best.pt"
    )

    # Imagem para teste
    image_path = "imagens_teste/tomate_estragado_3.jpg"

    # Faz a detecção
    results = model(image_path, conf=0.4)

    # Pega a imagem com as caixas desenhadas
    annotated = results[0].plot()

    cv2.imshow("YOLO Detection", annotated)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()