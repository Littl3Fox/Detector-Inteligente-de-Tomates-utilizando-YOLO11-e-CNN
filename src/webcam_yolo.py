from ultralytics import YOLO
import cv2


def main():

    # Carrega o modelo treinado
    model = YOLO(
        "runs/detect/runs/tomato_detector/weights/best.pt"
    )

    # Webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro ao abrir a webcam.")
        return

    print("Pressione 'q' para sair.")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Faz a detecção
        results = model(frame, conf=0.6)

        # Desenha as caixas automaticamente
        annotated_frame = results[0].plot()

        # Mostra o resultado
        cv2.imshow("YOLO Tomato Detector", annotated_frame)

        # Fecha com Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()