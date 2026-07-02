from ultralytics import YOLO

def main():

    model = YOLO("yolo11n.pt")

    results = model.train(
        data="dataset_yolo/data.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        workers=2,
        device="cpu",
        cache=True,
        project="runs",
        name="tomato_detector",
        exist_ok=True,
        verbose=True
    )

    print(results)

if __name__ == "__main__":
    main()