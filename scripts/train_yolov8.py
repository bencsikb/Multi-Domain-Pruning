from ultralytics import YOLO


if __name__ == "__main__":

    model = YOLO("yolov8x.pt")

    results = model.train(data="config/data/kitti.yaml", epochs=100, imgsz=640, batch=8)