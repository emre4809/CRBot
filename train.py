# Training models for troop detection and deck classification using YOLOv11s
# Had to wrap around main() due to a windows error when using multiple workers
# Ran it on: Ultralytics 8.3.170  Python-3.12.0 torch-2.8.0+cu128 CUDA:0 (NVIDIA GeForce RTX 4060 Laptop GPU, 8188MiB)
from ultralytics import YOLO

# =====================
# CHANGE THIS VALUE
# 0 = detection
# 1 = classification
# =====================
RUN_CLASSIFICATION = 1


def main():
    if RUN_CLASSIFICATION == 1:
        print("Running CLASSIFICATION training")

        model = YOLO("yolo11s-cls.pt")
        model.train(
            data="dataset/deck",   # classification dataset root
            epochs=60,
            imgsz=224,
            device=0,
            workers=8
        )

    else:
        print("Running DETECTION training")

        model = YOLO("yolo11s.pt")
        model.train(
            data="dataset/troops/data.yaml",
            epochs=100,
            imgsz=640,
            device=0,
            workers=8
        )


if __name__ == "__main__":
    main()