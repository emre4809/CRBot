import time
import cv2
import numpy as np
import mss
from ultralytics import YOLO

model = YOLO("runs/detect/current_model/weights/best.pt")

monitor = {"top": 40, "left": 735, "width": 555, "height": 780}

_sct = mss.mss()


def get_troops():
    """Capture one frame and return detected troops with positions relative to the capture region."""
    screenshot = np.array(_sct.grab(monitor))
    frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
    results = model.predict(source=frame, imgsz=640, conf=0.3, verbose=False)

    troops = []
    if results[0].boxes is None:
        return troops

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0]
        troops.append({
            "troop": results[0].names[cls_id],
            "x": int((x1 + x2) / 2),
            "y": int((y1 + y2) / 2),
            "conf": float(box.conf[0]),
        })

    return troops


if __name__ == "__main__":
    while True:
        start_time = time.time()
        troops = get_troops()
        print(troops)
        print(f"FPS: {1 / (time.time() - start_time):.2f}")
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break