# Detection of troops in the game real time, code partially taken from Python mss documentation

import time
import cv2
import numpy as np
import mss
from ultralytics import YOLO

#Loading trained model, change path if needed
model = YOLO("runs/detect/current_model/weights/best.pt")

# Define the screen region to capture
monitor = {"top": 40, "left": 735, "width": 555, "height": 780}  # Update these according to your screen setup

with mss.mss() as sct:
    while True:
        start_time = time.time()

        # Capture the screen
        screenshot = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        # Run YOLO inference on the selected screen region and detect troops
        results = model.predict(source=frame, imgsz=640, conf=0.3, verbose=False)
        annotated = results[0].plot()  # Draw boxes and labels

        # Show the annotated frame
        cv2.imshow("Troop detection", annotated)

        # Print FPS
        fps = 1 / (time.time() - start_time)
        print(f"FPS: {fps:.2f}")

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

