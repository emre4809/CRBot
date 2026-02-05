import pyautogui
import numpy as np

# Global variables for when elixir method is called
ELIXIR_REGION = (885, 985, 380, 20) # Change to your elixir bar region (left, top, width, height)

PURPLE_COLOR = np.array([208, 33, 214]) #RGB value of purple elixir bar and pink "Elixir bar is full" text, stored as numpy arrays
PINK_COLOR = np.array([255, 171, 255])
TOLERANCE = 80 # Allowed difference between pixel color and target color. Makes detection robust to minor color variations.
PINK_THRESHOLD = 2800 # If pink pixel count exceeds this number, the code assumes the "Elixir bar is full" message is visible.

# Midpoint thresholds for elixir levels based on purple pixel counts
THRESHOLDS = [
    223,   # If number of purple pixels is less than this, elixir = 0
    544,   # If greater than previous and less than this, elixir = 1
    975,   # 2
    1502,  # 3
    2009,  # 4
    2493,  # 5
    3012,  # 6
    3513,  # 7
    4004,  # 8
    4423   # 9
]

def get_current_elixir():
    screenshot = pyautogui.screenshot(region=ELIXIR_REGION) # Captures elixir bar region
    img = np.array(screenshot) # Converts the screenshot into a NumPy array with shape: (height, width, 3), ex: (20 rows, 380 columns, 3 color channels)

    # Purple detection
    diff_purple = np.abs(img - PURPLE_COLOR) # Subtracts each pixel from the target purple color and takes absolute value. Result: how far each pixel is from purple.
    mask_purple = np.all(diff_purple <= TOLERANCE, axis=2) # Checks if each pixel is within tolerance in R, G, and B channels. Produces a boolean mask: [True, True, False, ...]
    purple_pixels = np.sum(mask_purple) # Counts how many pixels are True (AKA purple).

    # Pink detection
    diff_pink = np.abs(img - PINK_COLOR) # Same process for pink color which indicates "Elixir bar is full" message
    mask_pink = np.all(diff_pink <= TOLERANCE, axis=2)
    pink_pixels = np.sum(mask_pink)

    # print(f"Purple pixels: {purple_pixels}, Pink pixels: {pink_pixels}")

    # Special case: if there are a lot of pink pixels ("Elixir bar is full") or purple pixels exceed the last threshold (which happens when elixir is full), return 10 elixir
    if pink_pixels > PINK_THRESHOLD or purple_pixels > THRESHOLDS[-1]:
        return 10

    # Normal threshold logic
    for i, threshold in enumerate(THRESHOLDS):
        if purple_pixels < threshold:
            return i

    return 10

# You can run this file directly to test elixir detection, CTRL+C to break the loop.
if __name__ == "__main__":
    import time
    while True:
        print(get_current_elixir())
        # print("Elixir:", test_elixir_capture())
        time.sleep(0.5)