import pyautogui
import numpy as np
import time

# Post-match "OK" button detection pixels (screen-absolute, not region-relative)
OK_BUTTON_PIXEL_1 = (1026, 959)
OK_BUTTON_COLOR_1 = np.array([34, 37, 40])

OK_BUTTON_PIXEL_2 = (1012, 944)
OK_BUTTON_COLOR_2 = np.array([78, 175, 255])

TOLERANCE = 40  # Kept the tolerance relatively low because these pixels are always same color and same position, there isn't much variability

# Click sequence positions
OK_CLICK_POS = (1012, 944)         # Presses the OK button
CHEST_SKIP_CLICK_POS = (797, 690)  # Dismisses/skips chest-reward popups
CHEST_SKIP_CLICK_COUNT = 10
BATTLE_CLICK_POS = (1007, 801)     # Presses the "Battle" button to queue the next match

INTER_CLICK_DELAY = 0.3    # Short delay between clicks in the sequence
POST_BATTLE_WAIT = 3.0     # Wait after clicking Battle before the next match is assumed to have started


def color_match(pixel, target, tolerance):
    """Check if a pixel matches a target color within tolerance, return True if it does or False otherwise"""
    return np.all(np.abs(np.array(pixel) - target) <= tolerance)


def is_match_over():
    """
    Sample the two post-match OK-button pixels and return True only if both
    match their target colors within tolerance.
    """
    pixel_1 = pyautogui.pixel(*OK_BUTTON_PIXEL_1)
    pixel_2 = pyautogui.pixel(*OK_BUTTON_PIXEL_2)

    return bool(
        color_match(pixel_1, OK_BUTTON_COLOR_1, TOLERANCE)
        and color_match(pixel_2, OK_BUTTON_COLOR_2, TOLERANCE)
    )


def advance_to_next_battle():
    """
    Click through the post-match flow: press OK, dismiss chest popups,
    press Battle to queue the next match, then wait for it to start.
    """
    pyautogui.click(*OK_CLICK_POS)
    time.sleep(INTER_CLICK_DELAY)

    for _ in range(CHEST_SKIP_CLICK_COUNT):
        pyautogui.click(*CHEST_SKIP_CLICK_POS)
        time.sleep(INTER_CLICK_DELAY)

    pyautogui.click(*BATTLE_CLICK_POS)
    time.sleep(INTER_CLICK_DELAY)

    time.sleep(POST_BATTLE_WAIT)


# FOR TESTING / DEBUGGING

if __name__ == "__main__":
    while True:
        print(f"is_match_over(): {is_match_over()}")
        time.sleep(1)
