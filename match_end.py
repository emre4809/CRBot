import time

import pyautogui

# Template images for the two buttons we care about after a match ends.
BATTLE_BUTTON_IMAGE = "images/Battle button.png"
OK_BUTTON_IMAGE = "images/OK button.png"

MATCH_CONFIDENCE = 0.8  # Fuzzy template-match confidence threshold; tune if too strict/loose.

# Battle/OK buttons only ever appear within this region of the screen, so
# scoping locateOnScreen() to it (instead of the full screen) speeds up
# every template match considerably.
# Region given as (left, top, width, height); calibrated from corners
# (738, 740) top-left and (1279, 1008) bottom-right.
SEARCH_REGION = (738, 740, 1279 - 738, 1008 - 740)

# Click sequence positions / timings
CHEST_SKIP_CLICK_POS = (1016, 891)  # Dismisses/skips chest-reward and reward-ladder popups
INTER_CLICK_DELAY = 0.3            # Delay between chest-skip clicks in the sequence
CHECK_INTERVAL = 1.0               # Minimum time between Battle/OK visibility checks

MAX_ADVANCE_SECONDS = 120  # Safety cap: give up waiting for the Battle button after this long


def is_battle_button_visible():
    """Return True if the Battle button template matches within SEARCH_REGION."""
    try:
        box = pyautogui.locateOnScreen(BATTLE_BUTTON_IMAGE, confidence=MATCH_CONFIDENCE, region=SEARCH_REGION)
    except pyautogui.ImageNotFoundException:
        return False
    return box is not None


def is_ok_button_visible():
    """Return True if the OK button template matches within SEARCH_REGION."""
    try:
        box = pyautogui.locateOnScreen(OK_BUTTON_IMAGE, confidence=MATCH_CONFIDENCE, region=SEARCH_REGION)
    except pyautogui.ImageNotFoundException:
        return False
    return box is not None


def click_battle_button():
    """Locate and click the Battle button. Return True if it was found and clicked."""
    try:
        box = pyautogui.locateOnScreen(BATTLE_BUTTON_IMAGE, confidence=MATCH_CONFIDENCE, region=SEARCH_REGION)
    except pyautogui.ImageNotFoundException:
        return False
    if box is None:
        return False
    pyautogui.click(pyautogui.center(box))
    return True


def click_ok_button():
    """Locate and click the OK button. Return True if it was found and clicked."""
    try:
        box = pyautogui.locateOnScreen(OK_BUTTON_IMAGE, confidence=MATCH_CONFIDENCE, region=SEARCH_REGION)
    except pyautogui.ImageNotFoundException:
        return False
    if box is None:
        return False
    pyautogui.click(pyautogui.center(box))
    return True


def is_match_over():
    """
    True when the post-match OK button is visible on screen.
    """
    return is_ok_button_visible()


def advance_to_next_battle():
    """
    Click through the post-match flow: press OK, then repeatedly click the
    chest-skip position (clicking OK again if a second prompt — e.g. the
    reward ladder — shows up) until the Battle button appears, then click it
    to queue the next match.

    Chest-skip clicks happen every INTER_CLICK_DELAY (cheap — just a click,
    no screen read), but the Battle/OK visibility checks (each a template
    match over SEARCH_REGION) only run once every CHECK_INTERVAL, since they
    don't need to run as often as the clicks do and are the expensive part.
    """
    click_ok_button()
    time.sleep(INTER_CLICK_DELAY)

    start_time = time.time()
    last_check_time = 0.0
    while True:
        now = time.time()

        if now - start_time > MAX_ADVANCE_SECONDS:
            print(
                f"[match_end] WARNING: Battle button not detected after "
                f"{int(now - start_time)}s of clicking — giving up, "
                f"reset() will proceed anyway. Check template images / screen state."
            )
            return

        if now - last_check_time >= CHECK_INTERVAL:
            last_check_time = now

            if is_battle_button_visible():
                click_battle_button()
                return

            if is_ok_button_visible():
                click_ok_button()

        pyautogui.click(*CHEST_SKIP_CLICK_POS)
        time.sleep(INTER_CLICK_DELAY)


# FOR TESTING / DEBUGGING

if __name__ == "__main__":
    while True:
        print(
            f"is_battle_button_visible(): {is_battle_button_visible()}  "
            f"is_ok_button_visible(): {is_ok_button_visible()}"
        )
        time.sleep(1)