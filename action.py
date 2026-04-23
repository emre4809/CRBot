import pyautogui
import time

# Pixel coordinates (x, y) for each card slot in hand (left to right)
CARD_SLOTS = [
    (0, 0),   # slot 0 — fill in
    (0, 0),   # slot 1 — fill in
    (0, 0),   # slot 2 — fill in
    (0, 0),   # slot 3 — fill in
]

# Friendly arena bounds — bot will only place cards within this region
# (x_min, y_min, x_max, y_max)
ARENA_BOUNDS = (0, 0, 0, 0)  # fill in: left, top, right, bottom of friendly side


def is_in_arena(x, y):
    """Return True if (x, y) is within the friendly arena bounds."""
    x_min, y_min, x_max, y_max = ARENA_BOUNDS
    return x_min <= x <= x_max and y_min <= y <= y_max


def play_card(slot_index, arena_x, arena_y):
    """
    Play a card by clicking its hand slot then clicking the arena position.
    Returns True if the action was taken, False if the target was out of bounds.

    slot_index: 0–3 (left to right in hand)
    arena_x, arena_y: target position in the arena (relative to screen)
    """
    if not (0 <= slot_index <= 3):
        print(f"Invalid slot index: {slot_index}")
        return False

    if not is_in_arena(arena_x, arena_y):
        print(f"Target ({arena_x}, {arena_y}) is outside friendly arena bounds — skipping.")
        return False

    card_x, card_y = CARD_SLOTS[slot_index]
    pyautogui.click(card_x, card_y)
    time.sleep(0.1)
    pyautogui.click(arena_x, arena_y)
    return True
