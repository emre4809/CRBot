import pyautogui
import time

# Pixel coordinates (x, y) for each card slot in hand (left to right)
CARD_SLOTS = [
    (907, 900),   # slot 0
    (1008, 896),  # slot 1
    (1113, 896),  # slot 2
    (1215, 892),  # slot 3
]

# Friendly arena bounds — bot will only place cards within this region
# (x_min, y_min, x_max, y_max)
ARENA_BOUNDS = (779, 473, 1251, 774)


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


# test function to demonstrate card placement — will click each slot and place it at the arena center
if __name__ == "__main__":
    # Place each card slot at the center of the friendly arena with a 3s gap.
    arena_cx = (ARENA_BOUNDS[0] + ARENA_BOUNDS[2]) // 2
    arena_cy = (ARENA_BOUNDS[1] + ARENA_BOUNDS[3]) // 2

    print("Starting card placement test in 3 seconds — switch to BlueStacks now.")
    time.sleep(3)

    for i in range(4):
        print(f"Playing slot {i} -> arena center ({arena_cx}, {arena_cy})")
        result = play_card(i, arena_cx, arena_cy)
        print(f"  result: {result}")
        time.sleep(3)
