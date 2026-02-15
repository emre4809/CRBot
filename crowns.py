import pyautogui
import numpy as np

# Crown slot regions (x, y, width, height)
MY_CROWN_REGION = (906, 536, 203, 64)
ENEMY_CROWN_REGION = (905, 296, 209, 69)

# These pixel positions are all relative to the screenshot region defined above
MY_CROWN_PIXELS = [
    (36, 27),    # slot1 top
    (26, 38),    # slot1 bottom
    (96, 27),    # slot2 top
    (86, 38),    # slot2 bottom
    (156, 27),   # slot3 top
    (146, 38)    # slot3 bottom
]

ENEMY_CROWN_PIXELS = [
    (37, 29),
    (27, 40),
    (97, 29),
    (87, 40),
    (157, 29),
    (147, 40)
]

# Color definitions (RGB). Friendly crowns are blue, enemy crowns are red. Both have yellow tops. Gray is defined for empty slots.
YELLOW = np.array([254, 190, 100])
BLUE   = np.array([34, 164, 253])
RED    = np.array([240, 26, 113])
GRAY   = np.array([53, 43, 22])

TOLERANCE = 40 # Kept the tolerance relatively low because crown slots are always same color and same position, there isn't much variability


# Helper functions

def color_match(pixel, target):
    """Check if a pixel matches a target color within tolerance, return True if it does or False otherwise"""
    return np.all(np.abs(pixel - target) <= TOLERANCE)


def capture_region(region):
    """Capture a region once and return numpy image"""
    img = pyautogui.screenshot(region=region)
    return np.array(img)


def get_pixel_from_image(img, pos):
    """Get pixel from already captured image"""
    return img[pos[1], pos[0]]


# CROWN DETECTION

def detect_crowns_from_image(img, pixel_positions, friendly=True):
    """Determines crown count using slot logic based on pixel colors"""
    bottom_color = BLUE if friendly else RED # if friendly = True use BLUE when checking bottom pixel, otherwise use RED

    pixels = [get_pixel_from_image(img, p) for p in pixel_positions] # Loop over all 6 positions, read each pixel (ex. [143 177 67]), store them in a list

    # Helper checks
    def slot_colored(top, bottom):
        return color_match(top, YELLOW) and color_match(bottom, bottom_color)

    def slot_gray(top, bottom):
        return color_match(top, GRAY) and color_match(bottom, GRAY)

    # Slot 1
    s1 = slot_colored(pixels[0], pixels[1])
    # No need for g1 as we will never have to first crown slot grayed out

    # Slot 2
    s2 = slot_colored(pixels[2], pixels[3])
    g2 = slot_gray(pixels[2], pixels[3])

    # Slot 3
    s3 = slot_colored(pixels[4], pixels[5])
    g3 = slot_gray(pixels[4], pixels[5])

    # Crown logic
    # If slot 1 is colored (pixels 0 and 1) and slot 2 is gray (pixels 2 and 3) and slot 3 is gray (pixels 4 and 5), then there's 1 crown
    # If slot 1 is colored and slot 2 is colored and slot 3 is gray, then there's 2 crowns, etc.
    if s1 and g2 and g3:
        return 1
    elif s1 and s2 and g3:
        return 2
    elif s1 and s2 and s3:
        return 3
    else:
        return 0


def get_crown_counts():
    """Capture both friendly and enemy regions and return crown counts"""
    my_img = capture_region(MY_CROWN_REGION)
    enemy_img = capture_region(ENEMY_CROWN_REGION)

    my_crowns = detect_crowns_from_image(my_img, MY_CROWN_PIXELS, friendly=True) # friendly=True means we will check for blue bottom pixels
    enemy_crowns = detect_crowns_from_image(enemy_img, ENEMY_CROWN_PIXELS, friendly=False) # false so we check for red bottom pixels

    return my_crowns, enemy_crowns



# FOR TESTING / DEBUGGING

def test_crown_detection():
    """
    Debug tool to verify:
    - Correct screenshot regions
    - Correct pixel positions
    - Detected crown counts
    """

    print("=== Testing Crown Detection ===")

    # Capture regions
    my_img = capture_region(MY_CROWN_REGION)
    enemy_img = capture_region(ENEMY_CROWN_REGION)

    # Print pixel colors for friendly crowns
    print("\nMy crown pixels:")
    for i, pos in enumerate(MY_CROWN_PIXELS):
        pixel = get_pixel_from_image(my_img, pos)
        print(f"Pixel {i+1} at {pos}: {pixel}")

    # Print pixel colors for enemy crowns
    print("\nEnemy crown pixels:")
    for i, pos in enumerate(ENEMY_CROWN_PIXELS):
        pixel = get_pixel_from_image(enemy_img, pos)
        print(f"Pixel {i+1} at {pos}: {pixel}")

    # Detect crowns
    my_crowns = detect_crowns_from_image(my_img, MY_CROWN_PIXELS, friendly=True)
    enemy_crowns = detect_crowns_from_image(enemy_img, ENEMY_CROWN_PIXELS, friendly=False)

    print("\nDetected crowns:")
    print(f"My crowns: {my_crowns}")
    print(f"Enemy crowns: {enemy_crowns}")

    print("=== Test complete ===")


# Run test when file is executed directly, take screenshots every 0.5 seconds
if __name__ == "__main__":
    import time
    while True:
        test_crown_detection()
        time.sleep(1)