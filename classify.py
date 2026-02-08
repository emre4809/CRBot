import pyautogui
import numpy as np
import cv2
from ultralytics import YOLO
from card_data import elixir_costs
from elixir import get_current_elixir

# Load your trained model with best weights
model = YOLO("runs/classify/current_model/weights/best.pt")

HAND_REGION = (860, 850, 405, 120) # Update these according to your screen setup (x, y, width, height)
CARD_SPLITS = [0, 100, 200, 300, 405] # Splitting the hand region into 4 cards for classification


def capture_hand():
    
    """Captures the hand region, splits it into 4 card images, and returns them as a list"""
    
    screenshot = pyautogui.screenshot(region=HAND_REGION)
    img = np.array(screenshot)

    cards = []
    for i in range(len(CARD_SPLITS) - 1):
        x_start = CARD_SPLITS[i]
        x_end = CARD_SPLITS[i + 1]
        card_img = img[:, x_start:x_end]
        cards.append(card_img)

    return cards


def get_game_state():
    
    """Captures the current game state with elixir, hand cards, and playable cards based on elixir cost"""
    
    cards = capture_hand()
    hand = []

    for card in cards:
        # Convert RGB to BGR because OpenCV and YOLO expect BGR format
        card_bgr = cv2.cvtColor(card, cv2.COLOR_RGB2BGR)

        # Run classification
        results = model(card_bgr, verbose=False)

        class_index = results[0].probs.top1 # Get the index of the highest probability class
        class_name = results[0].names[class_index] # Get its card name

        # Get elixir cost from dictionary
        cost = elixir_costs[class_name]

        hand.append({
            "name": class_name,
            "cost": cost
        })

    # Get current elixir
    current_elixir = get_current_elixir()

    # Determine playable cards
    playable_cards = [
        card["name"]
        for card in hand
        if card["cost"] <= current_elixir
    ]

    # Final game state
    game_state = {
        "elixir": current_elixir,
        "hand": hand,
        "playable_cards": playable_cards
    }

    return game_state

# example game state: {'elixir': 3, 'hand': [{'name': 'musketeer', 'cost': 4}, {'name': 'arrows', 'cost': 3}, {'name': 'giant', 'cost': 5}, 
# {'name': 'knight', 'cost': 3}], 'playable_cards': ['arrows', 'knight']}

# Test loop
if __name__ == "__main__":
    import time
    while True:
        state = get_game_state()
        print(state)
        time.sleep(0.5)
