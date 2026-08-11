import time
from classify import get_game_state
from crowns import get_crown_counts
from action import ARENA_BOUNDS, play_card

LOOP_INTERVAL = 1.0
PLAY_COOLDOWN = 2.0

ARENA_CX = (ARENA_BOUNDS[0] + ARENA_BOUNDS[2]) // 2
ARENA_CY = (ARENA_BOUNDS[1] + ARENA_BOUNDS[3]) // 2


def pick_card(hand, elixir):
    """Return slot index of the cheapest affordable card, or None."""
    best_slot = None
    best_cost = float("inf")
    for i, card in enumerate(hand):
        if card["cost"] <= elixir and card["cost"] < best_cost:
            best_cost = card["cost"]
            best_slot = i
    return best_slot


def run():
    prev_my_crowns = 0
    prev_enemy_crowns = 0
    last_play_time = 0
    print("Agent started. Press Ctrl+C to stop.")

    while True:
        try:
            state = get_game_state()
            my_crowns, enemy_crowns = get_crown_counts()

            # Never let crown counts go down — detection can flicker to 0
            my_crowns = max(my_crowns, prev_my_crowns)
            enemy_crowns = max(enemy_crowns, prev_enemy_crowns)

            if my_crowns > prev_my_crowns:
                gained = (my_crowns - prev_my_crowns) * 50
                print(f"*** CROWN GAINED! +{gained} ***")
            if enemy_crowns > prev_enemy_crowns:
                lost = (enemy_crowns - prev_enemy_crowns) * 50
                print(f"*** ENEMY CROWN! -{lost} ***")
            prev_my_crowns = my_crowns
            prev_enemy_crowns = enemy_crowns

            now = time.time()
            crown_str = f"crowns(me={my_crowns} enemy={enemy_crowns})"

            if now - last_play_time >= PLAY_COOLDOWN:
                slot = pick_card(state["hand"], state["elixir"])
                if slot is not None:
                    card_name = state["hand"][slot]["name"]
                    print(f"Playing slot {slot} ({card_name}) | elixir={state['elixir']} | {crown_str}")
                    play_card(slot, ARENA_CX, ARENA_CY)
                    last_play_time = now
                else:
                    print(f"Waiting for elixir | elixir={state['elixir']} | {crown_str}")
            else:
                print(f"Cooldown | elixir={state['elixir']} | {crown_str}")

        except Exception as e:
            print(f"Error: {e}")

        time.sleep(LOOP_INTERVAL)


if __name__ == "__main__":
    run()
