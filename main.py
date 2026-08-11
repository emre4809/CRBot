# =============================
# REWARD SYSTEM
# =============================

class CrownTracker:
    def __init__(self):
        self.prev_my_crowns = 0
        self.prev_enemy_crowns = 0

    def update(self):
        my_crowns, enemy_crowns = get_crown_counts()

        reward = 0

        # Reward for gaining crowns
        if my_crowns > self.prev_my_crowns:
            reward += (my_crowns - self.prev_my_crowns) * 50

        # Penalty for enemy crowns
        if enemy_crowns > self.prev_enemy_crowns:
            reward -= (enemy_crowns - self.prev_enemy_crowns) * 50

        self.prev_my_crowns = my_crowns
        self.prev_enemy_crowns = enemy_crowns

        return reward, my_crowns, enemy_crowns


# =============================
# TEST LOOP
# =============================

if __name__ == "__main__":
    import time

    tracker = CrownTracker()

    while True:
        reward, my_crowns, enemy_crowns = tracker.update()
        print(
            f"My crowns: {my_crowns}, "
            f"Enemy crowns: {enemy_crowns}, "
            f"Reward: {reward}"
        )
        time.sleep(1)