def compute_reward(prev_my_crowns, prev_enemy_crowns, my_crowns, enemy_crowns, kills, terminated, won):
    """
    Compute the scalar reward for one environment step.

    prev_my_crowns / prev_enemy_crowns: crown counts from the previous step
    my_crowns / enemy_crowns: crown counts from the current step
    kills: list of killed enemy troop dicts (already gated by tracker)
    terminated: True if the episode ended this step
    won: True if we won (only meaningful when terminated=True)
    """
    reward = 0.0
    reward += (my_crowns - prev_my_crowns) * 8.0
    reward -= (enemy_crowns - prev_enemy_crowns) * 8.0
    reward += len(kills) * 0.1
    if terminated:
        reward += 15.0 if won else -15.0
    return reward
