CROWN_REWARD = 8.0    # reward per friendly crown gained
CROWN_PENALTY = 8.0   # reward subtracted per enemy crown gained
KILL_REWARD = 0.1     # reward per enemy troop killed
WIN_REWARD = 15.0     # reward bonus for winning the match
LOSS_PENALTY = 15.0   # reward penalty for losing (or drawing) the match


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
    reward += (my_crowns - prev_my_crowns) * CROWN_REWARD
    reward -= (enemy_crowns - prev_enemy_crowns) * CROWN_PENALTY
    reward += len(kills) * KILL_REWARD
    if terminated:
        reward += WIN_REWARD if won else -LOSS_PENALTY
    return reward