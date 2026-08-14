import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from detect import get_troops, model as detect_model, monitor as detect_monitor
from classify import get_game_state
from crowns import get_crown_counts
from action import play_card, ARENA_BOUNDS
from tracker import TroopTracker
from reward import compute_reward
from match_end import is_match_over, advance_to_next_battle

GRID_COLS = 6
GRID_ROWS = 4
N_GRID = GRID_COLS * GRID_ROWS   # 24
N_ACTIONS = 4 * N_GRID + 1       # 97
STEP_INTERVAL = 0.5               # seconds between steps
MAX_TROOPS = 20
OBS_SIZE = 87                     # 1 elixir + 1 my_crowns + 1 enemy_crowns + 4 hand + 20*4 troops


class ClashRoyaleEnv(gym.Env):
    metadata = {}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._troop_id = {name: idx for idx, name in detect_model.names.items()}
        self._n_classes = max(len(self._troop_id), 1)
        self._region_w = detect_monitor["width"]
        self._region_h = detect_monitor["height"]

        x_min, y_min, x_max, y_max = ARENA_BOUNDS
        xs = np.linspace(x_min, x_max, GRID_COLS)
        ys = np.linspace(y_min, y_max, GRID_ROWS)
        self._grid = [(int(x), int(y)) for y in ys for x in xs]

        self._tracker = TroopTracker()
        self._prev_my_crowns = 0
        self._prev_enemy_crowns = 0
        self._last_state = {
            "elixir": 0,
            "hand": [{"name": None, "cost": 0} for _ in range(4)],
            "playable_cards": [],
        }

    def _safe_get_game_state(self):
        try:
            state = get_game_state()
        except Exception as e:
            print(f"[env] get_game_state() failed, reusing last known state: {e}")
            return self._last_state
        self._last_state = state
        return state

    def _safe_get_troops(self):
        try:
            return get_troops()
        except Exception as e:
            print(f"[env] get_troops() failed, assuming no troops detected: {e}")
            return []

    def _safe_get_crown_counts(self):
        try:
            return get_crown_counts()
        except Exception as e:
            print(f"[env] get_crown_counts() failed, reusing last known crowns: {e}")
            return self._prev_my_crowns, self._prev_enemy_crowns

    def _safe_play_card(self, card_slot, px, py):
        try:
            play_card(card_slot, px, py)
        except Exception as e:
            print(f"[env] play_card() failed, skipping action: {e}")

    def _safe_is_match_over(self):
        try:
            return is_match_over()
        except Exception as e:
            print(f"[env] is_match_over() failed, assuming match still in progress: {e}")
            return False

    def _safe_advance_to_next_battle(self):
        try:
            advance_to_next_battle()
        except Exception as e:
            print(f"[env] advance_to_next_battle() failed: {e}")
            return

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._safe_is_match_over():
            self._safe_advance_to_next_battle()
        self._tracker.clear()
        state = self._safe_get_game_state()
        my_crowns, enemy_crowns = self._safe_get_crown_counts()
        self._prev_my_crowns = my_crowns
        self._prev_enemy_crowns = enemy_crowns
        obs = self._build_obs(state, my_crowns, enemy_crowns)
        return obs, {}

    def step(self, action):
        if action < N_ACTIONS - 1:
            card_slot = action // N_GRID
            cell = action % N_GRID
            px, py = self._grid[cell]
            state = self._safe_get_game_state()
            card_name = state["hand"][card_slot]["name"]
            if card_name in state["playable_cards"]:
                self._safe_play_card(card_slot, px, py)

        time.sleep(STEP_INTERVAL)

        state = self._safe_get_game_state()
        detections = self._safe_get_troops()
        my_crowns, enemy_crowns = self._safe_get_crown_counts()

        kills = self._tracker.update(detections, my_crowns, enemy_crowns)

        terminated = self._safe_is_match_over()
        won = my_crowns > enemy_crowns

        reward = compute_reward(
            self._prev_my_crowns, self._prev_enemy_crowns,
            my_crowns, enemy_crowns, kills, terminated, won,
        )

        self._prev_my_crowns = my_crowns
        self._prev_enemy_crowns = enemy_crowns

        obs = self._build_obs(state, my_crowns, enemy_crowns)
        return obs, float(reward), terminated, False, {}

    def _build_obs(self, state, my_crowns, enemy_crowns):
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        obs[0] = state["elixir"] / 10.0
        obs[1] = my_crowns / 3.0
        obs[2] = enemy_crowns / 3.0
        for i, card in enumerate(state["hand"][:4]):
            obs[3 + i] = card["cost"] / 9.0
        for i, t in enumerate(self._tracker.troops[:MAX_TROOPS]):
            base = 7 + i * 4
            obs[base]     = t["x"] / self._region_w
            obs[base + 1] = t["y"] / self._region_h
            obs[base + 2] = 1.0 if t["team"] == "enemy" else 0.0
            obs[base + 3] = self._troop_id.get(t["troop"], 0) / self._n_classes
        return obs
