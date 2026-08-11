# RL Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Gymnasium-compatible PPO reinforcement learning environment for the Clash Royale bot, including troop tracking, reward computation, and a Stable Baselines 3 training loop.

**Architecture:** Four new files handle all RL logic — `tracker.py` classifies ally/enemy troops by Y-position and detects deaths, `reward.py` computes step rewards from crown deltas and kill events, `env.py` wraps all sensors into a Gymnasium interface, and `agent.py` runs the PPO training loop. All existing sensor files are used unchanged.

**Tech Stack:** Python 3.12, PyTorch (already installed via Ultralytics), Stable Baselines 3, Gymnasium

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `tracker.py` | Create | Track troops frame-to-frame, classify ally/enemy by Y-position, detect deaths |
| `reward.py` | Create | Compute step reward from crown deltas and kill events |
| `env.py` | Create | Gymnasium environment — 87-value obs vector, Box(3) action space, step(), reset() |
| `agent.py` | Create | PPO training loop with checkpointing via Stable Baselines 3 |
| `tests/test_tracker.py` | Create | Unit tests for TroopTracker |
| `tests/test_reward.py` | Create | Unit tests for compute_reward |
| `tests/test_env.py` | Create | Unit tests for ClashRoyaleEnv with mocked sensors |
| `detect.py` | Read-only | Troop detection sensor |
| `crowns.py` | Read-only | Crown count sensor |
| `classify.py` | Read-only | Game state sensor (elixir + hand + playable cards) |
| `action.py` | Read-only | Card play actuator |
| `card_data.py` | Read-only | Elixir cost lookup |

---

## Task 1: Troop Tracker

**Files:**
- Create: `tracker.py`
- Create: `tests/test_tracker.py`

- [ ] **Step 1: Install dependencies**

```bash
pip install gymnasium stable-baselines3
```

Expected output: packages installed successfully with no errors.

- [ ] **Step 2: Write the failing tests**

Create `tests/__init__.py` (empty file), then create `tests/test_tracker.py`:

```python
import pytest
from tracker import TroopTracker, ARENA_MIDLINE_Y, PROXIMITY_THRESHOLD


def _det(troop, x, y):
    return {"troop": troop, "x": x, "y": y, "conf": 0.9}


def test_new_troop_above_midline_classified_as_enemy():
    tracker = TroopTracker()
    tracker.update([_det("goblins", 200, ARENA_MIDLINE_Y - 50)], my_crowns=0, enemy_crowns=0)
    assert len(tracker.troops) == 1
    assert tracker.troops[0].team == "enemy"


def test_new_troop_below_midline_classified_as_ally():
    tracker = TroopTracker()
    tracker.update([_det("knight", 200, ARENA_MIDLINE_Y + 50)], my_crowns=0, enemy_crowns=0)
    assert tracker.troops[0].team == "ally"


def test_new_troop_not_tracked_when_my_crown_lost():
    tracker = TroopTracker()
    kills = tracker.update([_det("giant", 200, ARENA_MIDLINE_Y - 50)], my_crowns=1, enemy_crowns=0)
    assert len(tracker.troops) == 0
    assert kills == []


def test_new_troop_not_tracked_when_enemy_crown_lost():
    tracker = TroopTracker()
    tracker.update([_det("giant", 200, ARENA_MIDLINE_Y - 50)], my_crowns=0, enemy_crowns=1)
    assert len(tracker.troops) == 0


def test_existing_troop_matched_by_proximity_and_position_updated():
    tracker = TroopTracker()
    tracker.update([_det("goblins", 100, ARENA_MIDLINE_Y - 10)], 0, 0)
    tracker.update([_det("goblins", 105, ARENA_MIDLINE_Y - 15)], 0, 0)
    assert len(tracker.troops) == 1
    assert tracker.troops[0].x == 105
    assert tracker.troops[0].y == ARENA_MIDLINE_Y - 15


def test_troop_beyond_proximity_threshold_creates_new_entry():
    tracker = TroopTracker()
    tracker.update([_det("goblins", 100, ARENA_MIDLINE_Y - 10)], 0, 0)
    tracker.update([_det("goblins", 100 + PROXIMITY_THRESHOLD + 1, ARENA_MIDLINE_Y - 10)], 0, 0)
    assert len(tracker.troops) == 2


def test_enemy_death_detected_after_3_frames_missing():
    tracker = TroopTracker()
    tracker.update([_det("goblins", 200, ARENA_MIDLINE_Y - 50)], 0, 0)
    kills1 = tracker.update([], 0, 0)
    kills2 = tracker.update([], 0, 0)
    kills3 = tracker.update([], 0, 0)
    assert kills1 == []
    assert kills2 == []
    assert kills3 == ["goblins"]


def test_ally_disappearance_does_not_trigger_kill():
    tracker = TroopTracker()
    tracker.update([_det("knight", 200, ARENA_MIDLINE_Y + 50)], 0, 0)
    kills = []
    for _ in range(3):
        kills = tracker.update([], 0, 0)
    assert kills == []


def test_reset_clears_all_troops():
    tracker = TroopTracker()
    tracker.update([_det("goblins", 200, ARENA_MIDLINE_Y - 50)], 0, 0)
    tracker.reset()
    assert tracker.troops == []
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
pytest tests/test_tracker.py -v
```

Expected: `ModuleNotFoundError: No module named 'tracker'` — confirms tests are wired correctly.

- [ ] **Step 4: Implement `tracker.py`**

```python
import math
from dataclasses import dataclass, field
from typing import List

# Y-coordinate of the arena midline in capture-region pixels.
# Troops above this line are classified as enemy, below as ally.
# Confirm this value by running detect.py and inspecting troop positions
# on both sides before first training run. Approximate: monitor["height"] / 2.
ARENA_MIDLINE_Y: int = 390

# Maximum pixel distance to match a detection to an existing tracked troop.
PROXIMITY_THRESHOLD: int = 50

# Frames a troop must be absent before being declared dead.
DEATH_FRAME_THRESHOLD: int = 3


@dataclass
class TrackedTroop:
    id: int
    troop_type: str
    x: float
    y: float
    team: str  # "ally" or "enemy"
    frames_since_seen: int = 0


class TroopTracker:
    def __init__(self) -> None:
        self._troops: List[TrackedTroop] = []
        self._next_id: int = 0

    def reset(self) -> None:
        """Clear all tracked troops. Call at the start of each episode."""
        self._troops = []
        self._next_id = 0

    def update(
        self,
        detections: list,
        my_crowns: int,
        enemy_crowns: int,
    ) -> List[str]:
        """
        Reconcile new detections with tracked troops.

        Args:
            detections: list of dicts with keys 'troop', 'x', 'y', 'conf'
            my_crowns: crowns I have earned this episode
            enemy_crowns: crowns enemy has earned this episode

        Returns:
            List of troop type strings for enemy troops that died this step.
        """
        matched_indices: set = set()

        # Match detections to existing troops by proximity
        for troop in self._troops:
            best_dist = float("inf")
            best_idx = -1
            for i, det in enumerate(detections):
                if i in matched_indices:
                    continue
                dist = math.hypot(det["x"] - troop.x, det["y"] - troop.y)
                if dist < best_dist and dist <= PROXIMITY_THRESHOLD:
                    best_dist = dist
                    best_idx = i
            if best_idx >= 0:
                troop.x = detections[best_idx]["x"]
                troop.y = detections[best_idx]["y"]
                troop.frames_since_seen = 0
                matched_indices.add(best_idx)
            else:
                troop.frames_since_seen += 1

        # Detect deaths: enemy troops absent for >= DEATH_FRAME_THRESHOLD frames
        kill_events: List[str] = []
        surviving: List[TrackedTroop] = []
        for troop in self._troops:
            if troop.frames_since_seen >= DEATH_FRAME_THRESHOLD:
                if troop.team == "enemy":
                    kill_events.append(troop.troop_type)
            else:
                surviving.append(troop)
        self._troops = surviving

        # Classify and add new (unmatched) detections
        crowns_clean = (my_crowns == 0 and enemy_crowns == 0)
        for i, det in enumerate(detections):
            if i not in matched_indices and crowns_clean:
                team = "enemy" if det["y"] < ARENA_MIDLINE_Y else "ally"
                self._troops.append(
                    TrackedTroop(
                        id=self._next_id,
                        troop_type=det["troop"],
                        x=det["x"],
                        y=det["y"],
                        team=team,
                    )
                )
                self._next_id += 1

        return kill_events

    @property
    def troops(self) -> List[TrackedTroop]:
        return list(self._troops)
```

- [ ] **Step 5: Run tests and confirm they pass**

```bash
pytest tests/test_tracker.py -v
```

Expected: all 9 tests pass.

- [ ] **Step 6: Commit**

```bash
git add tracker.py tests/__init__.py tests/test_tracker.py
git commit -m "feat: add troop tracker with ally/enemy Y-position classification"
```

---

## Task 2: Reward Calculator

**Files:**
- Create: `reward.py`
- Create: `tests/test_reward.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reward.py`:

```python
import pytest
from reward import compute_reward


def test_no_event_gives_zero_reward():
    assert compute_reward(0, 0, 0, 0, 0, False, False) == 0.0


def test_i_earn_a_crown():
    # my_crowns goes from 0 to 1
    assert compute_reward(0, 0, 1, 0, 0, False, False) == 8.0


def test_enemy_earns_a_crown():
    # enemy_crowns goes from 0 to 1
    assert compute_reward(0, 0, 0, 1, 0, False, False) == -8.0


def test_kill_reward_when_no_crowns_destroyed():
    r = compute_reward(0, 0, 0, 0, 3, False, False)
    assert abs(r - 0.3) < 1e-6


def test_kill_reward_suppressed_when_my_crown_was_already_lost():
    # prev_my_crowns=1 means a tower was destroyed in a prior step
    r = compute_reward(1, 0, 1, 0, 3, False, False)
    assert r == 0.0


def test_kill_reward_suppressed_when_enemy_crown_was_already_lost():
    r = compute_reward(0, 1, 0, 1, 3, False, False)
    assert r == 0.0


def test_win_adds_bonus_on_top_of_crown_reward():
    # I earn the 3rd crown (my_crowns 2→3) and win
    r = compute_reward(2, 0, 3, 0, 0, True, True)
    assert abs(r - (8.0 + 15.0)) < 1e-6


def test_loss_adds_penalty_on_top_of_crown_penalty():
    # Enemy earns the 3rd crown (enemy_crowns 2→3) and I lose
    r = compute_reward(0, 2, 0, 3, 0, True, False)
    assert abs(r - (-8.0 + -15.0)) < 1e-6


def test_two_crowns_in_one_step():
    # Unlikely but handled: I earn 2 crowns at once
    r = compute_reward(0, 0, 2, 0, 0, False, False)
    assert r == 16.0
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_reward.py -v
```

Expected: `ModuleNotFoundError: No module named 'reward'`

- [ ] **Step 3: Implement `reward.py`**

```python
KILL_REWARD: float = 0.1
CROWN_REWARD: float = 8.0
CROWN_PENALTY: float = -8.0
WIN_REWARD: float = 15.0
LOSS_PENALTY: float = -15.0


def compute_reward(
    prev_my_crowns: int,
    prev_enemy_crowns: int,
    curr_my_crowns: int,
    curr_enemy_crowns: int,
    kill_count: int,
    done: bool,
    won: bool,
) -> float:
    """
    Compute the reward for a single environment step.

    Args:
        prev_my_crowns: my crown count at the previous step
        prev_enemy_crowns: enemy crown count at the previous step
        curr_my_crowns: my crown count after this step
        curr_enemy_crowns: enemy crown count after this step
        kill_count: number of enemy troops that died this step
        done: True if this step ends the episode
        won: True if I won (only meaningful when done=True)

    Returns:
        Scalar reward for this step.
    """
    reward = 0.0

    # Crown deltas
    reward += (curr_my_crowns - prev_my_crowns) * CROWN_REWARD
    reward += (curr_enemy_crowns - prev_enemy_crowns) * CROWN_PENALTY

    # Kill reward — only active while no tower has been destroyed
    if prev_my_crowns == 0 and prev_enemy_crowns == 0:
        reward += kill_count * KILL_REWARD

    # Episode-end bonus/penalty
    if done:
        reward += WIN_REWARD if won else LOSS_PENALTY

    return reward
```

- [ ] **Step 4: Run tests and confirm they pass**

```bash
pytest tests/test_reward.py -v
```

Expected: all 9 tests pass.

- [ ] **Step 5: Commit**

```bash
git add reward.py tests/test_reward.py
git commit -m "feat: add reward calculator with crown delta and kill reward logic"
```

---

## Task 3: RL Environment

**Files:**
- Create: `env.py`
- Create: `tests/test_env.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_env.py`:

```python
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock

# Mock all external sensor/actuator modules before importing env,
# so tests never touch the screen or YOLO models.
_mock_classify = MagicMock()
_mock_crowns = MagicMock()
_mock_detect = MagicMock()
_mock_action = MagicMock()
_mock_detect.monitor = {"width": 555, "height": 780}
_mock_action.ARENA_BOUNDS = (0, 390, 555, 780)

sys.modules["classify"] = _mock_classify
sys.modules["crowns"] = _mock_crowns
sys.modules["detect"] = _mock_detect
sys.modules["action"] = _mock_action

from env import ClashRoyaleEnv, OBS_SIZE  # noqa: E402

MOCK_TROOP_MAP = {"goblins": 0, "giant": 1, "knight": 2, "arrows": 3}

MOCK_GAME_STATE = {
    "elixir": 5,
    "hand": [
        {"name": "goblins", "cost": 2},
        {"name": "giant", "cost": 5},
        {"name": "knight", "cost": 3},
        {"name": "arrows", "cost": 3},
    ],
    "playable_cards": ["goblins", "knight", "arrows"],
}


def make_env():
    _mock_classify.get_game_state.return_value = MOCK_GAME_STATE
    _mock_crowns.get_crown_counts.return_value = (0, 0)
    _mock_detect.get_troops.return_value = []
    # step_interval=0.0 skips the sleep so tests run instantly
    return ClashRoyaleEnv(troop_name_to_id=MOCK_TROOP_MAP, step_interval=0.0)


def test_observation_shape_on_reset():
    env = make_env()
    obs, info = env.reset()
    assert obs.shape == (OBS_SIZE,)
    assert obs.dtype == np.float32


def test_observation_values_on_reset():
    env = make_env()
    obs, _ = env.reset()
    # elixir = 5 → 5/10 = 0.5
    assert abs(obs[0] - 0.5) < 1e-5
    # my_crowns = 0 → 0/3 = 0.0
    assert obs[1] == 0.0
    # enemy_crowns = 0 → 0/3 = 0.0
    assert obs[2] == 0.0
    # card 0 cost = 2 → 2/9 ≈ 0.222
    assert abs(obs[3] - 2 / 9) < 1e-5


def test_step_do_nothing_returns_correct_shapes():
    env = make_env()
    env.reset()
    # action: card=4 (do nothing), x=0.5, y=0.5
    obs, reward, done, truncated, info = env.step(
        np.array([4.0, 0.5, 0.5], dtype=np.float32)
    )
    assert obs.shape == (OBS_SIZE,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert truncated is False


def test_step_does_not_play_unaffordable_card():
    env = make_env()
    env.reset()
    _mock_action.play_card.reset_mock()  # clear any calls from prior tests
    # card index 1 = "giant" cost 5, but let's set elixir to 3 so giant is unaffordable
    _mock_classify.get_game_state.return_value = {
        **MOCK_GAME_STATE,
        "elixir": 3,
        "playable_cards": ["goblins", "knight", "arrows"],  # giant excluded
    }
    env.step(np.array([1.0, 0.5, 0.5], dtype=np.float32))  # card 1 = giant
    _mock_action.play_card.assert_not_called()


def test_episode_done_when_i_reach_3_crowns():
    env = make_env()
    env.reset()
    _mock_crowns.get_crown_counts.return_value = (3, 0)
    _, _, done, _, _ = env.step(np.array([4.0, 0.5, 0.5], dtype=np.float32))
    assert done is True


def test_episode_done_when_enemy_reaches_3_crowns():
    env = make_env()
    env.reset()
    _mock_crowns.get_crown_counts.return_value = (0, 3)
    _, _, done, _, _ = env.step(np.array([4.0, 0.5, 0.5], dtype=np.float32))
    assert done is True


def test_obs_all_values_in_zero_to_one():
    env = make_env()
    obs, _ = env.reset()
    assert np.all(obs >= 0.0)
    assert np.all(obs <= 1.0)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_env.py -v
```

Expected: `ModuleNotFoundError: No module named 'env'`

- [ ] **Step 3: Implement `env.py`**

```python
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import classify
import crowns
import detect
import action
from tracker import TroopTracker
from reward import compute_reward

MAX_TROOPS: int = 20
OBS_SIZE: int = 1 + 1 + 1 + 4 + MAX_TROOPS * 4  # 87
STEP_INTERVAL: float = 0.5  # seconds between steps


class ClashRoyaleEnv(gym.Env):
    """
    Gymnasium environment for the Clash Royale bot.

    Observation (87 floats, all in [0, 1]):
        [0]     elixir / 10
        [1]     my_crowns / 3
        [2]     enemy_crowns / 3
        [3-6]   card costs / 9  (4 cards in hand)
        [7-86]  up to 20 troops × (norm_x, norm_y, is_enemy, norm_type_id)

    Action (Box of shape (3,)):
        [0]  card index as float in [0, 4.999] → int() gives 0-4 (4 = do nothing)
        [1]  placement x in [0, 1] (ignored when card=4)
        [2]  placement y in [0, 1] (ignored when card=4)
    """

    metadata = {"render_modes": []}

    def __init__(self, troop_name_to_id: dict = None, step_interval: float = STEP_INTERVAL) -> None:
        super().__init__()

        if troop_name_to_id is None:
            from ultralytics import YOLO
            _model = YOLO("runs/detect/current_model/weights/best.pt")
            troop_name_to_id = {name: idx for idx, name in _model.names.items()}

        self._troop_name_to_id = troop_name_to_id
        self._num_troop_types = max(len(troop_name_to_id), 1)
        self._step_interval = step_interval
        self._tracker = TroopTracker()
        self._prev_my_crowns: int = 0
        self._prev_enemy_crowns: int = 0

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        # action[0] in [0, 4.999] → int = card index (0-3 play, 4 do nothing)
        # action[1] in [0, 1]     → normalised placement x
        # action[2] in [0, 1]     → normalised placement y
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([4.999, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._tracker.reset()
        game_state = classify.get_game_state()
        my_c, enemy_c = crowns.get_crown_counts()
        self._prev_my_crowns = my_c
        self._prev_enemy_crowns = enemy_c
        return self._build_observation(game_state, my_c, enemy_c), {}

    def step(self, raw_action):
        card_idx = int(raw_action[0])   # 0-4
        norm_x = float(raw_action[1])
        norm_y = float(raw_action[2])

        # --- Act ---
        game_state = classify.get_game_state()
        if card_idx < 4:
            card_name = game_state["hand"][card_idx]["name"]
            if card_name in game_state["playable_cards"]:
                x_min, y_min, x_max, y_max = action.ARENA_BOUNDS
                arena_x = int(x_min + norm_x * (x_max - x_min))
                arena_y = int(y_min + norm_y * (y_max - y_min))
                action.play_card(card_idx, arena_x, arena_y)

        # --- Wait ---
        time.sleep(self._step_interval)

        # --- Sense ---
        game_state = classify.get_game_state()
        troops = detect.get_troops()
        my_crowns, enemy_crowns = crowns.get_crown_counts()

        # --- Track ---
        kill_events = self._tracker.update(troops, my_crowns, enemy_crowns)

        # --- Done ---
        done = my_crowns >= 3 or enemy_crowns >= 3
        won = my_crowns >= 3 if done else False

        # --- Reward ---
        reward = compute_reward(
            self._prev_my_crowns,
            self._prev_enemy_crowns,
            my_crowns,
            enemy_crowns,
            len(kill_events),
            done,
            won,
        )

        self._prev_my_crowns = my_crowns
        self._prev_enemy_crowns = enemy_crowns

        obs = self._build_observation(game_state, my_crowns, enemy_crowns)
        return obs, float(reward), done, False, {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self, game_state: dict, my_crowns: int, enemy_crowns: int
    ) -> np.ndarray:
        obs = np.zeros(OBS_SIZE, dtype=np.float32)

        obs[0] = game_state["elixir"] / 10.0
        obs[1] = my_crowns / 3.0
        obs[2] = enemy_crowns / 3.0

        for i, card in enumerate(game_state["hand"][:4]):
            obs[3 + i] = card["cost"] / 9.0

        capture_w = detect.monitor["width"]
        capture_h = detect.monitor["height"]
        troops = self._tracker.troops[:MAX_TROOPS]
        base = 7
        for i, troop in enumerate(troops):
            offset = base + i * 4
            obs[offset]     = troop.x / capture_w
            obs[offset + 1] = troop.y / capture_h
            obs[offset + 2] = 1.0 if troop.team == "enemy" else 0.0
            type_id = self._troop_name_to_id.get(troop.troop_type, 0)
            obs[offset + 3] = type_id / (self._num_troop_types - 1)

        return obs
```

- [ ] **Step 4: Run tests and confirm they pass**

```bash
pytest tests/test_env.py -v
```

Expected: all 7 tests pass.

- [ ] **Step 5: Validate environment with SB3's env checker**

Add a temporary script `check_env_quick.py` at project root:

```python
# check_env_quick.py — delete after running
from unittest.mock import MagicMock, patch
import sys

mock_classify = MagicMock()
mock_crowns = MagicMock()
mock_detect = MagicMock()
mock_action = MagicMock()
mock_detect.monitor = {"width": 555, "height": 780}
mock_action.ARENA_BOUNDS = (0, 390, 555, 780)
mock_classify.get_game_state.return_value = {
    "elixir": 5,
    "hand": [
        {"name": "goblins", "cost": 2},
        {"name": "giant", "cost": 5},
        {"name": "knight", "cost": 3},
        {"name": "arrows", "cost": 3},
    ],
    "playable_cards": ["goblins"],
}
mock_crowns.get_crown_counts.return_value = (0, 0)
mock_detect.get_troops.return_value = []
sys.modules["classify"] = mock_classify
sys.modules["crowns"] = mock_crowns
sys.modules["detect"] = mock_detect
sys.modules["action"] = mock_action

from stable_baselines3.common.env_checker import check_env
from env import ClashRoyaleEnv

env = ClashRoyaleEnv(troop_name_to_id={"goblins": 0, "giant": 1, "knight": 2, "arrows": 3})
check_env(env)
print("Environment passed SB3 check_env validation.")
```

Run it:

```bash
python check_env_quick.py
```

Expected output: `Environment passed SB3 check_env validation.`

Then delete the script:

```bash
del check_env_quick.py
```

- [ ] **Step 6: Commit**

```bash
git add env.py tests/test_env.py
git commit -m "feat: add Gymnasium RL environment with 87-value observation and Box(3) action space"
```

---

## Task 4: PPO Agent

**Files:**
- Create: `agent.py`

- [ ] **Step 1: Implement `agent.py`**

```python
import os
from stable_baselines3 import PPO
from env import ClashRoyaleEnv

MODEL_DIR: str = "models"
LATEST_MODEL_PATH: str = os.path.join(MODEL_DIR, "ppo_clash_royale_latest.zip")
CHECKPOINT_INTERVAL: int = 10  # save checkpoint every N episodes


def train(total_episodes: int = 1000) -> None:
    """
    Run the PPO training loop.

    Each call to model.learn() collects n_steps timesteps from the live game,
    then updates the policy. A checkpoint is saved every CHECKPOINT_INTERVAL
    episodes so training can be resumed after interruption.

    Args:
        total_episodes: number of complete games to train for.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    env = ClashRoyaleEnv()

    if os.path.exists(LATEST_MODEL_PATH):
        print(f"Resuming from {LATEST_MODEL_PATH}")
        model = PPO.load(LATEST_MODEL_PATH, env=env)
    else:
        print("Starting fresh training run.")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1,
        )

    for episode in range(1, total_episodes + 1):
        # Collect one episode's worth of steps and update the policy.
        # reset_num_timesteps=False preserves the global step counter across calls.
        model.learn(total_timesteps=2048, reset_num_timesteps=False)
        print(f"Episode {episode}/{total_episodes} complete.")

        if episode % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = os.path.join(MODEL_DIR, f"ppo_ep{episode:05d}.zip")
            model.save(checkpoint_path)
            model.save(LATEST_MODEL_PATH)
            print(f"Checkpoint saved → {checkpoint_path}")

    model.save(LATEST_MODEL_PATH)
    print(f"Training complete. Final model saved to {LATEST_MODEL_PATH}")


if __name__ == "__main__":
    train()
```

- [ ] **Step 2: Run a 1-episode smoke test against the live game**

Open Clash Royale to a live match, then run:

```bash
python agent.py
```

Watch for:
- `"Starting fresh training run."` printed
- No import errors or crashes on first `env.reset()` call
- Step loop running at roughly 0.5s intervals
- `"Episode 1/1000 complete."` printed after the first game ends

Stop with `Ctrl+C` after confirming the loop is running. A partial `models/` directory is expected.

- [ ] **Step 3: Commit**

```bash
git add agent.py
git commit -m "feat: add PPO training loop with SB3 and 10-episode checkpointing"
```

---

## Task 5: Calibrate Arena Midline

**Files:**
- Modify: `tracker.py` line 13 (`ARENA_MIDLINE_Y`)

- [ ] **Step 1: Find the midline Y-coordinate**

Run `detect.py` while a game is in progress and note the Y-coordinates of:
- A troop you just deployed in your half (should be > midline)
- An enemy troop visible in the enemy half (should be < midline)

```bash
python detect.py
```

Look at the printed Y values. The midline should sit between the two groups.

- [ ] **Step 2: Update `ARENA_MIDLINE_Y` in `tracker.py`**

Replace the current value with the confirmed one. For example if enemy troops appear around Y=150 and your troops around Y=600:

```python
ARENA_MIDLINE_Y: int = 390  # replace 390 with your confirmed value
```

- [ ] **Step 3: Re-run tracker tests to confirm nothing broke**

```bash
pytest tests/test_tracker.py -v
```

Expected: all 9 tests still pass (tests use `ARENA_MIDLINE_Y` directly so they adapt automatically).

- [ ] **Step 4: Commit**

```bash
git add tracker.py
git commit -m "calibrate: set ARENA_MIDLINE_Y to confirmed screen value"
```

---

## Final Check

Run the full test suite:

```bash
pytest tests/ -v
```

Expected: all 25 tests pass (9 tracker + 9 reward + 7 env).
