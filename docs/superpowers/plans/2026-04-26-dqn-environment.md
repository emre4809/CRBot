# DQN Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the rule-based agent with a DQN (Deep Q-Network) reinforcement learning agent that learns to play Clash Royale by selecting cards and placement positions from a discretized 6×4 grid.

**Architecture:** Four new files — `tracker.py` (troop tracking across frames), `reward.py` (reward computation), `env.py` (Gymnasium environment), and a rewritten `agent.py` (SB3 DQN training loop). Existing sensor files (`detect.py`, `crowns.py`, `classify.py`, `action.py`, `card_data.py`) are read-only and untouched.

**Tech Stack:** Python, Gymnasium, Stable Baselines 3 (DQN), PyTorch (via SB3), existing Ultralytics/YOLO stack.

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `tracker.py` | **Create** | Track troops across frames, detect kills |
| `reward.py` | **Create** | Compute scalar reward each step |
| `env.py` | **Create** | Gymnasium `Discrete(97)` environment |
| `agent.py` | **Replace** | SB3 DQN training loop |
| `tests/test_tracker.py` | **Create** | Unit tests for TroopTracker |
| `tests/test_reward.py` | **Create** | Unit tests for compute_reward |
| `tests/test_env.py` | **Create** | Unit tests for ClashRoyaleEnv (mocked sensors) |

**Do not modify:** `detect.py`, `crowns.py`, `classify.py`, `action.py`, `card_data.py`, `elixir.py`

---

## Sensor API Reference (read-only)

```python
# detect.get_troops() → list of:
{"troop": str, "x": int, "y": int, "conf": float}
# x, y are in capture-region pixels: x ∈ [0, 555], y ∈ [0, 780]

# detect.model.names → {int: str}  e.g. {0: "giant", 1: "knight", ...}
# detect.monitor → {"top": 40, "left": 735, "width": 555, "height": 780}

# classify.get_game_state() → dict:
{"elixir": int, "hand": [{"name": str, "cost": int}, ...], "playable_cards": [str]}

# crowns.get_crown_counts() → (my_crowns: int, enemy_crowns: int)  each 0–3

# action.ARENA_BOUNDS → (779, 473, 1251, 774)  (x_min, y_min, x_max, y_max)
# action.play_card(slot_index: int, arena_x: int, arena_y: int) → bool
```

---

## Task 1: Install Stable Baselines 3

**Files:** none

- [ ] **Step 1: Install SB3**

```bash
.venv\Scripts\activate && pip install stable-baselines3
```

Expected: SB3 installs successfully (or "already satisfied").

- [ ] **Step 2: Verify import**

```bash
python -c "from stable_baselines3 import DQN; print('DQN ok')"
```

Expected output: `DQN ok`

---

## Task 2: Troop Tracker

**Files:**
- Create: `tracker.py`
- Create: `tests/test_tracker.py`

- [ ] **Step 1: Create tests directory**

```bash
mkdir tests
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_tracker.py`:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from tracker import TroopTracker, MIDLINE_Y


def test_enemy_classified_above_midline():
    tracker = TroopTracker()
    det = [{"troop": "giant", "x": 100, "y": MIDLINE_Y - 50, "conf": 0.9}]
    tracker.update(det, my_crowns=0, enemy_crowns=0)
    assert tracker.troops[0]["team"] == "enemy"


def test_ally_classified_below_midline():
    tracker = TroopTracker()
    det = [{"troop": "knight", "x": 100, "y": MIDLINE_Y + 50, "conf": 0.9}]
    tracker.update(det, my_crowns=0, enemy_crowns=0)
    assert tracker.troops[0]["team"] == "ally"


def test_no_new_troops_when_crown_fallen():
    tracker = TroopTracker()
    det = [{"troop": "giant", "x": 100, "y": 100, "conf": 0.9}]
    tracker.update(det, my_crowns=1, enemy_crowns=0)
    assert tracker.troops == []


def test_kill_on_third_missing_frame():
    tracker = TroopTracker()
    det = [{"troop": "giant", "x": 100, "y": MIDLINE_Y - 10, "conf": 0.9}]
    tracker.update(det, 0, 0)
    kills1 = tracker.update([], 0, 0)
    kills2 = tracker.update([], 0, 0)
    kills3 = tracker.update([], 0, 0)
    assert kills1 == [] and kills2 == []
    assert len(kills3) == 1 and kills3[0]["troop"] == "giant"
    assert tracker.troops == []


def test_position_updated_on_match():
    tracker = TroopTracker()
    tracker.update([{"troop": "giant", "x": 100, "y": 100, "conf": 0.9}], 0, 0)
    tracker.update([{"troop": "giant", "x": 115, "y": 108, "conf": 0.9}], 0, 0)
    assert tracker.troops[0]["x"] == 115
    assert tracker.troops[0]["y"] == 108


def test_far_detection_not_matched():
    tracker = TroopTracker()
    tracker.update([{"troop": "giant", "x": 100, "y": 100, "conf": 0.9}], 0, 0)
    # detection 200px away — should create a new troop, not match
    tracker.update([{"troop": "giant", "x": 300, "y": 100, "conf": 0.9}], 0, 0)
    assert len(tracker.troops) == 2


def test_ally_vanish_no_kill():
    tracker = TroopTracker()
    det = [{"troop": "knight", "x": 100, "y": MIDLINE_Y + 50, "conf": 0.9}]
    tracker.update(det, 0, 0)
    kills1 = tracker.update([], 0, 0)
    kills2 = tracker.update([], 0, 0)
    kills3 = tracker.update([], 0, 0)
    assert kills1 == [] and kills2 == [] and kills3 == []


def test_clear_resets_state():
    tracker = TroopTracker()
    tracker.update([{"troop": "giant", "x": 100, "y": 100, "conf": 0.9}], 0, 0)
    tracker.clear()
    assert tracker.troops == []
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
python -m pytest tests/test_tracker.py -v
```

Expected: All tests fail with `ModuleNotFoundError: No module named 'tracker'`

- [ ] **Step 4: Implement tracker.py**

Create `tracker.py`:

```python
MIDLINE_Y = 390  # capture-region midpoint: monitor height 780 / 2
MATCH_THRESHOLD = 50  # pixels — max distance to match a detection to a tracked troop
DEATH_FRAMES = 3  # frames_since_seen threshold to declare an enemy dead


class TroopTracker:
    def __init__(self):
        self.troops = []
        self._next_id = 0

    def clear(self):
        self.troops = []
        self._next_id = 0

    def update(self, detections, my_crowns, enemy_crowns):
        """
        Match detections to tracked troops, add new ones, detect deaths.
        Returns list of killed enemy troop dicts (only when no crowns have fallen).
        """
        for t in self.troops:
            t["frames_since_seen"] += 1

        matched_ids = set()
        unmatched = []

        for det in detections:
            best_id = None
            best_dist = float("inf")
            for t in self.troops:
                if t["id"] in matched_ids:
                    continue
                dist = ((t["x"] - det["x"]) ** 2 + (t["y"] - det["y"]) ** 2) ** 0.5
                if dist < best_dist and dist <= MATCH_THRESHOLD:
                    best_dist = dist
                    best_id = t["id"]

            if best_id is not None:
                matched_ids.add(best_id)
                troop = next(t for t in self.troops if t["id"] == best_id)
                troop["x"] = det["x"]
                troop["y"] = det["y"]
                troop["frames_since_seen"] = 0
            else:
                unmatched.append(det)

        if my_crowns == 0 and enemy_crowns == 0:
            for det in unmatched:
                team = "enemy" if det["y"] < MIDLINE_Y else "ally"
                self.troops.append({
                    "id": self._next_id,
                    "troop": det["troop"],
                    "x": det["x"],
                    "y": det["y"],
                    "team": team,
                    "frames_since_seen": 0,
                })
                self._next_id += 1

        kills = [t for t in self.troops if t["team"] == "enemy" and t["frames_since_seen"] >= DEATH_FRAMES]
        self.troops = [t for t in self.troops if t["frames_since_seen"] < DEATH_FRAMES]
        return kills
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_tracker.py -v
```

Expected: All 8 tests pass.

- [ ] **Step 6: Commit**

```bash
git add tracker.py tests/test_tracker.py
git commit -m "feat: add TroopTracker with proximity matching and death detection"
```

---

## Task 3: Reward Function

**Files:**
- Create: `reward.py`
- Create: `tests/test_reward.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_reward.py`:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from reward import compute_reward


def test_neutral_step_no_reward():
    assert compute_reward(0, 0, 0, 0, [], False, False) == pytest.approx(0.0)


def test_my_crown_gain():
    assert compute_reward(0, 0, 1, 0, [], False, False) == pytest.approx(8.0)


def test_enemy_crown_gain():
    assert compute_reward(0, 0, 0, 1, [], False, False) == pytest.approx(-8.0)


def test_kill_reward():
    kills = [{"troop": "giant", "team": "enemy"}]
    assert compute_reward(0, 0, 0, 0, kills, False, False) == pytest.approx(0.1)


def test_win_terminal_bonus():
    # 3 crowns earned + win bonus
    result = compute_reward(0, 0, 3, 0, [], True, True)
    assert result == pytest.approx(8.0 * 3 + 15.0)


def test_loss_terminal_penalty():
    result = compute_reward(0, 0, 0, 3, [], True, False)
    assert result == pytest.approx(-8.0 * 3 - 15.0)


def test_no_terminal_reward_when_not_done():
    # even if won=True is passed, terminal reward only applies when terminated=True
    assert compute_reward(0, 0, 0, 0, [], False, True) == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_reward.py -v
```

Expected: All tests fail with `ModuleNotFoundError: No module named 'reward'`

- [ ] **Step 3: Implement reward.py**

Create `reward.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_reward.py -v
```

Expected: All 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add reward.py tests/test_reward.py
git commit -m "feat: add compute_reward with crown delta, kill, and terminal signals"
```

---

## Task 4: Gymnasium Environment

**Files:**
- Create: `env.py`
- Create: `tests/test_env.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_env.py`:

```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# --- Mock all sensor modules before importing env ---
mock_detect = MagicMock()
mock_detect.get_troops.return_value = []
mock_detect.model.names = {0: "giant", 1: "knight", 2: "archer"}
mock_detect.monitor = {"width": 555, "height": 780}

_default_state = {
    "elixir": 5,
    "hand": [
        {"name": "giant", "cost": 5},
        {"name": "knight", "cost": 3},
        {"name": "archer", "cost": 3},
        {"name": "arrows", "cost": 3},
    ],
    "playable_cards": ["giant", "knight", "archer", "arrows"],
}
mock_classify = MagicMock()
mock_classify.get_game_state.return_value = _default_state

mock_crowns = MagicMock()
mock_crowns.get_crown_counts.return_value = (0, 0)

mock_action = MagicMock()
mock_action.play_card.return_value = True
mock_action.ARENA_BOUNDS = (779, 473, 1251, 774)

sys.modules["detect"] = mock_detect
sys.modules["classify"] = mock_classify
sys.modules["crowns"] = mock_crowns
sys.modules["action"] = mock_action

from env import ClashRoyaleEnv, N_ACTIONS, N_GRID


# --- Tests ---

def make_env():
    return ClashRoyaleEnv()


def test_observation_shape():
    env = make_env()
    obs, _ = env.reset()
    assert obs.shape == (87,)


def test_action_space_size():
    env = make_env()
    assert env.action_space.n == 97


def test_elixir_normalized_in_obs():
    env = make_env()
    obs, _ = env.reset()
    assert obs[0] == pytest.approx(0.5)  # elixir=5, /10=0.5


def test_do_nothing_skips_play_card():
    env = make_env()
    env.reset()
    mock_action.play_card.reset_mock()
    with patch("time.sleep"):
        env.step(96)  # do nothing
    mock_action.play_card.assert_not_called()


def test_affordable_card_calls_play_card():
    env = make_env()
    env.reset()
    mock_action.play_card.reset_mock()
    # card slot 1 (knight, cost 3), cell 0 → action = 1 * 24 + 0 = 24
    with patch("time.sleep"):
        env.step(24)
    mock_action.play_card.assert_called_once()


def test_unaffordable_card_skips_play_card():
    env = make_env()
    env.reset()
    mock_action.play_card.reset_mock()
    mock_classify.get_game_state.return_value = {
        "elixir": 2,
        "hand": [
            {"name": "giant", "cost": 5},
            {"name": "knight", "cost": 3},
            {"name": "archer", "cost": 3},
            {"name": "arrows", "cost": 3},
        ],
        "playable_cards": [],
    }
    with patch("time.sleep"):
        env.step(0)  # card 0 (giant), not in playable_cards
    mock_action.play_card.assert_not_called()
    mock_classify.get_game_state.return_value = _default_state


def test_episode_terminates_at_3_my_crowns():
    env = make_env()
    env.reset()
    mock_crowns.get_crown_counts.return_value = (3, 0)
    with patch("time.sleep"):
        _, _, terminated, _, _ = env.step(96)
    assert terminated is True
    mock_crowns.get_crown_counts.return_value = (0, 0)


def test_episode_terminates_at_3_enemy_crowns():
    env = make_env()
    env.reset()
    mock_crowns.get_crown_counts.return_value = (0, 3)
    with patch("time.sleep"):
        _, _, terminated, _, _ = env.step(96)
    assert terminated is True
    mock_crowns.get_crown_counts.return_value = (0, 0)


def test_step_returns_five_tuple():
    env = make_env()
    env.reset()
    with patch("time.sleep"):
        result = env.step(96)
    assert len(result) == 5  # obs, reward, terminated, truncated, info


def test_obs_dtype_is_float32():
    env = make_env()
    obs, _ = env.reset()
    assert obs.dtype == np.float32
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_env.py -v
```

Expected: All tests fail with `ModuleNotFoundError: No module named 'env'`

- [ ] **Step 3: Implement env.py**

Create `env.py`:

```python
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._tracker.clear()
        state = get_game_state()
        my_crowns, enemy_crowns = get_crown_counts()
        self._prev_my_crowns = my_crowns
        self._prev_enemy_crowns = enemy_crowns
        obs = self._build_obs(state, my_crowns, enemy_crowns)
        return obs, {}

    def step(self, action):
        if action < N_ACTIONS - 1:
            card_slot = action // N_GRID
            cell = action % N_GRID
            px, py = self._grid[cell]
            state = get_game_state()
            card_name = state["hand"][card_slot]["name"]
            if card_name in state["playable_cards"]:
                play_card(card_slot, px, py)

        time.sleep(STEP_INTERVAL)

        state = get_game_state()
        detections = get_troops()
        my_crowns, enemy_crowns = get_crown_counts()

        kills = self._tracker.update(detections, my_crowns, enemy_crowns)

        terminated = my_crowns == 3 or enemy_crowns == 3
        won = my_crowns == 3

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
        obs[1] = float(my_crowns)
        obs[2] = float(enemy_crowns)
        for i, card in enumerate(state["hand"][:4]):
            obs[3 + i] = card["cost"] / 9.0
        for i, t in enumerate(self._tracker.troops[:MAX_TROOPS]):
            base = 7 + i * 4
            obs[base]     = t["x"] / self._region_w
            obs[base + 1] = t["y"] / self._region_h
            obs[base + 2] = 1.0 if t["team"] == "enemy" else 0.0
            obs[base + 3] = self._troop_id.get(t["troop"], 0) / self._n_classes
        return obs
```

- [ ] **Step 4: Install gymnasium if needed**

```bash
python -c "import gymnasium" 2>/dev/null || pip install gymnasium
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_env.py -v
```

Expected: All 10 tests pass.

- [ ] **Step 6: Commit**

```bash
git add env.py tests/test_env.py
git commit -m "feat: add ClashRoyaleEnv with Discrete(97) action space and 87-value observation"
```

---

## Task 5: DQN Agent

**Files:**
- Modify: `agent.py` (full replacement)

- [ ] **Step 1: Replace agent.py**

Overwrite `agent.py` with:

```python
import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from env import ClashRoyaleEnv

CHECKPOINT_DIR = "checkpoints"
TOTAL_TIMESTEPS = 500_000


def train(resume_from=None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    env = ClashRoyaleEnv()

    checkpoint_callback = CheckpointCallback(
        save_freq=5_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="dqn_cr",
    )

    if resume_from:
        model = DQN.load(resume_from, env=env)
        print(f"Resuming from {resume_from}")
    else:
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=1e-4,
            buffer_size=10_000,
            learning_starts=1_000,
            batch_size=32,
            gamma=0.99,
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            target_update_interval=500,
            verbose=1,
        )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback, reset_num_timesteps=resume_from is None)
    model.save(os.path.join(CHECKPOINT_DIR, "dqn_cr_final"))
    print("Training complete. Final model saved.")


if __name__ == "__main__":
    import sys
    resume = sys.argv[1] if len(sys.argv) > 1 else None
    train(resume_from=resume)
```

- [ ] **Step 2: Verify agent imports cleanly (without running a game)**

```bash
python -c "from agent import train; print('agent ok')"
```

Expected output: `agent ok`

- [ ] **Step 3: Commit**

```bash
git add agent.py
git commit -m "feat: replace rule-based agent with SB3 DQN training loop"
```

---

## Task 6: Smoke Test Full Stack

**Files:** none (read-only verification)

- [ ] **Step 1: Run all unit tests**

```bash
python -m pytest tests/ -v
```

Expected: All tests pass (tracker, reward, env).

- [ ] **Step 2: Verify env instantiates live**

With BlueStacks running and a game on screen:

```bash
python -c "
from env import ClashRoyaleEnv
env = ClashRoyaleEnv()
obs, _ = env.reset()
print('obs shape:', obs.shape)
print('obs[:7]:', obs[:7])
print('Smoke test passed')
"
```

Expected: prints `obs shape: (87,)` and the first 7 values (elixir, crowns, hand costs).

- [ ] **Step 3: Start training**

With BlueStacks running and in a live game:

```bash
python agent.py
```

To resume from a checkpoint:

```bash
python agent.py checkpoints/dqn_cr_5000_steps
```

---

## Calibration Note

Before first training run, verify `MIDLINE_Y = 390` in `tracker.py` by inspecting live troop detections:

```bash
python -c "
from detect import get_troops, monitor
troops = get_troops()
for t in troops:
    print(t['troop'], 'y =', t['y'], '(midline =', monitor['height']//2, ')')
"
```

Adjust `MIDLINE_Y` in `tracker.py` if ally troops are appearing above 390 or enemy troops below.
