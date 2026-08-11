# DQN Environment Design — Clash Royale Bot

**Date:** 2026-04-26  
**Status:** Approved

---

## Overview

Replace the PPO-based RL layer with DQN (Deep Q-Network) via Stable Baselines 3. The existing sensing files (`detect.py`, `crowns.py`, `classify.py`, `action.py`, `card_data.py`) remain unchanged and are treated as read-only sensors. All RL logic lives in four new files, mirroring the PPO spec's structure.

---

## Architecture

| File | Responsibility |
|---|---|
| `tracker.py` | Tracks troops across frames, classifies ally/enemy by Y-position, detects deaths |
| `reward.py` | Computes reward each step from crown deltas and kill events |
| `env.py` | Gymnasium-compatible RL environment — `step()`, `reset()`, observation/action spaces |
| `agent.py` | DQN agent setup and training loop via Stable Baselines 3 (PyTorch backend) |

---

## Observation Space

Identical to the PPO spec — a fixed-size flat vector of **87 values**:

| Field | Size | Encoding |
|---|---|---|
| Elixir | 1 | Normalized `value / 10` → `[0, 1]` |
| My crowns | 1 | Raw integer `0–3` |
| Enemy crowns | 1 | Raw integer `0–3` |
| Cards in hand | 4 | Elixir cost of each card, normalized `cost / 9` → `[0, 1]` |
| Troop slots | 80 | Up to 20 troops × 4 values each |

**Per-troop encoding (4 values):**
- `x` — normalized to `[0, 1]` relative to capture region width
- `y` — normalized to `[0, 1]` relative to capture region height
- `is_enemy` — `1.0` for enemy, `0.0` for ally
- `troop_type_id` — numeric ID from `model.names` (YOLO class index)

Remaining troop slots are zero-padded when fewer than 20 troops are detected.

---

## Action Space

**Single `Discrete(97)` space** — replaces the hybrid discrete+continuous heads required by PPO.

### Encoding

- Actions `0–95`: play card `c` at grid cell `g` → `action = c * 24 + g`
  - `c` ∈ `{0, 1, 2, 3}` — card slot in hand
  - `g` ∈ `{0, ..., 23}` — cell index in a **6-column × 4-row** grid over the friendly arena
- Action `96`: do nothing

### Grid Layout

The 6×4 grid is laid over `ARENA_BOUNDS = (779, 473, 1251, 774)` from `action.py`:
- **Columns:** 6 evenly-spaced x-positions across arena width (~79px apart)
- **Rows:** 4 evenly-spaced y-positions across arena height (~75px apart)
- Cell `g = row * 6 + col` → decoded to pixel coords at action execution time

### Invalid Action Masking

If the agent selects a card not in `playable_cards` (cards whose cost ≤ current elixir), the action is overridden to do nothing. This prevents wasted steps on unaffordable cards.

---

## Troop Tracker (`tracker.py`)

Unchanged from PPO spec. Each tracked troop stores:

```
id, troop_type, x, y, team (ally/enemy), frames_since_seen
```

**Per-step logic:**

1. **Match** — new detections matched to existing troops by proximity (closest within 50px)
2. **Update** — matched troops have position updated; `frames_since_seen` reset to 0
3. **Classify new** — unmatched detections are new troops:
   - If `my_crowns == 0 and enemy_crowns == 0`: classify by Y-position (above midline → enemy, below → ally)
   - If any crown destroyed: skip classification, no kill rewards tracked
4. **Detect deaths** — enemy troops with `frames_since_seen >= 3` are declared dead → kill reward, removed
5. **Prune** — vanished ally troops removed silently

The midline Y-coordinate is a calibration constant (~`monitor["height"] / 2` in capture-region pixels).

---

## Reward Structure (`reward.py`)

Unchanged from PPO spec:

| Event | Reward | Condition |
|---|---|---|
| Enemy troop killed | `+0.1` | Only while `my_crowns == 0 and enemy_crowns == 0` |
| I earn a crown | `+8.0` | Crown delta on enemy side going up |
| Enemy earns a crown | `-8.0` | Crown delta on my side going up |
| Win (3 crowns or most crowns) | `+15.0` | End of episode |
| Loss | `-15.0` | End of episode |

All values are starting points to be tuned once training is running.

---

## Game Loop (`env.py`)

Standard Gymnasium interface with `Discrete(97)` action space.

**`reset()`** — called at the start of each game:
- Clears the troop tracker
- Calls `classify.get_game_state()` and `crowns.get_crown_counts()` for initial state
- Returns the first observation vector

**`step(action)`** — called repeatedly during a game:
1. **Decode** — if action < 96: extract `card = action // 24`, `cell = action % 24`; compute pixel coords from grid cell
2. **Act** — if not do-nothing and card is in `playable_cards`, call `action.play_card(card, pixel_x, pixel_y)`
3. **Wait** — sleep `0.5s` step interval
4. **Sense** — call `classify.get_game_state()`, `detect.get_troops()`, `crowns.get_crown_counts()`
5. **Track** — update tracker, collect kill events
6. **Reward** — compute reward from crown deltas + kill events
7. **Done check** — episode ends when either side reaches 3 crowns
8. **Return** — `(observation, reward, terminated, truncated, info)`

---

## DQN Agent (`agent.py`)

Uses **Stable Baselines 3 `DQN`** with PyTorch backend.

- Policy: `MlpPolicy` — standard feedforward network for the flat 87-value observation vector
- Action space: `Discrete(97)` — single integer output per step

**Hyperparameters:**

| Parameter | Value | Purpose |
|---|---|---|
| `learning_rate` | `1e-4` | Weight update speed |
| `buffer_size` | `10_000` | Replay buffer capacity |
| `learning_starts` | `1_000` | Steps before training begins |
| `batch_size` | `32` | Minibatch from replay buffer |
| `gamma` | `0.99` | Future reward discount factor |
| `exploration_fraction` | `0.2` | Fraction of training steps for epsilon decay |
| `exploration_final_eps` | `0.05` | Minimum exploration rate (5%) |
| `target_update_interval` | `500` | Steps between target network syncs |

Model checkpoints saved to disk after every 10 episodes so training can be resumed.

---

## Key Constraints & Decisions

- **Single discrete action space** — DQN requires `Discrete(N)`; the 6×4 placement grid fuses card selection and placement into 97 actions
- **Grid resolution** — 6×4 (24 cells) balances placement precision against action space size and training speed
- **No model retraining** — YOLO used as-is; ally/enemy classification via Y-position heuristic
- **Kill reward gated on crown state** — once any tower falls, Y-position heuristic is unreliable; kill rewards disabled for remainder of episode
- **Existing files untouched** — `detect.py`, `crowns.py`, `classify.py`, `action.py`, `card_data.py` are pure sensors
