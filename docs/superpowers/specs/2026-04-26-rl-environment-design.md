# RL Environment Design — Clash Royale Bot

**Date:** 2026-04-26  
**Status:** Approved

---

## Overview

Add a reinforcement learning layer to the existing Clash Royale bot using PPO (Proximal Policy Optimization) via Stable Baselines 3. The existing sensing files (`detect.py`, `crowns.py`, `classify.py`, `action.py`, `card_data.py`) remain unchanged and are treated as read-only sensors. All RL logic lives in four new files.

---

## Architecture

| File | Responsibility |
|---|---|
| `tracker.py` | Tracks troops across frames, classifies ally/enemy by Y-position, detects deaths |
| `reward.py` | Computes reward each step from crown deltas and kill events |
| `env.py` | Gymnasium-compatible RL environment — `step()`, `reset()`, observation/action spaces |
| `agent.py` | PPO agent setup and training loop via Stable Baselines 3 (PyTorch backend) |

---

## Observation Space

A fixed-size flat vector of **87 values** fed to the PPO neural network each step:

| Field | Size | Encoding |
|---|---|---|
| Elixir | 1 | Normalized `value / 10` → `[0, 1]` |
| My crowns | 1 | Raw integer `0–3` |
| Enemy crowns | 1 | Raw integer `0–3` |
| Cards in hand | 4 | Elixir cost of each card, normalized `cost / 9` → `[0, 1]` |
| Troop slots | 80 | Up to 20 troops × 4 values each (see below) |

**Per-troop encoding (4 values):**
- `x` — normalized to `[0, 1]` relative to capture region width
- `y` — normalized to `[0, 1]` relative to capture region height
- `is_enemy` — `1.0` for enemy, `0.0` for ally
- `troop_type_id` — numeric ID mapped from troop name (e.g. `"giant" → 5`); the mapping is derived at startup directly from `model.names` (the YOLO model's class index), so it stays in sync with the detection model automatically

If fewer than 20 troops are detected, remaining slots are zero-padded.

Cards-in-hand use elixir cost as the scalar representation — a meaningful value the agent can reason about without needing card embeddings.

---

## Action Space

Hybrid action space with two heads output simultaneously by PPO:

**Discrete head — card selection (5 options):**
| Value | Action |
|---|---|
| 0–3 | Play card in slot 0–3 |
| 4 | Do nothing |

**Continuous head — placement (2 values):**
- `(x, y)` each normalized to `[0, 1]` relative to friendly arena bounds from `action.py`
- Ignored when discrete head outputs `4` (do nothing)

**Invalid action masking:**  
`classify.get_game_state()` already returns a `playable_cards` list — cards whose cost is within current elixir. If the agent selects a card not in `playable_cards`, the action is overridden to "do nothing" rather than allowing a failed play. This prevents wasted steps.

---

## Troop Tracker (`tracker.py`)

Maintains a list of active troops across frames. Each tracked troop stores:

```
id, troop_type, x, y, team (ally/enemy), frames_since_seen
```

**Per-step logic:**

1. **Match** — new detections matched to existing tracked troops by proximity (closest within 50px threshold)
2. **Update** — matched troops have their position updated; `frames_since_seen` reset to 0
3. **Classify new** — unmatched detections are new troops:
   - If `my_crowns == 0 and enemy_crowns == 0`: classify by Y-position
     - Above arena midline → enemy
     - Below arena midline → ally
     - The midline Y-coordinate is a calibration constant (approximately `monitor["height"] / 2` in capture-region pixels) that must be confirmed by inspection before first run
   - If any crown has been destroyed: skip classification — new troops are not tracked for kill rewards
4. **Detect deaths** — tracked enemy troops with `frames_since_seen >= 3` are declared dead → triggers kill reward, removed from tracker
5. **Prune** — tracked ally troops that vanish are removed silently (no reward)

**Why 3-frame grace period:** YOLO occasionally misses a troop for 1–2 frames. Waiting 3 frames before declaring death prevents false kill rewards from detection gaps.

---

## Reward Structure (`reward.py`)

Rewards computed each step by comparing current state to previous step's state:

| Event | Reward | Condition |
|---|---|---|
| Enemy troop killed | `+0.1` | Only while `my_crowns == 0 and enemy_crowns == 0` |
| I earn a crown | `+8.0` | Crown delta on enemy side going up |
| Enemy earns a crown | `-8.0` | Crown delta on my side going up |
| Win (3 crowns or most crowns) | `+15.0` | End of episode |
| Loss | `-15.0` | End of episode |

**Crown delta logic:** each step, compare `current_crowns` vs `previous_crowns`. An increase triggers the reward — no special event detection needed, just a frame-to-frame diff.

**Magnitude rationale:** crowns are 80× more impactful than kills. This ensures the agent prioritises tower damage. Kill reward is a weak nudge toward pressure — noisy detections won't meaningfully skew training.

All values are starting points and should be tuned once training is running.

---

## Game Loop (`env.py`)

Standard Gymnasium interface.

**`reset()`** — called at the start of each game:
- Clears the troop tracker
- Calls `classify.get_game_state()` and `crowns.get_crown_counts()` for initial state
- Returns the first observation vector

**`step(action)`** — called repeatedly during a game:
1. **Act** — decode discrete + continuous action; if not "do nothing" and elixir is sufficient, call `action.play_card()`
2. **Wait** — sleep `0.5s` step interval to let game state evolve
3. **Sense** — call `classify.get_game_state()` (returns elixir + hand + playable cards), `detect.get_troops()`, `crowns.get_crown_counts()`
4. **Track** — update tracker with new detections, collect kill events
5. **Reward** — compute reward from crown deltas + kill events
6. **Done check** — episode ends when either side reaches 3 crowns
7. **Return** — `(observation, reward, done, info)`

**Step interval:** `0.5s` is the starting point. Can be tuned based on observed agent responsiveness.

---

## PPO Agent (`agent.py`)

Uses **Stable Baselines 3** with PyTorch backend (already installed via Ultralytics).

- Policy: `MlpPolicy` — standard feedforward network, appropriate for the flat 87-value observation vector
- The training loop, experience collection, advantage estimation, and weight updates are handled entirely by SB3

**Starting hyperparameters:**

| Parameter | Value | Purpose |
|---|---|---|
| `learning_rate` | `3e-4` | Weight update speed |
| `n_steps` | `2048` | Steps collected before each update |
| `batch_size` | `64` | Minibatch size during update |
| `n_epochs` | `10` | Update passes per collected batch |
| `gamma` | `0.99` | Future reward discount factor |

Model checkpoints are saved to disk after every 10 episodes so training can be resumed without starting over.

---

## Key Constraints & Decisions

- **No model retraining** — YOLO detection model is used as-is; ally/enemy classification is done purely via Y-position heuristic
- **Kill reward gated on crown state** — once any tower is destroyed, Y-position heuristic becomes unreliable; kill rewards are disabled for the rest of that episode
- **Existing files untouched** — `detect.py`, `crowns.py`, `classify.py`, `action.py`, `card_data.py` are pure sensors with no RL logic added
