# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CRBot is a Clash Royale bot that uses computer vision to read game state and (eventually) play cards autonomously using reinforcement learning. It runs on Windows and captures a live game screen.

## Environment Setup

```bash
# Activate virtual environment
.venv\Scripts\activate

# Install dependencies (if needed)
pip install ultralytics pyautogui opencv-python numpy mss
```

## Running the Modules

Each module can be run directly for testing:

```bash
python elixir.py          # Test elixir bar detection (loops every 0.5s)
python crowns.py          # Test crown detection (loops every 1s)
python classify.py        # Test hand card classification + game state (loops every 0.5s)
python detect.py          # Run live troop detection with YOLO (press Q to quit)
python main.py            # Run the reward tracking loop
```

## Training Models

Edit the `RUN_CLASSIFICATION` flag in `train.py` (0 = detection, 1 = classification), then:

```bash
python train.py
```

Trained weights are saved to:
- Detection: `runs/detect/current_model/weights/best.pt`
- Classification: `runs/classify/current_model/weights/best.pt`

Training was done on YOLOv11s with CUDA (RTX 4060). Must wrap training in `if __name__ == "__main__"` on Windows due to multiprocessing issues.

## Architecture

Game state is read through three parallel systems:

1. **Elixir** (`elixir.py`) — Counts purple pixels in the elixir bar region via `pyautogui.screenshot`. Uses pixel thresholds calibrated for each elixir level 0–10. Handles the "Elixir full" pink overlay as a special case.

2. **Crown detection** (`crowns.py`) — Samples specific pixel positions from crown slot regions. Matches yellow/blue/red/gray colors to determine 0–3 crowns for each side. Crown slots are always fixed-position, so tolerance is kept low (40).

3. **Card classification** (`classify.py`) — Captures the 4-card hand region, splits it into individual card images, and classifies each with a YOLO classification model. Combines with elixir to determine `playable_cards`. Returns a structured `game_state` dict.

4. **Troop detection** (`detect.py`) — Runs a YOLO detection model on a live `mss` screen capture. Returns troop positions as `{troop, x, y, conf}` dicts.

5. **Reward system** (`main.py`) — `CrownTracker` computes a reward signal: +50 per friendly crown gained, −50 per enemy crown gained.

`card_data.py` contains the static `elixir_costs` dictionary mapping card names to their elixir cost.

## Screen Region Configuration

All screen coordinates are hardcoded for a specific monitor setup. If running on a different machine, update these constants:

- `ELIXIR_REGION` in `elixir.py`
- `MY_CROWN_REGION`, `ENEMY_CROWN_REGION` in `crowns.py`
- `HAND_REGION` in `classify.py`
- `monitor` dict in `detect.py`

Use `python -m pyautogui` in the terminal to find pixel coordinates interactively.