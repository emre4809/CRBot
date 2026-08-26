import os
import re
import cv2
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
import detect
from env import ClashRoyaleEnv

CHECKPOINT_DIR = "checkpoints"
TOTAL_TIMESTEPS = 500_000

# Set True to show a live troop-detection window (mirrors the game with
# bounding boxes drawn on it) you can put next to your match while training.
# Off by default -- no window, no extra rendering cost.
SHOW_DETECTION_WINDOW = True


def _derive_replay_buffer_path(model_path):
    """Derive the replay buffer path that SB3's CheckpointCallback (with
    save_replay_buffer=True) would have saved alongside a given model
    checkpoint, e.g. 'checkpoints/dqn_cr_50000_steps.zip' ->
    'checkpoints/dqn_cr_replay_buffer_50000_steps.pkl'.

    Falls back to '<model_basename>_replay_buffer.pkl' for saves that don't
    follow the CheckpointCallback naming convention (e.g. the final model
    saved as 'dqn_cr_final').
    """
    directory, filename = os.path.split(model_path)
    base = filename[:-4] if filename.lower().endswith(".zip") else filename
    match = re.match(r"^(.*)_(\d+)_steps$", base)
    if match:
        name_prefix, steps = match.groups()
        buffer_name = f"{name_prefix}_replay_buffer_{steps}_steps.pkl"
    else:
        buffer_name = f"{base}_replay_buffer.pkl"
    return os.path.join(directory, buffer_name)


def train(resume_from=None):
    detect.SHOW_DETECTION_WINDOW = SHOW_DETECTION_WINDOW

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    env = ClashRoyaleEnv()

    checkpoint_callback = CheckpointCallback(
        save_freq=5_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="dqn_cr",
        save_replay_buffer=True,
    )

    if resume_from:
        model = DQN.load(resume_from, env=env)
        print(f"Resuming from {resume_from}")

        replay_buffer_path = _derive_replay_buffer_path(resume_from)
        if os.path.exists(replay_buffer_path):
            model.load_replay_buffer(replay_buffer_path)
            print(f"Loaded replay buffer from {replay_buffer_path}")
        else:
            print(f"No replay buffer found at {replay_buffer_path} — continuing with an empty buffer.")

        remaining_timesteps = TOTAL_TIMESTEPS - model.num_timesteps
        if remaining_timesteps <= 0:
            print(
                f"Resumed model already has {model.num_timesteps} timesteps, "
                f"which meets or exceeds TOTAL_TIMESTEPS ({TOTAL_TIMESTEPS}). Skipping training."
            )
            return
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
        remaining_timesteps = TOTAL_TIMESTEPS

    completed = False
    try:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=checkpoint_callback,
            reset_num_timesteps=resume_from is None,
        )
        completed = True
    except KeyboardInterrupt:
        print("Training interrupted by user — saving before exit.")
    except Exception as e:
        print(f"Training crashed: {e} — saving before exit.")
        raise
    finally:
        model.save(os.path.join(CHECKPOINT_DIR, "dqn_cr_final"))
        model.save_replay_buffer(os.path.join(CHECKPOINT_DIR, "dqn_cr_final_replay_buffer"))
        print("Model (and replay buffer) saved.")
        if SHOW_DETECTION_WINDOW:
            cv2.destroyAllWindows()

    if completed:
        print("Training complete. Final model saved.")


if __name__ == "__main__":
    import sys
    resume = sys.argv[1] if len(sys.argv) > 1 else None
    train(resume_from=resume)