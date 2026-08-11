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
