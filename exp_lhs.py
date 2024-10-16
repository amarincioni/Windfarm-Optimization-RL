from env_utils import get_lhs_env, get_6wt_env, get_4wt_symmetric_env
from utils import get_experiment_name, VideoEvalCallback

from stable_baselines3 import PPO
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CallbackList, BaseCallback

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import wandb

EVAL_REPS = 10
EPISODE_LEN = 100
TRAINING_STEPS = 1e6
RUNS = TRAINING_STEPS / EPISODE_LEN

degree_step = 5
evaluations = 360 // degree_step

print(f"Will do {RUNS} runs")

# Training on a bigger environment
agent = "ppo"
env = "lhs_env_nt8_md150_wb750x750"
privileged = True
changing_wind = True
mast_distancing = int(env.split("_md")[1].split("_")[0])
experiment_name = get_experiment_name(agent, env, privileged, mast_distancing, changing_wind, TRAINING_STEPS)

config = {
    "experiment_name": experiment_name,
    "agent": agent,
    "env": env,
    "privileged": privileged,
    "changing_wind": changing_wind,
    "mast_distancing": mast_distancing,
}
run = wandb.init(
    project="thesis_tests",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=False,  # optional
    name=experiment_name,
)
wandb_callback = WandbCallback(
    gradient_save_freq=100,
    model_save_path=f"models/{run.id}",
    verbose=2,
)
callback = CallbackList([wandb_callback, VideoEvalCallback(freq=10000)])

print(f"Training {env} for {TRAINING_STEPS} steps, privileged, episode length {EPISODE_LEN}, changing wind")
print(f"Experiment name: {experiment_name}")

env = get_lhs_env(env, episode_length=EPISODE_LEN, privileged=privileged, changing_wind=changing_wind)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}")

model.learn(total_timesteps=TRAINING_STEPS, callback=callback, progress_bar=True)

print("Training done")

model.save(f"{experiment_name}_t{TRAINING_STEPS}")

run.finish()