from env_utils import get_4wt_symmetric_env
from utils import get_experiment_name, initialize_wandb_run
from config import *

from stable_baselines3 import PPO

# Experiment parameters
agent = "ppo"
env = "4wt_symmetric"
privileged = True
changing_wind = True
mast_distancing = 75
noise = 0.05

# Initialize wandb run
experiment_name = get_experiment_name(agent, env, privileged, mast_distancing, changing_wind, noise, TRAINING_STEPS)
run, callback = initialize_wandb_run(experiment_name, agent, env, privileged, mast_distancing, changing_wind, noise, EVAL_REPS)
print(f"Experiment name: {experiment_name}")
print(f"Evaluations: {EVAL_REPS}")
print(f"Episode length: {EPISODE_LEN}")

# Training
env = get_4wt_symmetric_env(episode_length=EPISODE_LEN, privileged=privileged, changing_wind=changing_wind, mast_distancing=mast_distancing, noise=noise)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}")
model.learn(total_timesteps=TRAINING_STEPS, callback=callback, progress_bar=True)
print("Training done")

# Save model
model.save(f"data/models/{experiment_name}")
run.finish()