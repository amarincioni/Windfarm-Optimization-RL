import numpy as np

def get_circle_idxs(r):
    A = np.arange(-r,r+1)**2
    dists = np.sqrt(A[:,None] + A)
    return ((dists-r)<=0).astype(int)

def get_experiment_name(agent_name, env_name, privileged, mast_distancing, changing_wind, noise, dynamic_mode, training_steps, experiment_name_prefix=""):
    name = experiment_name_prefix + "_" if experiment_name_prefix != "" else ""
    name += f"{agent_name}_{env_name}"
    if dynamic_mode is not None:
        name += f"_{dynamic_mode}"
    if privileged:
        name += "_privileged" + f"_md{mast_distancing}"
    else:
        name += "_unprivileged"
    name = name + "_cw" if changing_wind else name
    if noise > 0:
        name += f"_n{noise}"
    name += f"_{training_steps/1e6:.2f}M"
    return name

base_slurm_script = """#!/bin/bash
#SBATCH --job-name=wf
#SBATCH --output=output/%x_%j.out
#SBATCH --error=output/%x_%j.err
#SBATCH --mail-user="s3442209@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="{node_type}"
#SBATCH --time=0{days}-{hours}:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gpus=1


# Starting logs
echo "#### Starting experiment"
echo "User: $SLURM_JOB_USER"
echo "Job ID: $SLURM_JOB_ID"
CWD=$(pwd)
DATE=$(date)
echo "This job was submitted from $SLURM_SUBMIT_DIR and I am currently in $CWD"
echo "It is now $DATE"

# Setup modules
ml purge
ml load shared DefaultModules ALICE/default gcc/11.2.0 slurm/alice/23.02.7 CUDA/11.8

# Setup conda environment
conda init bash
source ~/.bashrc
conda activate /home/s3442209/data1/Windfarm-Optimization-RL/conda_envs/windfarm_sb3_38

cd ..
CWD=$(pwd)
echo "Reached working directory $CWD"
echo "[$SHELL] Using GPU: "$CUDA_VISIBLE_DEVICES

### Actual experiment script
python train_ppo.py --agent_name {agent_name} --env_name {env_name} {privileged} {changing_wind} \
--mast_distancing {mast_distancing} --noise {noise} --dynamic_mode {dynamic_mode} \
--learning_rate {learning_rate} --batch_size {batch_size} --n_steps {n_steps} \
--experiment_name_prefix "{experiment_name_prefix}" --training_steps {training_steps} \
--gamma {gamma} --gae_lambda {gae_lambda} --clip_range {clip_range} \
--entropy_coefficient {entropy_coefficient} --vf_coefficient {vf_coefficient} --net_layers {net_layers}  --net_width {net_width} \
--n_epochs_ppo {n_epochs_ppo} --max_grad_norm {max_grad_norm}

echo "#### Finished experiment :)"
DATE=$(date)
echo "It is now $DATE"
"""

base_terminal_script = """python train_ppo.py --agent_name {agent_name} --env_name {env_name} {privileged} {changing_wind} \
--mast_distancing {mast_distancing} --noise {noise} --dynamic_mode {dynamic_mode} \
--learning_rate {learning_rate} --batch_size {batch_size} --n_steps {n_steps} \
--experiment_name_prefix "{experiment_name_prefix}" --training_steps {training_steps} \
--gamma {gamma} --gae_lambda {gae_lambda} --clip_range {clip_range} \
--entropy_coefficient {entropy_coefficient} --vf_coefficient {vf_coefficient} --net_layers {net_layers}  --net_width {net_width} \
--n_epochs_ppo {n_epochs_ppo} --max_grad_norm {max_grad_norm} --sb3_seed {sb3_seed}"""

base_slurm_sh_script = """#!/bin/bash
#SBATCH --job-name=wf
#SBATCH --output=output/%x_%j.out
#SBATCH --error=output/%x_%j.err
#SBATCH --mail-user="s3442209@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="{node_type}"
#SBATCH --time=0{days}-{hours}:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --gpus=1


# Starting logs
echo "#### Starting experiment"
echo "User: $SLURM_JOB_USER"
echo "Job ID: $SLURM_JOB_ID"
CWD=$(pwd)
DATE=$(date)
echo "This job was submitted from $SLURM_SUBMIT_DIR and I am currently in $CWD"
echo "It is now $DATE"

# Setup modules
ml purge
ml load shared DefaultModules ALICE/default gcc/11.2.0 slurm/alice/23.02.7 CUDA/11.8

# Setup conda environment
conda init bash
source ~/.bashrc
conda activate /home/s3442209/data1/Windfarm-Optimization-RL/conda_envs/windfarm_sb3_38

cd ..
CWD=$(pwd)
echo "Reached working directory $CWD"
echo "[$SHELL] Using GPU: "$CUDA_VISIBLE_DEVICES

"""

##### Evaluation utilities #####

import numpy as np
from config import *
import tqdm
from wind_processes import SetSequenceWindProcess
from PIL import Image

# Evaluation wind
EVAL_WIND_DIRECTIONS = np.load("data/eval/wind_directions.npy")
EVAL_WIND_SPEEDS = np.load("data/eval/wind_speeds.npy")
EVAL_FIXED_WIND_DIRECTIONS = [[EVAL_WIND_DIRECTIONS[i,0] for j in range(len(EVAL_WIND_DIRECTIONS[0]))]for i in range(len(EVAL_WIND_DIRECTIONS))]
EVAL_FIXED_WIND_SPEEDS = [[EVAL_WIND_SPEEDS[i,0] for j in range(len(EVAL_WIND_SPEEDS[0]))]for i in range(len(EVAL_WIND_SPEEDS))]

# List from 270        
EVAL_RENDER_WIND_DIRECTIONS = np.concatenate((
            np.ones((10))*270,
            np.linspace(270, 250, 20),
            np.linspace(250, 290, 40),
            np.linspace(290, 250, 20),
            np.linspace(250, 270, 11),
        ))
EVAL_RENDER_WIND_SPEEDS = np.ones_like(EVAL_RENDER_WIND_DIRECTIONS) * 8.0

# Render for fixed wind environment
EVAL_RENDER_FIXED_WIND_DIRECTIONS = np.ones_like(EVAL_RENDER_WIND_DIRECTIONS) * 270
EVAL_RENDER_FIXED_WIND_SPEEDS = np.ones_like(EVAL_RENDER_WIND_DIRECTIONS) * 8.0

def evaluate_model(env, agent_fn, wind_directions=EVAL_WIND_DIRECTIONS, wind_speeds=EVAL_WIND_SPEEDS, 
                   EVAL_REPS=EVAL_REPS, EPISODE_LEN=EPISODE_LEN, N_SEEDS=1, verbose=False):
    total_rewards = np.zeros((N_SEEDS, EVAL_REPS, EPISODE_LEN))
    total_powers = np.zeros((N_SEEDS, EVAL_REPS, EPISODE_LEN))
    for seed in range(N_SEEDS):
        for i in tqdm.tqdm(range(EVAL_REPS)):
            env.wind_process = SetSequenceWindProcess(wind_directions=wind_directions[i], wind_speeds=wind_speeds[i])
            env._np_random, env._seed = env.seed(seed)
            obs, info = env.reset()
            for j in range(EPISODE_LEN):
                action = agent_fn(env, obs)
                obs, reward, terminated, truncated, info = env.step(action)

                total_rewards[seed, i, j] = reward
                total_powers[seed, i, j] = info["power_output"]

                # if terminated or truncated:
                #     break
            if verbose:
                print("Episode {} finished with reward {}".format(i, np.sum(total_rewards[seed, i])))
                print("Episode {} finished with power {}".format(i, np.sum(total_powers[seed, i])))
            env.reset()
    return total_rewards, total_powers

def vectorized_evaluate_model(envs, model, wind_directions=EVAL_WIND_DIRECTIONS, wind_speeds=EVAL_WIND_SPEEDS, 
                   EVAL_REPS=EVAL_REPS, EPISODE_LEN=EPISODE_LEN, N_SEEDS=1, verbose=False):
    total_rewards = np.zeros((N_SEEDS, EVAL_REPS, EPISODE_LEN))
    total_powers = np.zeros((N_SEEDS, EVAL_REPS, EPISODE_LEN))
    for seed in range(N_SEEDS):
        #assert len(envs) == EVAL_REPS
        # for i in range(EVAL_REPS):
        ret = envs.env_method("seed", seed, indices=range(EVAL_REPS))
        # print(ret)
        for i in range(EVAL_REPS):
            envs.set_attr("_np_random", ret[i][0], indices=[i])
            envs.set_attr("_seed", ret[i][1], indices=[i])
            envs.set_attr("wind_process", SetSequenceWindProcess(wind_directions=wind_directions[i], wind_speeds=wind_speeds[i]), indices=[i])

        obs = envs.reset()
        for j in range(EPISODE_LEN):
            actions, states = model.predict(obs, deterministic=True)
            # print(actions)
            obs, rewards, done, info = envs.step(actions)
            # print(" INFO POWER: ", info)
            power_outputs = np.array([i["power_output"] for i in info])
            total_rewards[seed, :, j] = rewards
            total_powers[seed, :, j] = power_outputs
            # print(done)
            if done.any():
                print("Done, but this should not happen")
                break
        envs.reset()
    return total_rewards, total_powers

def render_model(env, agent_fn, wind_directions=EVAL_RENDER_WIND_DIRECTIONS, wind_speeds=EVAL_RENDER_WIND_SPEEDS, EPISODE_LEN=EPISODE_LEN):
    env.wind_process = SetSequenceWindProcess(wind_speeds=wind_speeds, wind_directions=wind_directions)
    obs, info = env.reset()
    video = []
    for j in range(EPISODE_LEN):
        action = agent_fn(env, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        f = env.render(mode="rgb_array")
        video.append(f)
    video = [Image.fromarray(img) for img in video]
    return video

def get_propSR_action(agent):
    return lambda env, obs: agent.predict(env.wind_process.wind_direction, env.yaws_from_wind)

def get_noisy_propSR_action(agent):
    return lambda env, obs: agent.predict(env.wind_process.wind_direction + np.random.normal(0, 5), env.yaws_from_wind)

def get_model_action(model):
    return lambda env, obs: model.predict(obs, deterministic=True)[0]

def get_nondeterministic_model_action(model):
    return lambda env, obs: model.predict(obs, deterministic=False)[0]

def get_naive_action():
    return lambda env, obs: 0

def get_random_action():
    return lambda env, obs: env.action_space.sample()