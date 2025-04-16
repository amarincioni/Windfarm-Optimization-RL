from utils import base_slurm_script, base_terminal_script, base_slurm_sh_script, get_experiment_name
from config import TRAINING_STEPS
import pathlib
import numpy as np
from dataclasses import dataclass
import random

N_HP_SEARCH = 10
N_REPS = 10

N_HP_SEARCH_SCRIPT = 10
N_REPS_SCRIPT = 2

@dataclass
class Experiment:
    agent_name: str
    env_name: str
    privileged: bool
    changing_wind: bool
    mast_distancing: int
    noise: float
    dynamic_mode: str = None
    training_steps: int = TRAINING_STEPS / 10
    # Hyperparameter search
    experiment_name_prefix: str = ""
    learning_rate: float = 0.0001
    batch_size: int = 256
    n_steps: int = 2048
    gamma: float = 0.99
    epsilon: float = 0.2        # SAME THING AS CLIPRANGE
    gae_lambda: float = 0.95
    clip_range: float = 0.2     # SAME THING AS EPSILON
    entropy_coefficient: float = 0.0
    vf_coefficient: float = 0.5
    net_layers: int = 2
    net_width: int = 256
    n_epochs_ppo: int = 10
    max_grad_norm: float = 0.5

# Hyperparameter search bounds
HP_SEARCH_BOUNDS = {
    "batch_size": [128, 256, 512],
    "net_layers": [2, 3, 4, 5],
    "net_width": [128, 256, 512],
    "n_epochs_ppo": [5, 10, 15],
}

HP_SEARCH_BOUNDS_WITH_RANGE = {
    "learning_rate": (0.00001, 0.001),
    "gamma": (0.8, 0.99),
    "gae_lambda": (0.9, 1.0),
    "clip_range": (0.1, 0.3),
    "entropy_coefficient": (0.0, 0.1),
    "vf_coefficient": (0.4, 0.6),
    "max_grad_norm": (0.25, 0.75),
}
def get_hp_params():
    disc_hp = {k: np.random.choice(v) for k, v in HP_SEARCH_BOUNDS.items()}
    cont_hp = {k: np.random.uniform(*v) for k, v in HP_SEARCH_BOUNDS_WITH_RANGE.items()}
    return {**disc_hp, **cont_hp}

mastdists = {
    "4wt_symmetric": [25, 75, 125], 
    "lhs_env_nt8_md150_wb750x750":[100,150,200],
    "lhs_env_nt16_md75_wb1500x1500": [200,250,300],
}
n_steps_options = {
    "": 2048,
    "fast_": 256,
}

experiments = []
exp_folder = "hp_search3"
np.random.seed(123)
pathlib.Path("scripts/" + exp_folder).mkdir(parents=True, exist_ok=True)
for training_steps, prefix in [(TRAINING_STEPS, ""), (TRAINING_STEPS / 10, "fast_")]:
    for i in range(N_HP_SEARCH):
        for envname in ["4wt_symmetric", "lhs_env_nt8_md150_wb750x750", "lhs_env_nt16_md75_wb1500x1500"]:
            mastdist = mastdists[envname]
            n_steps = n_steps_options[prefix]
            experiments += [
                Experiment("PPO", envname, False, False, -1, 0.0, 
                            experiment_name_prefix=f"{exp_folder}/{prefix}s{i}", n_steps=n_steps,
                            **get_hp_params(), training_steps=training_steps),
                Experiment("PPO", envname, False, True, -1, 0.0, 
                            experiment_name_prefix=f"{exp_folder}/{prefix}s{i}", n_steps=n_steps,
                            **get_hp_params(), training_steps=training_steps),
                Experiment("PPO", envname, False, True, -1, 0.0, 
                            experiment_name_prefix=f"{exp_folder}/{prefix}s{i}", n_steps=n_steps,
                                **get_hp_params(), training_steps=training_steps, dynamic_mode='observation_points'),
                Experiment("PPO", envname, True, True, mastdist[0], 0.0, 
                            experiment_name_prefix=f"{exp_folder}/{prefix}s{i}", n_steps=n_steps,
                                **get_hp_params(), training_steps=training_steps, dynamic_mode='observation_points'),
                Experiment("PPO", envname, True, True, mastdist[1], 0.0, 
                            experiment_name_prefix=f"{exp_folder}/{prefix}s{i}", n_steps=n_steps,
                                **get_hp_params(), training_steps=training_steps, dynamic_mode='observation_points'),
                Experiment("PPO", envname, True, True, mastdist[2], 0.0, 
                            experiment_name_prefix=f"{exp_folder}/{prefix}s{i}", n_steps=n_steps,
                                **get_hp_params(), training_steps=training_steps, dynamic_mode='observation_points'),
            ]

for e in experiments[:3]:
    # for key in hp_keys:
    values = [e.__dict__[key] for key in HP_SEARCH_BOUNDS.keys()]
    print(values)

for experiment in experiments:

    privileged_option = "--privileged" if experiment.privileged else ""
    changing_wind_option = "--changing_wind" if experiment.changing_wind else ""

    if "4wt_symmetric" in experiment.env_name:
        node_type = "cpu-medium"#"gpu-medium"
        days = 1
        hours = "00"
    elif "nt16" in experiment.env_name:
        node_type = "cpu-long"
        days = 4
        hours = "00"
    elif "nt8" in experiment.env_name:
        node_type = "cpu-long"
        days = 3
        hours = "00"
    
    if experiment.training_steps < TRAINING_STEPS or not experiment.changing_wind:
        node_type = "cpu-medium"
        days = 1
        hours = "00"

    # if experiment.dynamic_mode == "observation_points" and not ("4wt_symmetric" in experiment.env_name):
    #     node_type = "gpu-long"
    #     days = 5
    #     hours = "00"

    script_txt = base_slurm_script.format(
        node_type=node_type,
        days=days,
        hours=hours,
        agent_name=experiment.agent_name,
        env_name=experiment.env_name,
        privileged=privileged_option,#
        changing_wind=changing_wind_option,#
        mast_distancing=experiment.mast_distancing,
        noise=experiment.noise,
        dynamic_mode=experiment.dynamic_mode,
        # Hyperparameter search parameters
        training_steps=int(experiment.training_steps),
        n_steps=experiment.n_steps,
        experiment_name_prefix=experiment.experiment_name_prefix,
        learning_rate=experiment.learning_rate,
        batch_size=experiment.batch_size,
        gamma=experiment.gamma,
        epsilon=experiment.epsilon,
        gae_lambda=experiment.gae_lambda,
        clip_range=experiment.clip_range,
        entropy_coefficient=experiment.entropy_coefficient,
        vf_coefficient=experiment.vf_coefficient,
        net_layers=experiment.net_layers,
        n_epochs_ppo=experiment.n_epochs_ppo,
        max_grad_norm=experiment.max_grad_norm,
    )

    experiment_name = get_experiment_name(
        agent_name=experiment.agent_name,
        env_name=experiment.env_name,
        privileged=experiment.privileged,
        changing_wind=experiment.changing_wind,
        mast_distancing=experiment.mast_distancing,
        noise=experiment.noise,
        dynamic_mode=experiment.dynamic_mode,
        training_steps=experiment.training_steps,
        experiment_name_prefix=experiment.experiment_name_prefix,
    )
    with open("scripts/" + experiment_name + ".slurm", "w") as f:
        f.write(script_txt)


for rep_n in range(2):#N_REPS):
    with open(f"scripts/{exp_folder}/_run_all.sh", "w") as f:
        for experiment in experiments:
            privileged_option = "--privileged" if experiment.privileged else ""
            changing_wind_option = "--changing_wind" if experiment.changing_wind else ""
            experiment_name = get_experiment_name(
                agent_name=experiment.agent_name,
                env_name=experiment.env_name,
                privileged=experiment.privileged,
                changing_wind=experiment.changing_wind,
                mast_distancing=experiment.mast_distancing,
                noise=experiment.noise,
                dynamic_mode=experiment.dynamic_mode,
                training_steps=experiment.training_steps,
                experiment_name_prefix=experiment.experiment_name_prefix,
            )
            f.write(f"#{experiment_name}\n")
            line = base_terminal_script.format(
                agent_name=experiment.agent_name,
                env_name=experiment.env_name,
                privileged=privileged_option,#
                changing_wind=changing_wind_option,#
                mast_distancing=experiment.mast_distancing,
                noise=experiment.noise,
                dynamic_mode=experiment.dynamic_mode,
                # Hyperparameter search parameters
                training_steps=int(experiment.training_steps),
                n_steps=experiment.n_steps,
                experiment_name_prefix=experiment.experiment_name_prefix,
                learning_rate=experiment.learning_rate,
                batch_size=experiment.batch_size,
                gamma=experiment.gamma,
                epsilon=experiment.epsilon,
                gae_lambda=experiment.gae_lambda,
                clip_range=experiment.clip_range,
                entropy_coefficient=experiment.entropy_coefficient,
                vf_coefficient=experiment.vf_coefficient,
                net_layers=experiment.net_layers,
                n_epochs_ppo=experiment.n_epochs_ppo,
                max_grad_norm=experiment.max_grad_norm,
                # Set seed for reproducibility
                sb3_seed=rep_n,
            )
            f.write(line + "\n")

# Runs that are done
## Note that check for unprivileged passes unprivileged_cw as well
RUNS = [
    "PPO_4wt_symmetric_unprivileged",
    "PPO_4wt_symmetric_unprivileged_cw",
    "PPO_4wt_symmetric_observation_points_unprivileged_cw",
    "PPO_4wt_symmetric_observation_points_privileged_md25_cw",
    "PPO_lhs_env_nt8_md150_wb750x750_unprivileged",
    "PPO_lhs_env_nt8_md150_wb750x750_unprivileged_cw",
    "PPO_lhs_env_nt8_md150_wb750x750_observation_points_unprivileged_cw",
    "PPO_lhs_env_nt8_md150_wb750x750_observation_points_privileged_md100_cw",
    "PPO_lhs_env_nt16_md75_wb1500x1500_unprivileged",
    "PPO_lhs_env_nt16_md75_wb1500x1500_unprivileged_cw",
]
for search_n in range(N_HP_SEARCH_SCRIPT):
    for rep_n in range(N_REPS_SCRIPT):
        with open(f"scripts/{exp_folder}/_run_search{search_n}_rep{rep_n}.sh", "w") as f:
            for experiment in experiments:
                privileged_option = "--privileged" if experiment.privileged else ""
                changing_wind_option = "--changing_wind" if experiment.changing_wind else ""
                if f"fast_s{search_n}rep{rep_n}" in experiment.experiment_name_prefix:
                    experiment_name = get_experiment_name(
                        agent_name=experiment.agent_name,
                        env_name=experiment.env_name,
                        privileged=experiment.privileged,
                        changing_wind=experiment.changing_wind,
                        mast_distancing=experiment.mast_distancing,
                        noise=experiment.noise,
                        dynamic_mode=experiment.dynamic_mode,
                        training_steps=experiment.training_steps,
                        experiment_name_prefix=experiment.experiment_name_prefix,
                    )
                    
                    # Filter runs we are not doing
                    valid = False
                    for run in RUNS:
                        if run in experiment_name: valid = True
                    if not valid: continue

                    f.write(f"#{experiment_name}\n")
                    f.write(f"echo {experiment_name}\n")
                    line = base_terminal_script.format(
                        agent_name=experiment.agent_name,
                        env_name=experiment.env_name,
                        privileged=privileged_option,#
                        changing_wind=changing_wind_option,#
                        mast_distancing=experiment.mast_distancing,
                        noise=experiment.noise,
                        dynamic_mode=experiment.dynamic_mode,
                        # Hyperparameter search parameters
                        training_steps=int(experiment.training_steps),
                        n_steps=experiment.n_steps,
                        experiment_name_prefix=experiment.experiment_name_prefix,
                        learning_rate=experiment.learning_rate,
                        batch_size=experiment.batch_size,
                        gamma=experiment.gamma,
                        epsilon=experiment.epsilon,
                        gae_lambda=experiment.gae_lambda,
                        clip_range=experiment.clip_range,
                        entropy_coefficient=experiment.entropy_coefficient,
                        vf_coefficient=experiment.vf_coefficient,
                        net_layers=experiment.net_layers,
                        n_epochs_ppo=experiment.n_epochs_ppo,
                        max_grad_norm=experiment.max_grad_norm,
                        # Set seed for reproducibility
                        sb3_seed=rep_n,
                    )
                    f.write(line + "\n")
        with open(f"scripts/{exp_folder}/_slurm_run_search{search_n}_rep{rep_n}.slurm", "w") as f:
            text = base_slurm_sh_script.format(
                node_type="cpu-medium",
                days=1,
                hours="00",
            )
            f.write(text + "\n")
            f.write(f"chmod +x ./scripts/{exp_folder}/_run_search{search_n}_rep{rep_n}.sh\n")
            f.write(f"./scripts/{exp_folder}/_run_search{search_n}_rep{rep_n}.sh\n")
print("Done!")