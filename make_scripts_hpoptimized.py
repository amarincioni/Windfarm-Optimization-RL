from utils import base_slurm_script, base_terminal_script, base_slurm_sh_script, get_experiment_name
from config import TRAINING_STEPS
import pathlib
import numpy as np
from dataclasses import dataclass
import dataclasses
import random

import pandas as pd

optimal_hp = pd.read_csv("data/eval/scores/hpsearch3_best_hp.csv")
print(optimal_hp)


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
    dynamic_mode: str = None
    noise: float = 0.0
    training_steps: int = TRAINING_STEPS / 10
    # Hyperparameter search
    experiment_name_prefix: str = ""
    learning_rate: float = 0.0001
    batch_size: int = 256
    n_steps: int = 2048
    gamma: float = 0.99
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

keys = ["learning_rate", "batch_size", "gamma", "train/clip_range", "gae_lambda",
        "ent_coef", "vf_coef", "net_layers", "net_width", "n_epochs", "max_grad_norm"]
experiment_keys = ["learning_rate", "batch_size", "gamma", "clip_range", "gae_lambda",
                   "entropy_coefficient", "vf_coefficient", "net_layers", "net_width", "n_epochs_ppo", "max_grad_norm"]
exp_params_keys = ["mast_distancing", "privileged", "changing_wind", "dynamic_mode"]

# iterate pd rows
experiments = []
for i, row in optimal_hp.iterrows():
    print(row["filtered_name"])
    hps = {exp_key: row[key] for exp_key, key in zip(experiment_keys, keys)}
    exp_params = {key: row[key] for key in exp_params_keys}
    exp_name = f"full_run_{row['filtered_name']}"
    exp_name = exp_name.replace("_0.10M", "")
    if exp_params["dynamic_mode"] != "observation_points":
        exp_params["dynamic_mode"] = None
    if "4wt" in exp_name:
        env_name = "4wt_symmetric"
    elif "nt8" in exp_name:
        env_name = "lhs_env_nt8_md150_wb750x750"
    elif "nt16" in exp_name:
        env_name = "lhs_env_nt16_md75_wb1500x1500"    
    print(exp_params)
    exp = Experiment(
        env_name=env_name,
        agent_name="PPO",
        experiment_name_prefix=exp_name, 
        **exp_params, 
        **hps,
    )

    # Cleaning data types
    for field in dataclasses.fields(exp):
        if field.name == "dynamic_mode":
            continue
        value = getattr(exp, field.name)
        if not isinstance(value, field.type):
            old = value
            setattr(exp, field.name, field.type(value))
            print(f"Converted {field.name} from {old} to {field.type(value)}")

    experiments.append(exp)

for e in experiments[:3]:
    # for key in hp_keys:
    values = [e.__dict__[key] for key in HP_SEARCH_BOUNDS.keys()]
    print(values)

pathlib.Path("scripts/full_run").mkdir(parents=True, exist_ok=True)
for rep_n in range(N_REPS):
    for experiment in experiments:
        privileged_option = "--privileged" if experiment.privileged else ""
        changing_wind_option = "--changing_wind" if experiment.changing_wind else ""

        if experiment.dynamic_mode is None:
            node_type, days, hours = ("cpu-medium", 1, "00")
        else:
            node_type, days, hours = ("cpu-long", 4, "00")

        with open(f"scripts/full_run/_slurm_run_{experiment.experiment_name_prefix}.slurm", "w") as f:
            text = base_slurm_sh_script.format(node_type=node_type, days=days, hours=hours)
            f.write(text + "\n")
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

            text = line = base_terminal_script.format(
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
                gae_lambda=experiment.gae_lambda,
                clip_range=experiment.clip_range,
                entropy_coefficient=experiment.entropy_coefficient,
                vf_coefficient=experiment.vf_coefficient,
                net_layers=experiment.net_layers,
                net_width=experiment.net_width,
                n_epochs_ppo=experiment.n_epochs_ppo,
                max_grad_norm=experiment.max_grad_norm,
                # Set seed for reproducibility
                sb3_seed=rep_n,
            )
            f.write(line + "\n")

print("Done!")