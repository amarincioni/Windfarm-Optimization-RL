from utils import base_slurm_script, get_experiment_name
from config import TRAINING_STEPS

import numpy as np
from dataclasses import dataclass

@dataclass
class Experiment:
    agent_name: str
    env_name: str
    privileged: bool
    changing_wind: bool
    mast_distancing: int
    noise: float
    dynamic_mode: str = None
    training_steps: int = TRAINING_STEPS
    # Hyperparameter search
    experiment_name_prefix: str = ""
    learning_rate: float = 0.0001
    batch_size: int = 64
    n_steps: int = 2048

# Hyperparameter search bounds
learning_rates_exp = [4, 8]
batch_sizes = [6, 10]
n_steps = [1024, 8192]

experiments = []
for i in range(50):
    # sample hyperparameters
    # exponential sampling
    lr = 10 ** (-np.random.uniform(learning_rates_exp[0], learning_rates_exp[1]))
    bs = 2**np.random.randint(batch_sizes[0], batch_sizes[1])
    ns = np.random.randint(n_steps[0], n_steps[1])
    # print(f"lr: {lr}, bs: {bs}, ns: {ns}")
    experiments += [
        Experiment("PPO", "4wt_symmetric", False, True, -1, 0.0, 
            experiment_name_prefix=f"hp_search/{i}", learning_rate=lr, batch_size=bs, n_steps=ns),
        Experiment("PPO", "4wt_symmetric", True, True, 25, 0.0, 
            experiment_name_prefix=f"hp_search/{i}", learning_rate=lr, batch_size=bs, n_steps=ns),
    ]

# Test other models
agents = ["A2C", "SAC", "TRPO"]
for agent_name in agents:
    experiments += [
        Experiment(agent_name, "4wt_symmetric", False, True, -1, 0.0, experiment_name_prefix="test_models/"),
        Experiment(agent_name, "4wt_symmetric", True, True, 25, 0.0, experiment_name_prefix="test_models/"),
    ]

experiments += [
    # 4wt symmetric experiments
    #   -Fixed wind experiments
    Experiment("PPO", "4wt_symmetric", False, False, -1, 0.0),
    Experiment("PPO", "4wt_symmetric", True, False, 25, 0.0),
    Experiment("PPO", "4wt_symmetric", True, False, 50, 0.0),
    Experiment("PPO", "4wt_symmetric", True, False, 75, 0.0),
    Experiment("PPO", "4wt_symmetric", True, False, 100, 0.0),
    Experiment("PPO", "4wt_symmetric", True, False, 125, 0.0),
    Experiment("PPO", "4wt_symmetric", True, False, 150, 0.0),
    #   -Changing wind experiments
    Experiment("PPO", "4wt_symmetric", False, True, -1, 0.0),
    Experiment("PPO", "4wt_symmetric", True, True, 25, 0.0),
    Experiment("PPO", "4wt_symmetric", True, True, 50, 0.0),
    Experiment("PPO", "4wt_symmetric", True, True, 75, 0.0),
    Experiment("PPO", "4wt_symmetric", True, True, 100, 0.0),
    Experiment("PPO", "4wt_symmetric", True, True, 125, 0.0),
    Experiment("PPO", "4wt_symmetric", True, True, 150, 0.0),
    # lhs8 experiments
    #   -Fixed wind experiments
    Experiment("PPO", "lhs_env_nt8_md150_wb750x750", False, False, -1, 0.0),
    Experiment("PPO", "lhs_env_nt8_md150_wb750x750", True, False, 100, 0.0),
    Experiment("PPO", "lhs_env_nt8_md150_wb750x750", True, False, 150, 0.0),
    Experiment("PPO", "lhs_env_nt8_md150_wb750x750", True, False, 200, 0.0),
    #   -Changing wind experiments
    Experiment("PPO", "lhs_env_nt8_md150_wb750x750", False, True, -1, 0.0),
    Experiment("PPO", "lhs_env_nt8_md150_wb750x750", True, True, 100, 0.0),
    Experiment("PPO", "lhs_env_nt8_md150_wb750x750", True, True, 150, 0.0),
    Experiment("PPO", "lhs_env_nt8_md150_wb750x750", True, True, 200, 0.0),
    # lhs16 experiments
    #   -Fixed wind experiments
    Experiment("PPO", "lhs_env_nt16_md75_wb1500x1500", False, False, -1, 0.0),
    Experiment("PPO", "lhs_env_nt16_md75_wb1500x1500", True, False, 200, 0.0),
    Experiment("PPO", "lhs_env_nt16_md75_wb1500x1500", True, False, 250, 0.0),
    Experiment("PPO", "lhs_env_nt16_md75_wb1500x1500", True, False, 300, 0.0),
    #   -Changing wind experiments
    Experiment("PPO", "lhs_env_nt16_md75_wb1500x1500", False, True, -1, 0.0),
    Experiment("PPO", "lhs_env_nt16_md75_wb1500x1500", True, True, 200, 0.0),
    Experiment("PPO", "lhs_env_nt16_md75_wb1500x1500", True, True, 250, 0.0),
    Experiment("PPO", "lhs_env_nt16_md75_wb1500x1500", True, True, 300, 0.0),

    # Dynamic mode experiments
    # 4wt symmetric experiments
    #   -Changing wind experiments
    Experiment("PPO", "4wt_symmetric", False, True, -1, 0.0, "observation_points"),
    Experiment("PPO", "4wt_symmetric", True, True, 25, 0.0, "observation_points"),
    Experiment("PPO", "4wt_symmetric", True, True, 50, 0.0, "observation_points"),
    Experiment("PPO", "4wt_symmetric", True, True, 75, 0.0, "observation_points"),
    Experiment("PPO", "4wt_symmetric", True, True, 100, 0.0, "observation_points"),
    Experiment("PPO", "4wt_symmetric", True, True, 125, 0.0, "observation_points"),
    Experiment("PPO", "4wt_symmetric", True, True, 150, 0.0, "observation_points"),

    # lhs8 experiments
    #   -Changing wind experiments
    Experiment("PPO", "lhs_env_nt8_md150_wb750x750", False, True, -1, 0.0, "observation_points"),
    Experiment("PPO", "lhs_env_nt8_md150_wb750x750", True, True, 100, 0.0, "observation_points"),
    Experiment("PPO", "lhs_env_nt8_md150_wb750x750", True, True, 150, 0.0, "observation_points"),
    Experiment("PPO", "lhs_env_nt8_md150_wb750x750", True, True, 200, 0.0, "observation_points"),
    # lhs16 experiments
    #   -Changing wind experiments
    Experiment("PPO", "lhs_env_nt16_md75_wb1500x1500", False, True, -1, 0.0, "observation_points"),
    Experiment("PPO", "lhs_env_nt16_md75_wb1500x1500", True, True, 200, 0.0, "observation_points"),
    Experiment("PPO", "lhs_env_nt16_md75_wb1500x1500", True, True, 250, 0.0, "observation_points"),
    Experiment("PPO", "lhs_env_nt16_md75_wb1500x1500", True, True, 300, 0.0, "observation_points"),
]
    

for experiment in experiments:

    privileged_option = "--privileged" if experiment.privileged else ""
    changing_wind_option = "--changing_wind" if experiment.changing_wind else ""

    # Set experiment duration
    if False:
        node_type = "gpu-short"
        days = 0
        hours = "04"
    elif "4wt_symmetric" in experiment.env_name:
        node_type = "gpu-medium"
        days = 1
        hours = "00"
    elif "nt16" in experiment.env_name:
        node_type = "gpu-long"
        days = 4
        hours = "00"
    elif "nt8" in experiment.env_name:
        node_type = "gpu-long"
        days = 3
        hours = "00"

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
        experiment_name_prefix=experiment.experiment_name_prefix,
        learning_rate=experiment.learning_rate,
        batch_size=experiment.batch_size,
        n_steps=experiment.n_steps,
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