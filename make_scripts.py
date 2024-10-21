from utils import base_slurm_script, get_experiment_name
from config import TRAINING_STEPS

from dataclasses import dataclass

@dataclass
class Experiment:
    agent_name: str
    env_name: str
    privileged: bool
    changing_wind: bool
    mast_distancing: int
    noise: float
    training_steps: int = TRAINING_STEPS

experiments = [
    # 4wt symmetric experiments
    # Fixed wind experiments
    Experiment("PPO", "4wt_symmetric", False, False, -1, 0.0),
    Experiment("PPO", "4wt_symmetric", True, False, 25, 0.0),
    Experiment("PPO", "4wt_symmetric", True, False, 50, 0.0),
    Experiment("PPO", "4wt_symmetric", True, False, 75, 0.0),
    Experiment("PPO", "4wt_symmetric", True, False, 100, 0.0),
    Experiment("PPO", "4wt_symmetric", True, False, 125, 0.0),
    Experiment("PPO", "4wt_symmetric", True, False, 150, 0.0),
    # Changing wind experiments
    Experiment("PPO", "4wt_symmetric", False, True, -1, 0.0),
    Experiment("PPO", "4wt_symmetric", True, True, 25, 0.0),
    Experiment("PPO", "4wt_symmetric", True, True, 50, 0.0),
    Experiment("PPO", "4wt_symmetric", True, True, 75, 0.0),
    Experiment("PPO", "4wt_symmetric", True, True, 100, 0.0),
    Experiment("PPO", "4wt_symmetric", True, True, 125, 0.0),
    Experiment("PPO", "4wt_symmetric", True, True, 150, 0.0),
    # lhs16 experiments
    # Fixed wind experiments
    Experiment("PPO", "lhs_env_nt16_md75_wb1500x1500", False, False, -1, 0.0),
    Experiment("PPO", "lhs_env_nt16_md75_wb1500x1500", True, False, 200, 0.0),
    Experiment("PPO", "lhs_env_nt16_md75_wb1500x1500", True, False, 250, 0.0),
    Experiment("PPO", "lhs_env_nt16_md75_wb1500x1500", True, False, 300, 0.0),
    # Changing wind experiments
    Experiment("PPO", "lhs_env_nt16_md75_wb1500x1500", False, True, -1, 0.0),
    Experiment("PPO", "lhs_env_nt16_md75_wb1500x1500", True, True, 200, 0.0),
    Experiment("PPO", "lhs_env_nt16_md75_wb1500x1500", True, True, 250, 0.0),
    Experiment("PPO", "lhs_env_nt16_md75_wb1500x1500", True, True, 300, 0.0),
    # lhs8 experiments
    # Fixed wind experiments
    Experiment("PPO", "lhs_env_nt8_md150_wb750x750", False, False, -1, 0.0),
    Experiment("PPO", "lhs_env_nt8_md150_wb750x750", True, False, 100, 0.0),
    Experiment("PPO", "lhs_env_nt8_md150_wb750x750", True, False, 150, 0.0),
    Experiment("PPO", "lhs_env_nt8_md150_wb750x750", True, False, 200, 0.0),
    # Changing wind experiments
    Experiment("PPO", "lhs_env_nt8_md150_wb750x750", False, True, -1, 0.0),
    Experiment("PPO", "lhs_env_nt8_md150_wb750x750", True, True, 100, 0.0),
    Experiment("PPO", "lhs_env_nt8_md150_wb750x750", True, True, 150, 0.0),
    Experiment("PPO", "lhs_env_nt8_md150_wb750x750", True, True, 200, 0.0),

]
    

for experiment in experiments:

    privileged_option = "--privileged" if experiment.privileged else ""
    changing_wind_option = "--changing_wind" if experiment.changing_wind else ""

    # Set experiment duration
    if "4wt_symmetric" in experiment.env_name:
        node_type = "gpu-medium"
        days = 1
    elif "nt16" in experiment.env_name:
        node_type = "gpu-long"
        days = 4
    elif "nt8" in experiment.env_name:
        node_type = "gpu-long"
        days = 3

    script_txt = base_slurm_script.format(
        node_type=node_type,
        days=days,
        agent_name=experiment.agent_name,
        env_name=experiment.env_name,
        privileged=privileged_option,#
        changing_wind=changing_wind_option,#
        mast_distancing=experiment.mast_distancing,
        noise=experiment.noise,
    )

    experiment_name = get_experiment_name(
        agent_name=experiment.agent_name,
        env_name=experiment.env_name,
        privileged=experiment.privileged,
        changing_wind=experiment.changing_wind,
        mast_distancing=experiment.mast_distancing,
        noise=experiment.noise,
        training_steps=experiment.training_steps,
    )
    with open("scripts/" + experiment_name + ".slurm", "w") as f:
        f.write(script_txt)