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
]
    

for experiment in experiments:

    script_txt = base_slurm_script.format(
        agent_name=experiment.agent_name,
        env_name=experiment.env_name,
        privileged=experiment.privileged,
        changing_wind=experiment.changing_wind,
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