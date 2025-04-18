# Windfarm Optimization RL
A reinforcement learning framework for wind farm optimization with quasi-dynamic wind conditions that implements:

- A quasi-dynamic wind farm environment using OpenAI Gym
- Training and evaluation of PPO RL agents via Stable Baselines3
- Comparison with baseline optimization methods
- Visualization and plotting scripts
- HPC scripts for parallel training and evaluation

It is based on the FLORIS Gym environment [wind-farm-env](https://github.com/AlgTUDelft/wind-farm-env) and extends it with a dynamic wind formulation based on the [FLORIDyn](https://doi.org/10.5194/wes-2021-154) model.

## Installation
The project was developed in Python 3.9. 
The environment to run the code in this repo can be installed using pip as following:
```bash
pip install -r requirements.txt
```
The code also requires a fork of the FLORIS Gym environment. The forked FLORIS Gym environment can be installed using the following command:
```bash
git clone https://github.com/amarincioni/wind-farm-env
```

## Usage
Different scripts are provided to run the code. They allow to generate envirnoment layouts, create scripts for HPC, train agents, evaluate baselines, plot results, and run experiments.
The main scripts are:
- `train_ppo.py`: This script is used to train PPO agents. It takes care of training the agents, evaluating them, and saving the results. It also allows to run the training in parallel on multiple environments. Training parameters can be set as command line arguments.
- `evaluate_baselines.py`: This script is used to evaluate the baseline methods. It computes the scores for the baseline methods across all the environments.
- `dynamic_windfarm_env.py`: This script contains the implementation of the dynamic wind farm environment. It is based on the FLORIS Gym environment and extends it with a dynamic wind formulation.

Notebooks are provided to run the experiments and visualize the results. The main notebooks are:

- `test_floris_sr.ipynb`: Runs preliminary experiments on FLORIS serial refine method.

- `evaluate_baselines.ipynb`: Computes performance scores for different baseline methods.

- `get_wandb_results.ipynb`: Downloads the performance scores for the PPO agent from Weights & Biases for further analysis.

- `test_computational_efficiency.ipynb`: Runs computational efficiency experiments to compare the speed of the quasi-dynamic environment.

- Various `plot_*.ipynb` notebooks: These can be used to visualize and plot results from the experiments. Different notebooks focus on specific aspects of the analysis and performance metrics.
