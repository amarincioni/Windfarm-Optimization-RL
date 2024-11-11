def get_experiment_name(agent_name, env_name, privileged, mast_distancing, changing_wind, noise, training_steps):
    name = f"{agent_name}_{env_name}"
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
python run_experiment.py --agent_name {agent_name} --env_name {env_name} {privileged} {changing_wind} --mast_distancing {mast_distancing} --noise {noise}

echo "#### Finished experiment :)"
DATE=$(date)
echo "It is now $DATE"
"""