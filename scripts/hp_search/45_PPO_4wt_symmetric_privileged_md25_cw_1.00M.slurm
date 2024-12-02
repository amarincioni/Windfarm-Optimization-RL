#!/bin/bash
#SBATCH --job-name=wf
#SBATCH --output=output/%x_%j.out
#SBATCH --error=output/%x_%j.err
#SBATCH --mail-user="s3442209@vuw.leidenuniv.nl"
#SBATCH --mail-type="ALL"
#SBATCH --partition="gpu-medium"
#SBATCH --time=01-00:00:00
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
python run_experiment.py --agent_name PPO --env_name 4wt_symmetric --privileged --changing_wind --mast_distancing 25 --noise 0.0 --dynamic_mode None --learning_rate 5.6902963433101125e-05 --batch_size 256 --n_steps 5809 --experiment_name_prefix "hp_search/45"

echo "#### Finished experiment :)"
DATE=$(date)
echo "It is now $DATE"
