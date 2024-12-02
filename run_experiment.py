from env_utils import get_4wt_symmetric_env, get_lhs_env
from utils import get_experiment_name
from utils_wandb import initialize_wandb_run
from config import *

from stable_baselines3 import PPO, A2C, SAC#, TRPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
import argparse

if __name__ == "__main__":
    # Parse
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_name", type=str, help="Agent name")
    parser.add_argument("--env_name", type=str, help="Env name")
    parser.add_argument("--privileged", help="Enables privileged observations", action="store_true")
    parser.add_argument("--changing_wind", help="Enables changing wind environment", action="store_true")
    parser.add_argument("--mast_distancing", type=int, help="Sets mast distancing")
    parser.add_argument("--noise", type=float, help="Sets noise in observations")
    parser.add_argument("--dynamic_mode", type=str, help="Sets dynamic mode")
    parser.add_argument("--experiment_name_prefix", type=str, default="", help="Sets experiment name prefix")
    parser.add_argument("--learning_rate", type=float, default=0.0003, help="Sets learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Sets batch size")
    parser.add_argument("--n_steps", type=int, default=1024, help="Sets n_steps")
    #parser.add_argument("--load_pyglet", thelp="Enables the logging of videos and loads pyglet", action="store_true")
    args = parser.parse_args()
    print(args)

    # Filter out "None" dynamic mode
    if args.dynamic_mode not in ["momentum", "observation_points"]:
        args.dynamic_mode = None

    if "4wt_symmetric" in args.env_name:
        env_fn = lambda: get_4wt_symmetric_env(episode_length=EPISODE_LEN, privileged=args.privileged, changing_wind=args.changing_wind, mast_distancing=args.mast_distancing, noise=args.noise, dynamic_mode=args.dynamic_mode)
        # Non parallel implementation is faster
        eval_env_fn = lambda: get_4wt_symmetric_env(
            load_pyglet_visualization=True, parallel_dynamic_computations=False,
            episode_length=EPISODE_LEN, privileged=args.privileged, changing_wind=args.changing_wind, mast_distancing=args.mast_distancing, noise=args.noise, dynamic_mode=args.dynamic_mode)
        # eval_env_fn = env_fn
    elif "lhs" in args.env_name:
        env_fn = lambda: get_lhs_env(layout_name=args.env_name, episode_length=EPISODE_LEN, privileged=args.privileged, changing_wind=args.changing_wind, mast_distancing=args.mast_distancing, noise=args.noise, dynamic_mode=args.dynamic_mode)
        eval_env_fn = env_fn
    else:
        raise NotImplementedError(f"Environment {args.env_name} not implemented")

    # Initialize wandb run
    eval_freq = 50000 # 20 logs for 1000000 steps
    experiment_name = get_experiment_name(args.agent_name, args.env_name, args.privileged, args.mast_distancing, args.changing_wind, args.noise, args.dynamic_mode, TRAINING_STEPS, experiment_name_prefix=args.experiment_name_prefix)
    run, callback = initialize_wandb_run(experiment_name, args.agent_name, args.env_name, args.privileged, args.mast_distancing, args.changing_wind, args.noise, args.dynamic_mode, EVAL_REPS, env_fn=eval_env_fn, eval_freq=eval_freq)
    print(f"Experiment name: {experiment_name}")
    print(f"Evaluations: {EVAL_REPS}")
    print(f"Episode length: {EPISODE_LEN}")

    # Create the multiprocess environment
    N_ENVS_PARALLEL = 2
    env_list = [env_fn for _ in range(N_ENVS_PARALLEL)]
    # env = SubprocVecEnv(env_list, start_method="fork")
    env = env_list[0]()
    
    # Train the model
    if args.agent_name == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}", 
            learning_rate=args.learning_rate, n_steps=args.n_steps, batch_size=args.batch_size,
            # Small model
            #policy_kwargs={"net_arch": [dict(pi=[32, 32], vf=[32, 32])]}
            vf_coef=0.01,
        )
    elif args.agent_name == "A2C":
        model = A2C("MlpPolicy", env, tensorboard_log=f"runs/{run.id}", 
            #learning_rate=args.learning_rate, n_steps=args.n_steps, batch_size=args.batch_size,
            # Small model
            #policy_kwargs={"net_arch": [dict(pi=[32, 32], vf=[32, 32])]}
        )
    elif args.agent_name == "SAC":
        model = SAC("MlpPolicy", env, tensorboard_log=f"runs/{run.id}", 
            #learning_rate=args.learning_rate, n_steps=args.n_steps, batch_size=args.batch_size,
            # Small model
            #policy_kwargs={"net_arch": [dict(pi=[32, 32], vf=[32, 32])]}
        )
    elif args.agent_name == "TRPO":
        model = TRPO("MlpPolicy", env, tensorboard_log=f"runs/{run.id}", 
            #learning_rate=args.learning_rate, n_steps=args.n_steps, batch_size=args.batch_size,
            # Small model
            #policy_kwargs={"net_arch": [dict(pi=[32, 32], vf=[32, 32])]}
        )
    model.learn(total_timesteps=TRAINING_STEPS, callback=callback, progress_bar=True)
    print("Training done")

    # Save the model
    model.save(f"data/models/{experiment_name}")
    run.finish()