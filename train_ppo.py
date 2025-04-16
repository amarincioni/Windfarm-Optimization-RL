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
    parser.add_argument("--training_steps", type=int, default=TRAINING_STEPS, help="Sets training steps")
    parser.add_argument("--device", type=str, default="cpu", help="Sets device")
    parser.add_argument("--experiment_name_prefix", type=str, default="", help="Sets experiment name prefix")
    parser.add_argument("--n_steps", type=int, default=1024, help="Sets n_steps")
    parser.add_argument("--learning_rate", type=float, default=0.0003, help="Sets learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Sets batch size")
    # New ones, not necessarily default of sb3
    parser.add_argument("--gamma", type=float, default=0.99, help="Sets gamma")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="Sets gae_lambda")
    parser.add_argument("--clip_range", type=float, default=0.2, help="Sets clip_range")
    parser.add_argument("--entropy_coefficient", type=float, default=0.0, help="Sets entropy_coefficient")
    parser.add_argument("--vf_coefficient", type=float, default=0.5, help="Sets vf_coefficient")
    parser.add_argument("--net_layers", type=int, default=2, help="Sets net layers")
    parser.add_argument("--net_width", type=int, default=256, help="Sets net width")
    parser.add_argument("--n_epochs_ppo", type=int, default=10, help="Sets n_epochs_ppo")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Sets max_grad_norm")
    parser.add_argument("--sb3_seed", type=int, default=0, help="Sets seed")
    #parser.add_argument("--load_pyglet", thelp="Enables the logging of videos and loads pyglet", action="store_true")
    args = parser.parse_args()
    print(args)

    # Filter out "None" dynamic mode
    if args.dynamic_mode not in ["momentum", "observation_points"]:
        args.dynamic_mode = None

    if "4wt_symmetric" in args.env_name:
        env_fn = lambda: get_4wt_symmetric_env(episode_length=EPISODE_LEN, privileged=args.privileged, changing_wind=args.changing_wind, mast_distancing=args.mast_distancing, noise=args.noise, dynamic_mode=args.dynamic_mode)
        # Non parallel implementation is faster
        # eval_env_fn = lambda: get_4wt_symmetric_env(
        #     load_pyglet_visualization=True, parallel_dynamic_computations=False,
        #     episode_length=EPISODE_LEN, privileged=args.privileged, changing_wind=args.changing_wind, mast_distancing=args.mast_distancing, noise=args.noise, dynamic_mode=args.dynamic_mode)
        eval_env_fn = lambda: get_4wt_symmetric_env(episode_length=EPISODE_LEN, privileged=args.privileged, 
            changing_wind=True, mast_distancing=args.mast_distancing, noise=args.noise, dynamic_mode=args.dynamic_mode)
    elif "lhs" in args.env_name:
        env_fn = lambda: get_lhs_env(layout_name=args.env_name, episode_length=EPISODE_LEN, privileged=args.privileged, changing_wind=args.changing_wind, mast_distancing=args.mast_distancing, noise=args.noise, dynamic_mode=args.dynamic_mode)
        eval_env_fn = lambda: get_lhs_env(layout_name=args.env_name, episode_length=EPISODE_LEN, privileged=args.privileged, 
            changing_wind=True, mast_distancing=args.mast_distancing, noise=args.noise, dynamic_mode=args.dynamic_mode)
    else:
        raise NotImplementedError(f"Environment {args.env_name} not implemented")
    
    # Create the multiprocess environment
    # if ("_nt16_" in args.env_name) or (args.dynamic_mode is not None): # 
    #     N_ENVS_PARALLEL = 4
    # elif ("_nt8_" in args.env_name and args.dynamic_mode is not None):
    #     N_ENVS_PARALLEL = 2
    # else:
    if ("_nt8_" in args.env_name and args.dynamic_mode is not None):
        N_ENVS_PARALLEL = 8
    else:
        N_ENVS_PARALLEL = 16
    env_list = [env_fn for _ in range(N_ENVS_PARALLEL)]
    env = SubprocVecEnv(env_list, start_method="fork")
    #env = env_list[0]()

    # Initialize wandb run
    eval_freq = args.training_steps // 10 # 50000 = 20 logs for 1000000 steps
    print(f"Eval freq: {eval_freq}")
    experiment_name = get_experiment_name(args.agent_name, args.env_name, args.privileged, args.mast_distancing, args.changing_wind, args.noise, args.dynamic_mode, args.training_steps, experiment_name_prefix=args.experiment_name_prefix)
    run, callback = initialize_wandb_run(experiment_name, args.agent_name, args.env_name, args.privileged, 
                                         args.mast_distancing, args.changing_wind, args.noise, args.dynamic_mode, 
                                         EVAL_REPS, env_fn=eval_env_fn, eval_freq=eval_freq, net_layers=args.net_layers,
                                         net_width=args.net_width)
    print(f"Experiment name: {experiment_name}")
    print(f"Evaluations: {EVAL_REPS}")
    print(f"Episode length: {EPISODE_LEN}")

    
    net = dict(pi = [args.net_width for _ in range(args.net_layers)], vf = [args.net_width for _ in range(args.net_layers)])

    # Train the model
    if args.agent_name == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"runs/{run.id}", 
            device=args.device,
            n_steps=args.n_steps, 
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.entropy_coefficient,
            vf_coef=args.vf_coefficient,
            n_epochs=args.n_epochs_ppo,
            max_grad_norm=args.max_grad_norm,
            policy_kwargs={"net_arch": net},
            # vf_coef=0.01,
            seed=args.sb3_seed,
        )
    # elif args.agent_name == "A2C":
    #     model = A2C("MlpPolicy", env, tensorboard_log=f"runs/{run.id}", 
    #         device=args.device,
    #         #learning_rate=args.learning_rate, n_steps=args.n_steps, batch_size=args.batch_size,
    #         # Small model
    #         #policy_kwargs={"net_arch": [dict(pi=[32, 32], vf=[32, 32])]}
    #     )
    # elif args.agent_name == "SAC":
    #     model = SAC("MlpPolicy", env, tensorboard_log=f"runs/{run.id}", 
    #         device=args.device,
    #         #learning_rate=args.learning_rate, n_steps=args.n_steps, batch_size=args.batch_size,
    #         # Small model
    #         #policy_kwargs={"net_arch": [dict(pi=[32, 32], vf=[32, 32])]}
    #     )
    # elif args.agent_name == "TRPO":
    #     model = TRPO("MlpPolicy", env, tensorboard_log=f"runs/{run.id}", 
    #         device=args.device,
    #         #learning_rate=args.learning_rate, n_steps=args.n_steps, batch_size=args.batch_size,
    #         # Small model
    #         #policy_kwargs={"net_arch": [dict(pi=[32, 32], vf=[32, 32])]}
    #     )
    model.learn(total_timesteps=args.training_steps, callback=callback, progress_bar=True)
    print("Training done")

    # Save the model
    model.save(f"data/models/{experiment_name}")
    run.finish()