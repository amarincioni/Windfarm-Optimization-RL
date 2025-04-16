from stable_baselines3.common.callbacks import BaseCallback
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import matplotlib.pyplot as plt
import wandb
import tensorflow as tf
import os
import time
from wind_processes import SetSequenceWindProcess
from utils import *
import multiprocessing

class VideoEvalCallback(BaseCallback):
    def __init__(self, freq=1000, eval_reps=10, experiment_name="", run_id="", verbose: int = 0, env_fn=None, changing_wind=True,
                 parallelize_evaluation=True):
        super().__init__(verbose)
        # self.model = None  # type: BaseAlgorithm
        # self.training_env # type: VecEnv
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # self.logger # type: stable_baselines3.common.logger.Logger
        self.freq = freq
        self.eval_reps = eval_reps
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.env_fn = env_fn
        self.is_changing_wind_experiment = changing_wind
        self.parallelize_evaluation = parallelize_evaluation

        self.n_logged = 0

        self.wind_direction_lists = np.load(f"data/eval/wind_directions.npy")
        self.wind_speed_lists = np.load(f"data/eval/wind_speeds.npy")
        self.val_wind_directions = np.load(f"data/eval/val_wind_directions.npy")
        self.val_wind_speeds = np.load(f"data/eval/val_wind_speeds.npy")
        assert len(self.wind_speed_lists) >= self.eval_reps, "Not enough wind sequences"
        self.wind_direction_lists = self.wind_direction_lists[:self.eval_reps]
        self.wind_speed_lists = self.wind_speed_lists[:self.eval_reps]
        #wind_directions = np.concatenate((np.ones((10))*270, np.linspace(290, 250, 20), np.linspace(250, 290, 20)))

        self.video_log_wind_directions = np.concatenate((
            np.ones((10))*270,
            np.linspace(270, 250, 20),
            np.linspace(250, 290, 40),
            np.linspace(290, 250, 20),
            np.linspace(250, 270, 10),
        ))
        self.video_log_wind_speeds = np.ones_like(self.video_log_wind_directions) * 8.0

        if not self.is_changing_wind_experiment:
            print("Fixed wind experiment detected. Will use fixed wind processes for evaluations!")
            # self.wind_direction_lists = np.tile(self.wind_direction_lists[:,0], (self.wind_direction_lists.shape[1], 1)).T
            # self.val_wind_directions = np.tile(self.val_wind_directions[:,0], (self.val_wind_directions.shape[1], 1)).T
            self.wind_direction_lists = [[self.wind_direction_lists[i,0] for j in range(len(self.wind_direction_lists[0]))]for i in range(len(self.wind_direction_lists))]
            self.wind_speed_lists = [[self.wind_speed_lists[i,0] for j in range(len(self.wind_speed_lists[0]))]for i in range(len(self.wind_speed_lists))]
            self.val_wind_directions = [[self.val_wind_directions[i,0] for j in range(len(self.val_wind_directions[0]))]for i in range(len(self.val_wind_directions))]
            self.val_wind_speeds = [[self.val_wind_speeds[i,0] for j in range(len(self.val_wind_speeds[0]))]for i in range(len(self.val_wind_speeds))]
        else:
            print("Changing wind environment detected.")
            
        if self.parallelize_evaluation:
            self.ft_eval_envs = [env_fn for _ in range(self.eval_reps)]
            print("Making parallel environments")
            self.vec_env = SubprocVecEnv(self.ft_eval_envs, start_method="spawn")
            print("Parallel environments created")
            # for i in range(self.eval_reps):
            #     self.vec_env.set_attr("wind_process", SetSequenceWindProcess(wind_directions=self.wind_direction_lists[i], wind_speeds=self.wind_speed_lists[i]))
        
        self.fixed_trajectory_eval_env = env_fn()

        # Initialize folders for logging the model weights periodically
        os.makedirs(os.path.dirname(f"data/models/run_checkpoints/{self.experiment_name}"), exist_ok=True)
        os.makedirs(os.path.dirname(f"data/models/run_checkpoints/{self.experiment_name}/{self.run_id}"), exist_ok=True)
        
        self.best_mean_reward = 0

    def _on_training_start(self) -> None:
        self._on_rollout_start()
        pass

    def _on_rollout_start(self) -> None:
        print(f"timestep {self.num_timesteps}, n_logged {self.n_logged}, freq {self.freq}, reps {self.eval_reps}")
        
        if self.num_timesteps >= (self.n_logged * self.freq):
            print("Running evaluation")
            t0 = time.time()
            
            env = self.fixed_trajectory_eval_env
            EPISODE_LEN = env.episode_length

            # Run a rollout saving a video and performance
            total_rewards = np.zeros((self.eval_reps, EPISODE_LEN,))
            total_powers = np.zeros((self.eval_reps, EPISODE_LEN,))

            # Video rollout
            if env.load_pyglet_visualization:
                video = render_model(env, get_model_action(self.model), wind_directions=self.video_log_wind_directions, wind_speeds=self.video_log_wind_speeds, EPISODE_LEN=EPISODE_LEN)
                video = [np.array(img) for img in video]
            t_after_video = time.time()

            if self.parallelize_evaluation:
                # Parallelize example sum]
                # from stable_baselines3.common.evaluation import evaluate_policy
                # rewards, lens = evaluate_policy(self.model, self.vec_env, n_eval_episodes=1, deterministic=True, return_episode_rewards=True)
                # print(rewards, lens)
                # 1/0
                total_rewards, total_powers = vectorized_evaluate_model(self.vec_env, self.model, 
                    wind_directions=self.wind_direction_lists, wind_speeds=self.wind_speed_lists,
                    EVAL_REPS=self.eval_reps, EPISODE_LEN=EPISODE_LEN, N_SEEDS=1, verbose=False
                )
                total_val_rewards, total_val_powers = vectorized_evaluate_model(self.vec_env, self.model, 
                    wind_directions=self.val_wind_directions, wind_speeds=self.val_wind_speeds,
                    EVAL_REPS=self.eval_reps, EPISODE_LEN=EPISODE_LEN, N_SEEDS=1, verbose=False
                )
            else:
                total_rewards, total_powers = evaluate_model(env, get_model_action(self.model), 
                    wind_directions=self.wind_direction_lists, wind_speeds=self.wind_speed_lists,
                    EVAL_REPS=self.eval_reps, EPISODE_LEN=EPISODE_LEN, N_SEEDS=1, verbose=False)
                total_val_rewards, total_val_powers = evaluate_model(env, get_model_action(self.model), 
                    wind_directions=self.val_wind_directions, wind_speeds=self.val_wind_speeds,
                    EVAL_REPS=self.eval_reps, EPISODE_LEN=EPISODE_LEN, N_SEEDS=1, verbose=False)
            assert total_rewards.shape[0] == 1, "Evaluation with multiple seeds not implemented" 
            total_rewards = total_rewards[0]
            total_powers = total_powers[0]
            total_val_rewards = total_val_rewards[0]
            total_val_powers = total_val_powers[0]
            # Get rollout sums
            total_reward_rollout = np.sum(total_rewards, axis=1)
            total_power_rollout = np.sum(total_powers, axis=1)
            total_val_reward_rollout = np.sum(total_val_rewards, axis=1)
            total_val_power_rollout = np.sum(total_val_powers, axis=1)
            
            # Log metrics
            logs_dict = {
                "eval/total_power": total_power_rollout.mean(),
                "eval/total_power_std": total_power_rollout.std(),
                "eval/total_power_max": total_power_rollout.max(),
                "eval/total_power_min": total_power_rollout.min(),
                "eval/total_reward": total_reward_rollout.mean(),
                "eval/total_reward_std": total_reward_rollout.std(),
                "eval/total_reward_max": total_reward_rollout.max(),
                "eval/total_reward_min": total_reward_rollout.min(),
                "eval/validation_total_power": total_val_power_rollout.mean(),
                "eval/validation_total_power_std": total_val_power_rollout.std(),
                "eval/validation_total_power_max": total_val_power_rollout.max(),
                "eval/validation_total_power_min": total_val_power_rollout.min(),
                "eval/validation_total_reward": total_val_reward_rollout.mean(),
                "eval/validation_total_reward_std": total_val_reward_rollout.std(),
                "eval/validation_total_reward_max": total_val_reward_rollout.max(),
                "eval/validation_total_reward_min": total_val_reward_rollout.min(),
            }
            # Log video if available
            if env.load_pyglet_visualization:
                logs_dict = {
                    **logs_dict, 
                    **{"eval/video": wandb.Video(np.transpose(np.array(video), (0,3,1,2)), fps=30, format="gif"), }
                }
            wandb.log(logs_dict)

            # Saving the model weights (for future visualizations or use)
            self.model.save(f"data/models/run_checkpoints/{self.experiment_name}/{self.run_id}/ckpt_{self.n_logged}")
            if np.mean(total_val_power_rollout) > self.best_mean_reward:
                self.best_mean_reward = np.mean(total_val_power_rollout)
                self.model.save(f"data/models/run_checkpoints/{self.experiment_name}/{self.run_id}/ckpt_best_mean_reward")
                
            # Update logging variables
            self.n_logged += 1
            print(f"Logged {self.n_logged} times")

            # Print time elapsed
            time_elapsed = time.time()-t0
            time_elapsed_no_video = time.time()-t_after_video
            print(f"Evaluation took {time_elapsed} seconds ({time_elapsed/self.eval_reps:.2f} per rollout and {time_elapsed/EPISODE_LEN/self.eval_reps:.2f} per timestep)")
            print(f"Evaluation took {time_elapsed_no_video} seconds without video ({time_elapsed_no_video/self.eval_reps:.2f} per rollout and {time_elapsed_no_video/EPISODE_LEN/self.eval_reps:.2f} per timestep)")
            # Reset env (may be unnecessary)
            obs, info = env.reset()

            return
        pass

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        self.model.save(f"data/models/run_checkpoints/{self.experiment_name}/{self.run_id}/ckpt_end")
        pass

def get_env_rewards(env, model, wind_process):
    EPISODE_LEN = env.episode_length
    env.wind_process = wind_process
    # Run a rollout saving a video and performance
    total_rewards = np.zeros((EPISODE_LEN,))
    total_powers = np.zeros((EPISODE_LEN,))
    obs, info = env.reset()
    for j in range(EPISODE_LEN):
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_rewards[j] = reward
        total_powers[j] = info["power_output"]
        if terminated or truncated:
            break
    return total_rewards, total_powers

class WandbLogBestCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        wandb.log({"eval/best_mean_reward": self.parent.best_mean_reward}) # TODO commit=False)
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass

def initialize_wandb_run(
        experiment_name, 
        agent, 
        env, 
        privileged, 
        mast_distancing, 
        changing_wind,
        noise,
        dynamic_mode,
        eval_reps,
        eval_freq=10000,
        env_fn=None,
        n_envs=16,
        net_layers=2,
        net_width=256,
        ):
    config = {
        "experiment_name": experiment_name,
        "agent": agent,
        "env": env,
        "privileged": privileged,
        "changing_wind": changing_wind,
        "mast_distancing": mast_distancing,
        "noise": noise,
        "dynamic_mode": dynamic_mode,
        "eval_reps": eval_reps,
        "n_envs": n_envs,
        "net_layers": net_layers,
        "net_width": net_width,
    }
    run = wandb.init(
        project="thesis_tests",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=False,  # optional
        name=experiment_name,
    )
    wandb_callback = WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    )

    real_eval_freq = max(eval_freq//n_envs, 1)

    video_eval_callback = VideoEvalCallback(
        freq=real_eval_freq,
        eval_reps=eval_reps,
        experiment_name=experiment_name,
        run_id=run.id,
        env_fn=env_fn,
        changing_wind=changing_wind,
    )
    callbacks = [wandb_callback, video_eval_callback]
    # eval_callback = EvalCallback(env_fn(), best_model_save_path=f"models/{run.id}/",
    #     log_path=f"models/{run.id}/", eval_freq=eval_freq,
    #     deterministic=True, render=False)
    # callbacks += [eval_callback]
    callbacks = CallbackList(callbacks)

    return run, callbacks