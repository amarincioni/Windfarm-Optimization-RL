from stable_baselines3.common.callbacks import BaseCallback
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, EvalCallback
import numpy as np
import matplotlib.pyplot as plt
import wandb
import tensorflow as tf
import os

class VideoEvalCallback(BaseCallback):
    def __init__(self, freq=1000, eval_reps=10, experiment_name="", run_id="", verbose: int = 0, env_fn=None):
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

        self.n_logged = 0
        
        # Initialize folders for logging the model weights periodically
        os.makedirs(os.path.dirname(f"data/models/run_checkpoints/{self.experiment_name}"), exist_ok=True)
        os.makedirs(os.path.dirname(f"data/models/run_checkpoints/{self.experiment_name}/{self.run_id}"), exist_ok=True)

    def _on_training_start(self) -> None:
        self._on_rollout_start()
        pass

    def _on_rollout_start(self) -> None:
        print(f"timestep {self.num_timesteps}, n_logged {self.n_logged}, freq {self.freq}, reps {self.eval_reps}")
        
        if self.num_timesteps >= (self.n_logged * self.freq):
            
            if self.env_fn is None:
                env = self.training_env.envs[0].unwrapped
            else:
                env = self.env_fn()
            
            EPISODE_LEN = env.episode_length
            # Run a rollout saving a video and performance
            total_rewards = np.zeros((self.eval_reps, EPISODE_LEN,))
            total_powers = np.zeros((self.eval_reps, EPISODE_LEN,))
            video = []
            for i in range(self.eval_reps):
                obs, info = env.reset()
                
                # Set wind to 280 degrees
                env.wind_process.wind_direction = 280
                env.wind_process.wind_speed = 8.0
                
                for j in range(EPISODE_LEN):
                    action, _states = self.model.predict(obs)

                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    if i == 0 and env.load_pyglet_visualization:
                        try:
                            video.append(env.render(mode="rgb_array"))
                        except:
                            video.append(np.zeros_like(video[0]))
                    
                    total_rewards[i,j] = reward
                    total_powers[i,j] = info["power_output"]
                    if terminated or truncated:
                        break
            
            total_power_rollout = np.sum(total_powers, axis=1)
            total_reward_rollout = np.sum(total_rewards, axis=1)
            # Log without progressing step count
            logs_dict = {
                "eval/total_power": total_power_rollout.mean(),
                "eval/total_power_std": total_power_rollout.std(),
                "eval/total_power_max": total_power_rollout.max(),
                "eval/total_power_min": total_power_rollout.min(),
                "eval/total_reward": total_reward_rollout.mean(),
                "eval/total_reward_std": total_reward_rollout.std(),
                "eval/total_reward_max": total_reward_rollout.max(),
                "eval/total_reward_min": total_reward_rollout.min(),
            }
            if env.load_pyglet_visualization:
                logs_dict = {
                    **logs_dict, 
                    **{"eval/video": wandb.Video(np.transpose(np.array(video), (0,3,1,2)), fps=30, format="gif"), }
                }
            wandb.log(logs_dict)

            # Saving the model weights (for future visualizations or use)
            self.model.save(f"data/models/run_checkpoints/{self.experiment_name}/{self.run_id}/ckpt_{self.n_logged}")
                
            # Update logging variables
            self.n_logged += 1
            print(f"Logged {self.n_logged} times")

            # Reset env
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

def initialize_wandb_run(
        experiment_name, 
        agent, 
        env, 
        privileged, 
        mast_distancing, 
        changing_wind,
        noise,
        eval_reps,
        eval_freq=10000,
        env_fn=None,
        n_envs=16,
        ):
    config = {
        "experiment_name": experiment_name,
        "agent": agent,
        "env": env,
        "privileged": privileged,
        "changing_wind": changing_wind,
        "mast_distancing": mast_distancing,
        "noise": noise,
        "eval_reps": eval_reps,
        "n_envs": n_envs,
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
    video_eval_callback = VideoEvalCallback(
        freq=eval_freq,
        eval_reps=eval_reps,
        experiment_name=experiment_name,
        run_id=run.id,
        env_fn=env_fn,
    )
    eval_callback = EvalCallback(env_fn(), best_model_save_path=f"models/{run.id}/",
        log_path=f"models/{run.id}/", eval_freq=eval_freq,
        deterministic=True, render=False)
    callbacks = CallbackList([wandb_callback, video_eval_callback, eval_callback])

    return run, callbacks