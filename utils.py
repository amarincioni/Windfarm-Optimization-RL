from stable_baselines3.common.callbacks import BaseCallback
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
import numpy as np
import matplotlib.pyplot as plt
import wandb
import tensorflow as tf

def get_experiment_name(agent, env, privileged, mast_distancing, changing_wind, noise, training_steps):
    name = f"{agent}_{env}"
    if privileged:
        name += "_privileged" + f"_md{mast_distancing}"
    else:
        name += "_unprivileged"
    name = name + "_cw" if changing_wind else name
    if noise > 0:
        name += f"_n{noise}"
    name += f"_t{training_steps}"
    return name

class VideoEvalCallback(BaseCallback):
    def __init__(self, freq=1000, eval_reps=10, verbose: int = 0):
        super().__init__(verbose)
        # self.model = None  # type: BaseAlgorithm
        # self.training_env # type: VecEnv
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # self.logger # type: stable_baselines3.common.logger.Logger
        self.freq = freq
        self.eval_reps = eval_reps

        self.n_logged = 0

    def _on_training_start(self) -> None:
        self._on_rollout_start()
        pass

    def _on_rollout_start(self) -> None:
        print(f"timestep {self.num_timesteps}, n_logged {self.n_logged}, freq {self.freq}, reps {self.eval_reps}")
        if self.num_timesteps >= (self.n_logged * self.freq):
            env = self.training_env.envs[0].unwrapped
            
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
                    
                    if i == 0:
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
            wandb.log({
                "eval/video": wandb.Video(np.transpose(np.array(video), (0,3,1,2)), fps=30, format="gif"), 
                "eval/total_power": total_power_rollout.mean(),
                "eval/total_power_std": total_power_rollout.std(),
                "eval/total_power_max": total_power_rollout.max(),
                "eval/total_power_min": total_power_rollout.min(),
                "eval/total_reward": total_reward_rollout.mean(),
                "eval/total_reward_std": total_reward_rollout.std(),
                "eval/total_reward_max": total_reward_rollout.max(),
                "eval/total_reward_min": total_reward_rollout.min(),
            })
            self.n_logged += 1
            print(f"Logged {self.n_logged} times")

            #env.visualization.close()
            #env.visualization = None

            obs, info = env.reset()

            return
        pass

    def _on_step(self) -> bool:
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
        eval_reps,
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
    callbacks = CallbackList([wandb_callback, VideoEvalCallback(freq=10000, eval_reps=eval_reps)])

    return run, callbacks