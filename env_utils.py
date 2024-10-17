from wind_farm_gym import WindFarmEnv
from wind_farm_gym.wind_process.wind_process import WindProcess
import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt

def get_grid_points(w, h, dist):
    x = np.linspace(0, w, w//dist).astype(int)
    y = np.linspace(0, h, h//dist).astype(int)
    xv, yv = np.meshgrid(x,y)
    xv, yv = xv.flatten(), yv.flatten()
    return (xv, yv)

def get_lhs_points(n, bounds=[1500, 1500]):
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=2)
    sample = sampler.random(n=n)
    sample = sample * np.array(bounds)
    x, y = sample[:,0], sample[:,1]
    return (x, y)

# Changes the wind speed and direction randomly at each reset
class RandomResetWindProcess(WindProcess):
    def __init__(self, sorted_wind=False, wind_speed=None, changing_wind=False):
        self.sorted_wind = sorted_wind
        self.set_wind_speed = wind_speed
        self.changing_wind = changing_wind

        self.wind_speed = 8
        self.wind_direction = 359

        assert not (self.sorted_wind and self.changing_wind), "Cannot have both sorted and changing wind"
    
    def step(self):

        if self.changing_wind:
            # Randomly change the wind speed and direction slightly
            self.wind_speed += np.random.uniform(-1, 1)
            self.wind_direction += np.random.uniform(-3, 3)

            # Clip values
            self.wind_speed = np.clip(self.wind_speed, 0, 25)
            self.wind_direction = int(self.wind_direction) % 360

        return {'wind_speed': self.wind_speed, 'wind_direction': self.wind_direction}
    
    def reset(self):
        if self.set_wind_speed is not None:
            self.wind_speed = self.set_wind_speed
        else:
            self.wind_speed = np.random.uniform(0, 25)

        if self.sorted_wind:
            self.wind_direction = (self.wind_direction + 1) % 360
        else:
            self.wind_direction = int(np.random.uniform(0, 360)) 
            # 297 is correct, 296 works, 297 does not
            # 154 is ok, 153 is not
            # so input range is [154,297)
        return self.step()

#class modified env inherits from windfarmenv
class modified_env(WindFarmEnv):
    def __init__(self, 
        turbine_layout, 
        mast_layout, 
        floris, 
        episode_length=10, 
        sorted_wind=False,      # To run evaluation
        wind_speed=None,        # To run evaluation
        lidar_observations=None,
        changing_wind=False,    # More difficult environment that requries actual control
        action_representation='wind',
        observation_noise=0.0,
        verbose=False,
        load_pyglet_visualization=False,
        ):

        # Setting the wind state
        self.sorted_wind = sorted_wind
        self.changing_wind = changing_wind

        # Changes wind direction and power at every reset
        # Changes the wind at each step if enabled
        self.wind_process = RandomResetWindProcess(
            sorted_wind=sorted_wind, 
            wind_speed=wind_speed,
            changing_wind=changing_wind,
        )

        # Building the environment
        super().__init__(
            turbine_layout=turbine_layout, 
            mast_layout=mast_layout, 
            floris=floris, 
            observe_yaws=True,
            wind_process=self.wind_process,
            lidar_observations=lidar_observations,
            action_representation=action_representation,
            load_pyglet_visualization=load_pyglet_visualization,
        )

        self.episode_length = episode_length
        self.step_count = 0
        self.runs = 0
        self.observation_noise = observation_noise
        self.verbose = verbose

    # New adaptation for gymnasium
    # Reset takes a seed as input and returns the observation 
    def reset(self, **kwargs):
        obs = super().reset()

        # Initialize renderable directions
        #self.wind_process.wind_direction = 155 + (int(self.runs * (295-155)/8) % (295-155))
        
        # Update environment after reset (wind speed and direction are logged and used later, so this is necessary)
        self.floris_interface.reinitialize_flow_field(wind_speed=self.wind_process.wind_speed, wind_direction=self.wind_process.wind_direction)

        # Initialize turbine directions randomly, but towards the wind
        for turbine in self.turbines:
            turbine.yaw_angle = self._np_random.uniform(self.desired_min_yaw, self.desired_max_yaw)
            #turbine.yaw_angle = 0


        self.step_count = 0
        self.runs += 1
        if self.verbose:
            print("Run count:", self.runs)
            print("Wind speed:", self.wind_process.wind_speed)
            print("Wind direction:", self.wind_process.wind_direction)
        return obs, None
    
    # New adaptation for gymnasium
    # Step now splits done into terminated and truncated
    def step(self, action):
        obs, reward, done, info = super().step(action)
        #print("Step count:", self.step_count)	
        power_output = np.sum(self.floris_interface.get_turbine_power())
        info['power_output'] = np.nan_to_num(power_output, nan=0)

        self.step_count += 1

        if any([np.isnan(x) for x in obs]):
            obs = np.nan_to_num(obs, nan=0)
            
        terminated, truncated = False, False
        if done: terminated, truncated = True, False
        elif self.step_count >= self.episode_length: terminated, truncated = True, False

        if self.observation_noise > 0:
            obs = obs + np.random.normal(0, self.observation_noise, len(obs))

        return obs, reward, terminated, truncated, info



def get_6wt_env(episode_length=10, privileged=True, mast_distancing=50):
    turbine_layout = ([0, 750, 1500, 0, 750, 1500], [0, 0, 0, 500, 500, 500])
    w, h = np.max(turbine_layout, axis=1)

    mast_layout = get_grid_points(w, h, mast_distancing) if privileged else None

    env = modified_env(
        turbine_layout=turbine_layout,
        #observe_yaws=True,
        #lidar_observations=None,
        mast_layout=mast_layout,
        floris="myfloris.json",
    )

    return env

def get_4wt_symmetric_env(
        episode_length=10, 
        privileged=True, 
        mast_distancing=50, 
        sorted_wind=False, 
        wind_speed=None,
        changing_wind=False,
        action_representation='wind',
        noise=0.0,
        verbose=False,
    ):
    turbine_layout = ([0, 250, 0, 250], [0, 0, 250, 250])
    w, h = np.max(turbine_layout, axis=1)

    if privileged: 
        print(f"Making env with n masts: {len(mast_layout[0])}")
        mast_layout = get_grid_points(w, h, mast_distancing) 
    else: 
        mast_layout = None

    env = modified_env(
        turbine_layout=turbine_layout,
        #lidar_observations=('wind_speed', 'wind_direction'),
        mast_layout=mast_layout,
        floris="myfloris.json",
        episode_length=episode_length,
        sorted_wind=sorted_wind,
        wind_speed=wind_speed,
        changing_wind=changing_wind,
        action_representation=action_representation,
        observation_noise=noise,
        verbose=verbose,
    )

    return env

# Define Lating hypercube sampling function
def get_lhs_env(
        layout_name,
        episode_length=100, 
        privileged=True, 
        sorted_wind=False, 
        wind_speed=None,
        changing_wind=False,
        action_representation='wind',
    ):
    
    turbine_layout = np.load(f"data/layouts/{layout_name}_turbine_layout.npy")
    mast_layout = np.load(f"data/layouts/{layout_name}_mast_layout.npy") if privileged else None
    
    env = modified_env(
        turbine_layout=turbine_layout,
        mast_layout=mast_layout,
        floris="myfloris.json",
        episode_length=episode_length,
        sorted_wind=sorted_wind,
        wind_speed=wind_speed,
        changing_wind=changing_wind,
        action_representation=action_representation,
    )

    # plot the turbine layout, and masts with and x
    plt.scatter(turbine_layout[0], turbine_layout[1])
    plt.scatter(mast_layout[0], mast_layout[1], marker='x')
    plt.axis('scaled')
    plt.show()
    return env


if __name__ == "__main__":

    env = get_4wt_symmetric_env(
        episode_length=100, 
        privileged=True, 
        mast_distancing=50, 
        sorted_wind=False, 
        wind_speed=8,
        changing_wind=True,
        verbose=True,
    )
    
    obs = env.reset()
    #print(obs)
    done = False
    ok = []
    non_ok = []
    is_ok = True
    nan_power_angles = []
    for j in range(50):
        for i in range(50):
            obs, reward, done, _, info = env.step([0, 0, 0, 0])
            # Print yaws and wind angle
            turbine_yaws = [t.yaw_angle for t in env.turbines]
            print(turbine_yaws, env.wind_process.wind_direction)
            # Print max and min difference to wind direction
            wind = env.wind_process.wind_direction
            print(np.max(np.abs(np.array(turbine_yaws) - wind)), np.min(np.abs(np.array(turbine_yaws) - wind)))
            try:
                img = env.render(mode='rgb_array')
                # save img
                plt.imsave(f"figures/renders/{j}.png", img)
            except:
                pass
            break
            #print(i, obs[:8], reward, done, info)
            if np.isnan(info['power_output']):
                nan_power_angles.append(env.wind_process.wind_direction)
                1/0
            try:
                env.render()
            except:
                #print(" > WIND DIRECTION: ", env.wind_process.wind_direction)
                is_ok = False
                break
        if is_ok:
            ok.append(env.wind_process.wind_direction)
        else:
            non_ok.append(env.wind_process.wind_direction)
        is_ok = True
        obs = env.reset()
        #print(obs)

    if False:
        print("OK:", ok)
        if len(ok) > 0: print(np.min(ok), np.max(ok))
        print("NON OK:", non_ok)
        if len(non_ok) > 0: print(np.min(non_ok), np.max(non_ok))
        print("NAN POWER ANGLES:", nan_power_angles)
        if len(nan_power_angles) > 0: print(np.min(nan_power_angles), np.max(nan_power_angles))
        
        # print unique sorted values of power angles
        print("UNIQUE POWER ANGLES:", np.unique(nan_power_angles))

        # clear plot
        plt.clf()
        # Scatterplot of ok and not ok
        # ok points are a dot, nan power points are a cross
        # has labels and title
        plt.scatter(ok, [0]*len(ok), label='OK', marker='.')
        plt.scatter(non_ok, [0]*len(non_ok), label='NOT OK', marker='x')
        plt.scatter(nan_power_angles, [5]*len(nan_power_angles), label='NAN POWER', marker='o')
        plt.legend()
        plt.title("Wind direction angles that break floris")
        # x axis label is angles
        plt.xlabel("Wind direction angles")
        plt.savefig("wind_direction_angles.png")
        plt.show()
    