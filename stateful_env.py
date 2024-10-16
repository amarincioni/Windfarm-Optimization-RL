import numpy as np
import matplotlib.pyplot as plt

from env_utils import modified_env, get_grid_points

# Wrapper for modified_env
class StatefulEnv(modified_env):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.figure = None
        
    def render(self, obs):
        if self.figure is None:
            self.figure = plt.figure()

        # check if any in obs is nan
        if np.isnan(obs).any():
            obs = np.nan_to_num(obs, nan=0)

        # plot obs (matrix of wind speeds)
        plt.imshow(obs)
        plt.show()
        
    

def get_4wt_symmetric_env(
        episode_length=10, 
        privileged=True, 
        mast_distancing=50, 
        sorted_wind=False, 
        wind_speed=None,
        changing_wind=False,
        action_representation='wind',
    ):
    turbine_layout = ([0, 250, 0, 250], [0, 0, 250, 250])
    w, h = np.max(turbine_layout, axis=1)

    mast_layout = get_grid_points(w, h, mast_distancing) if privileged else None
    if mast_layout is not None:
        print(f"Making env with n masts: {len(mast_layout[0])}")

    env = StatefulEnv(
        turbine_layout=turbine_layout,
        mast_layout=mast_layout,
        floris="myfloris.json",
        episode_length=episode_length,
        sorted_wind=sorted_wind,
        wind_speed=wind_speed,
        changing_wind=changing_wind,
        action_representation=action_representation,
    )

    return env

if __name__ == "__main__":

    env = get_4wt_symmetric_env(
        episode_length=100, 
        privileged=True, 
        mast_distancing=50, 
        sorted_wind=False, 
        wind_speed=8,
    )
    
    obs = env.reset()
    print(obs)
    done = False
    ok = []
    non_ok = []
    is_ok = True
    nan_power_angles = []
    for j in range(5):
        for i in range(50):
            obs, reward, done, _, info = env.step([0, 0, 0, 0])
            print(i, obs[:8], reward, done, info)
            if np.isnan(info['power_output']):
                nan_power_angles.append(env.wind_process.wind_direction)
            #try:
            print(f"Len obs {len(obs)}")
            env.render(obs[4:])
            #except:
            #    print(" > WIND DIRECTION: ", env.wind_process.wind_direction)
            #    is_ok = False
            #    break
        if is_ok:
            ok.append(env.wind_process.wind_direction)
        else:
            non_ok.append(env.wind_process.wind_direction)
        is_ok = True
        obs = env.reset()
        print(obs)

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
    