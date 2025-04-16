
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from dynamic_windfarm_env import DynamicPriviegedWindFarmEnv

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
        load_pyglet_visualization=False,
        floris_path="myfloris.json",
        dynamic_mode=None,
        parallel_dynamic_computations=False,
    ):
    turbine_layout = ([0, 250, 0, 250], [0, 0, 250, 250])
    w, h = np.max(turbine_layout, axis=1)

    if privileged: 
        mast_layout = get_grid_points(w, h, mast_distancing) 
        print(f"Making env with n masts: {len(mast_layout[0])}")
    else: 
        mast_layout = None

    env = DynamicPriviegedWindFarmEnv(
        turbine_layout=turbine_layout,
        #lidar_observations=('wind_speed', 'wind_direction'),
        mast_layout=mast_layout,
        floris=floris_path,
        episode_length=episode_length,
        sorted_wind=sorted_wind,
        wind_speed=wind_speed,
        changing_wind=changing_wind,
        action_representation=action_representation,
        observation_noise=noise,
        verbose=verbose,
        update_rule=dynamic_mode,
        load_pyglet_visualization=load_pyglet_visualization,
        parallel_dynamic_computations=parallel_dynamic_computations,
        time_delta=5,
        op_per_turbine=5,
        op_wake_matrix_horizon=20,
    )

    return env

# Define Lating hypercube sampling function
def get_lhs_env(
        layout_name,
        episode_length=100, 
        privileged=True, 
        mast_distancing=None,
        sorted_wind=False, 
        wind_speed=None,
        changing_wind=False,
        action_representation='wind',
        load_pyglet_visualization=False,
        noise=0.0,
        verbose=False,
        dynamic_mode=None,
    ):

    if noise > 0:
        raise NotImplementedError("Noise is not implemented for LHS environments")
    
    turbine_layout = np.load(f"data/layouts/{layout_name}_turbine_layout.npy")
    if privileged:
        if mast_distancing is None:
            print("Loading default mast layout")
            mast_layout = np.load(f"data/layouts/{layout_name}_mast_layout.npy") if privileged else None
        else:
            bounds = layout_name.split("_wb")[1].split("_")[0].split("x")
            w, h = [int(b) for b in bounds]
            mast_layout = get_grid_points(w, h, mast_distancing) 
            print(f"Making env with n masts: {len(mast_layout[0])}")
    else:
        mast_layout = None

    if len(turbine_layout) == 8:
        time_delta = 10
        op_per_turbine = 3
        op_wake_matrix_horizon = 20
    elif len(turbine_layout) == 16:
        time_delta = 10
        op_per_turbine = 3
        op_wake_matrix_horizon = 20
    else:
        raise NotImplementedError("Only 8 and 16 turbine layouts are implemented")
    
    env = DynamicPriviegedWindFarmEnv(
        turbine_layout=turbine_layout,
        mast_layout=mast_layout,
        floris="myfloris.json",
        episode_length=episode_length,
        sorted_wind=sorted_wind,
        wind_speed=wind_speed,
        changing_wind=changing_wind,
        action_representation=action_representation,
        load_pyglet_visualization=load_pyglet_visualization,
        verbose=verbose,
        update_rule=dynamic_mode,
        time_delta=time_delta,
        op_per_turbine=op_per_turbine,
        op_wake_matrix_horizon=op_wake_matrix_horizon,
    )

    # Plot turbine layout
    plt.figure()
    plt.scatter(turbine_layout[0], turbine_layout[1])
    if privileged:
        plt.scatter(mast_layout[0], mast_layout[1], marker='x')
    plt.axis('scaled')
    # Add legend
    plt.legend(['Turbines', 'Masts'])
    # Remove axis ticks and numbers
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'data/layouts/{layout_name}_md{mast_distancing}.png')
    # plt.show()
    return env


if __name__ == "__main__":
    for md in [150, 200, 250, 300, 375, 500]:
        env = get_lhs_env(
            "lhs_env_nt16_md75_wb1500x1500",
            episode_length=100, 
            privileged=True, 
            mast_distancing=md, 
            changing_wind=True,
            load_pyglet_visualization=True,
        )
    
    for md in [75, 100, 150, 200]:
        env = get_lhs_env(
            "lhs_env_nt8_md150_wb750x750",
            episode_length=100, 
            privileged=True, 
            mast_distancing=md, 
            changing_wind=True,
            load_pyglet_visualization=True,
        )
        
    obs = env.reset()
    #print(obs)
    done = False
    ok = []
    non_ok = []
    is_ok = True
    nan_power_angles = []
    for i in range(5):
        for j in range(50):
            action = np.zeros(env.action_space.shape)
            obs, reward, done, _, info = env.step(action)
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
    