from wind_farm_env.wind_farm_gym import WindFarmEnv
from wind_processes import RandomResetWindProcess, SetSequenceWindProcess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
from PIL import Image
import pandas as pd
from scipy.ndimage import gaussian_filter
from utils import get_circle_idxs
import time
import jax
import jax.numpy as jnp
import functools
from multiprocessing import Pool

class DynamicPriviegedWindFarmEnv(WindFarmEnv):
    def __init__(self, 
        turbine_layout, 
        mast_layout, 
        floris, 
        episode_length=10, 
        sorted_wind=False,      # To run evaluation
        wind_speed=None,        # To run evaluation
        lidar_observations=None,
        mast_observations=('wind_speed', 'wind_direction'),
        changing_wind=False,    # More difficult environment that requries actual control
        action_representation='wind',
        observation_noise=0.0,
        verbose=False,
        update_rule=None,
        momentum_alpha=0.95,
        momentum_beta=1,
        load_pyglet_visualization=False,
        parallel_dynamic_computations=False,
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
            farm_observations=['wind_speed', 'wind_direction'],
            time_delta=5
        )

        self.episode_length = episode_length
        self.step_count = 0
        self.runs = 0
        self.observation_noise = observation_noise
        self.lidar_observations = lidar_observations
        self.mast_observations = mast_observations
        self.verbose = verbose
        self.update_rule = update_rule
        self.momentum_alpha = momentum_alpha
        self.momentum_beta = momentum_beta
        self.parallel_dynamic_computations = parallel_dynamic_computations

        # Defines the points to sample the wind from 
        # This is used to get the wind state from the steady state flow field of floris
        self.recorded_points = self.initialize_recorded_points()
        if self.verbose:
            for k in self.recorded_points.keys():
                print(f"Points for {k}: {self.recorded_points[k].shape}")

        # These variables are used to store the wind state and plot state
        self.steady_state = None
        self.steady_state_plot = None
        self.dynamic_state = None
        self.dynamic_state_plot = None
        assert self.update_rule in [None, 'momentum', 'observation_points'], "Update rule not recognized"
        # Should we provide a wind history to the observation for the 
        # dynamic state?
        # self.wind_direction_history = []
        # self.wind_speed_history = []
        # self.wind_history_length = 10

        # Observation points update
        self.op_per_turbine = 5
        self.op_wake_matrix_horizon = 20
        self.op_dynamic_state_margin = 100
        self.observation_points = pd.DataFrame(columns=["x", "y", "z", "yaw", "represented_speed_u", "represented_speed_v", "represented_speed_w", "source", "t", "age"])
        self.wake_matrices = {}
        # mast and turbine take into account
        if mast_layout is not None:
            turbine_and_masts = np.concatenate((self.turbine_layout, self.mast_layout), axis=1)
        else:
            turbine_and_masts = self.turbine_layout
        self.wf_bounds = ((0, 0), np.max(turbine_and_masts, axis=1))
        print("Wind farm bounds", self.wf_bounds)
        self.t_power_log = [] # To correctly plot the power output
        if self.parallel_dynamic_computations:
            self.pool = Pool(8)
        if self.verbose: print("Wind farm bounds", self.wf_bounds)

        # Size of the represented state
        # Max side is 50
        if np.max(turbine_and_masts) < 500:
            self.dynamic_state_shape = (50, 50)
        else:
            self.dynamic_state_shape = (100,100)
        x_len = self.wf_bounds[1][0] - self.wf_bounds[0][0] + 2*self.op_dynamic_state_margin
        y_len = self.wf_bounds[1][1] - self.wf_bounds[0][1] + 2*self.op_dynamic_state_margin
        self.len_ratio = min(self.dynamic_state_shape[0]/x_len, self.dynamic_state_shape[1]/y_len)
        self.dynamic_state_shape = (int(x_len*self.len_ratio), int(y_len*self.len_ratio))
        #self.dynamic_state_offset = (self.wf_bounds[0][0]*len_ratio, self.wf_bounds[0][1]*len_ratio)
        # assert all([t.rotor_radius == self.turbines[0].rotor_radius for t in self.turbines]), "Turbines have different rotor radius"
        # self.dynamic_state_r = self.turbines[0].rotor_radius * self.len_ratio

        # Initialize visualization if enabled
        if self.load_pyglet_visualization:
            self.visualization = self.FarmVisualization(
                self.floris_interface, 
                flow_points=self._flow_points(),
                windfarm_info={
                    "bounds": self.wf_bounds,
                    "margin": self.op_dynamic_state_margin,
                },
                )

        # Double plot functionalities for debugging
        self.save_double_plot = False
        self.double_plot_images = []
        self.op_plts = []


    # Initialization
    def initialize_recorded_points(self):
        # Record points for all turbines and masts
        # These are the points where the wind is sampled from Floris

        recorded_points = {}
        for i, (coord, turbine) in enumerate(self.floris_interface.floris.farm.flow_field.turbine_map.items):
            if self.verbose: print("Turbine", i)
            # record points centered in the turbine, for n angles (in degrees)
            recorded_points[f"turbine_{i}"] = []

            # Record as many points as make sense according to the turbine's grid and rotor radius
            num_points = int(np.round(np.sqrt(turbine.grid_point_count)))
            pt = turbine.rloc * turbine.rotor_radius

            # Record points either in a square grid or in a cylindrical grid
            # The square grid evenly represents the space, the cylindrical grid 
            # samples too many points close to the turbine and not enough far away,
            # but it is closer to the ones used in floris.
            for dx in np.linspace(-pt, pt, num_points):
                for dy in np.linspace(-pt, pt, num_points):
                    for dz in np.linspace(-pt, pt, num_points):
                        x = coord.x1 + dx
                        y = coord.x2 + dy
                        z = coord.x3 + dz
                        recorded_points[f"turbine_{i}"].append((x, y, z))

            if self.verbose:
                # Plot and save points in 3d scatter plot
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                for (x, y, z) in recorded_points[f"turbine_{i}"]:
                    ax.scatter(x, y, z, c='r')
                plt.title(f"Flow points for turbine {i}")
                plt.savefig(f"figures/recorded_points_turbine_{i}.png")

        assert all([t.hub_height == self.turbines[0].hub_height for t in self.turbines]), "Turbines have different hub heights"
        self.turbine_hub_height = self.turbines[0].hub_height

        for i, (x, y) in enumerate(np.array(self.mast_layout).T):
            # Record one single point at the mast matching hub height
            recorded_points[f"mast_{i}"] = [(x, y, self.turbine_hub_height)]

        if self.verbose:
            # Plot and save points in 3d scatter plot for all turbines and masts
            # turbines are red, masts are blue
            # keep the axis range in scale for better visualization
            # add legend for colors
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for k, points in recorded_points.items():
                for x, y, z in points:
                    c, label = ('r', 'Turbines') if "turbine" in k else ('b', 'Masts')
                    ax.scatter(x, y, z, c=c, s=1, label=label)
            plt.title(f"Flow points for all turbines and masts")
            plt.legend(handles=[
                matplotlib.patches.Patch(color='red', label='Turbines'),
                matplotlib.patches.Patch(color='blue', label='Masts')
            ])
            plt.axis('equal')
            plt.savefig(f"figures/recorded_points_all.png")

        # Save as array
        for k, points in recorded_points.items():
            recorded_points[k] = np.array(points)

        return recorded_points

    # Reset, step and render functions
    def reset(self, wind_direction=None, **kwargs):
        obs = super().reset()

        # Set wind direction if provided
        if wind_direction is not None:
            assert isinstance(self.wind_process, RandomResetWindProcess), \
                "Wind process is not RandomResetWindProcess, cannot initialize it with preset wind direction"
            self.wind_process.wind_direction = wind_direction
        
        # Update environment recompute flow field with new wind parameters
        self.floris_interface.reinitialize_flow_field(wind_speed=self.wind_process.wind_speed, wind_direction=self.wind_process.wind_direction)

        # Initialize turbine directions randomly, but towards the wind
        for turbine in self.turbines:
            turbine.yaw_angle = self._np_random.uniform(self.desired_min_yaw, self.desired_max_yaw)

        # if self.update_rule is not None:
        #     for i in range(self.wind_history_length):
        #         self.wind_direction_history.append(self.wind_process.wind_direction)
        #         self.wind_speed_history.append(self.wind_process.wind_speed)
        #         self.step([0]*len(self.turbines))

        # Get info dictionary with power output
        power_output = np.sum(self.floris_interface.get_turbine_power())
        info = {'power_output': np.nan_to_num(power_output, nan=0)}

        # Update debugging metrics
        self.step_count = 0
        self.runs += 1
        if self.verbose:
            print("Run count:", self.runs)
            print("Wind speed:", self.wind_process.wind_speed)
            print("Wind direction:", self.wind_process.wind_direction)

        # Update state according to the update rule if dynamic environment
        if self.update_rule is not None:
            self.steady_state = self.get_steady_state()
            if self.load_pyglet_visualization:
                self.steady_state_plot = self.visualization.get_cut_plane().df

        # Double plot functionalities for debugging
        self.save_double_plot_images()
        self.double_plot_images = []
        self.op_plts = []

        return obs, info

    def save_double_plot_images(self, path="figures/renders/last_double_plot.gif"):
        if self.save_double_plot and len(self.double_plot_images) > 0:
            # Save the double plot images as a gif
            imgs_pil = [Image.fromarray(img) for img in self.double_plot_images]
            imgs_pil[0].save(path, save_all=True, append_images=imgs_pil[1:], loop=0, duration=100)
        if len(self.op_plts) > 0:
            imgs_pil = [Image.fromarray(img) for img in self.op_plts]
            imgs_pil[0].save("figures/renders/last_op_plot.gif", save_all=True, append_images=imgs_pil[1:], loop=0, duration=100)

    def close(self):
        self.save_double_plot_images()
        return super().close()

    def step(self, action):

        if self.update_rule is None:
            obs, reward, done, info = super().step(action)

            # Add power output to info
            power_output = np.sum(self.floris_interface.get_turbine_power())
            info['power_output'] = np.nan_to_num(power_output, nan=0)
        else:
            obs, reward, done, info = self.dynamic_step(action)
        

        # Sanitize observation
        if any([np.isnan(x) for x in obs]):
            print("NAN OBSERVATION")
            obs = np.nan_to_num(obs, nan=0)
            
        # Compute if the episode is terminated or truncated
        terminated, truncated = False, False
        if self.step_count >= self.episode_length: 
            terminated, truncated = True, False

        # Add observation noise
        if self.observation_noise > 0:
            obs = obs + np.random.normal(0, self.observation_noise, len(obs))

        # Update step count
        self.step_count += 1
        return obs, reward, terminated, truncated, info

    def dynamic_step(self, action):
        # Adjust yaws of the environment
        obs, reward, done, info = super().step(action)

        ##### Modified dynamics ########
        # Get the steady state flow field
        self.steady_state = self.get_steady_state()
        if self.load_pyglet_visualization:
            self.steady_state_plot = self.visualization.get_cut_plane().df

        # Update state according to the update rule
        self.dynamic_state, self.dynamic_state_plot = self.update_dynamic_state(self.steady_state, self.steady_state_plot)
        ##### End modified dynamics #####

        # Compute rewards, info, observations given the new state
        reward, info = self.get_dynamic_reward()
        obs = self.get_dynamic_state()
        done = False

        return obs, reward, done, info

    def render(self, mode='human'):
        
        assert self.load_pyglet_visualization, "Visualization is disabled, pyglet is not loaded"

        if self.state is None:
            return None
        
        if self.save_double_plot:
            ss_plt = self.visualization.render(return_rgb_array=True, 
                wind_state=self.steady_state_plot, observation_points=None,
                turbine_power=self.t_power_log)
            dyn_plt = self.visualization.render(return_rgb_array=True, 
                wind_state=self.dynamic_state_plot, observation_points=self.observation_points,
                turbine_power=self.t_power_log)
                
            # Merge dynamic and steady state plot
            plots = (dyn_plt, ss_plt)
            vis = np.concatenate(plots, axis=0)
            # Overlay observation points dynamic plot 
            if self.update_rule == 'observation_points':
                op_plt = self._op_render()
                self.op_plts.append(op_plt)
                side_pad = int((op_plt.shape[1] - op_plt.shape[0]) / 2)
                op_plt = op_plt[:, side_pad:-side_pad, :]

                ratio = vis.shape[0] / 2 / op_plt.shape[0]
                op_plt = cv2.resize(op_plt, dsize=None, fx=ratio, fy=ratio)
                
                vis[:op_plt.shape[0], -op_plt.shape[1]:, :] = op_plt
                cv2.putText(vis, 'Gaussian filter on observation points', (5 + vis.shape[1]-op_plt.shape[1], 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)


            # Add text to both images in black, small size
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(vis, 'Dynamic state', (10, 20), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(vis, 'Steady state', (10, 20 + dyn_plt.shape[0]), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            self.double_plot_images.append(vis)

            plt.imshow(vis)
            plt.show()
            
        return self.visualization.render(return_rgb_array=mode == 'rgb_array', 
            wind_state=self.dynamic_state_plot, observation_points=self.observation_points,
            turbine_power=self.t_power_log)

    # Functions to manage the dynamic state
    def get_steady_state(self):
        # Returns the steady state flow field for the current wind state
        # Output is a dictionary of dataframes of keys turbine_i and mast_i
        all_points = []
        for k, points in self.recorded_points.items():
            all_points.extend(points)
        all_points = np.array(all_points)

        steady_state_flow = self.floris_interface.get_set_of_points(all_points[:,0], all_points[:,1], all_points[:,2])
        if self.verbose: print(f"Got {len(steady_state_flow)} flow points")
        steady_state_flow = steady_state_flow.groupby(["x", "y", "z"]).mean().reset_index()
        
        steady_state = {}
        for k, points in self.recorded_points.items():
            # For each key k get the rows where at least one of the points matches the recorded points
            # TODO make sure this is correct, although assert is not catching mistakes..
            steady_state[k] = steady_state_flow[(steady_state_flow["x"].isin(points[:,0])) & (steady_state_flow["y"].isin(points[:,1])) & (steady_state_flow["z"].isin(points[:,2]))]        
            assert len(steady_state[k]) == len(points), f"Missing points for {k}"
        return steady_state

    def update_dynamic_state(self, steady_state, steady_state_plot=None):
        # Use the correct update rule
        if self.update_rule == 'momentum':
            return self._momentum_update(steady_state, steady_state_plot)
        elif self.update_rule == 'observation_points':
            return self._op_update(steady_state, steady_state_plot)
        else:
            raise NotImplementedError(f"Update rule {self.update_rule} not implemented")

    def get_dynamic_state(self):
        # Returns the observation of the dynamic state environment
        #   for the RL agent

        # Code taken from original environment _get_state()
        self.current_flow_points = self._flow_points()
        if len(self.current_flow_points[0]) > 0:
            self._current_flow = self.floris_interface.get_set_of_points(*self.current_flow_points)

        # Now only used for non mast observations
        state = [self._get_measurement_point_data(d) for d in self.observed_variables if "mast_" not in d["name"]]

        # Add the mast observations
        if self.update_rule == 'momentum':
            mast_state = self._momentum_get_mast_state()
        elif self.update_rule == 'observation_points':
            mast_state = self._op_get_mast_state()
        else:
            raise NotImplementedError(f"Update rule {self.update_rule} not implemented")
        state.extend(mast_state)

        state = np.array([float(s) for s in state])
        # rescale and clip off
        if self._normalize_observations:
            state = (np.array(state) - self.low) / self.state_delta
            state = np.clip(state, np.zeros_like(self.low), np.ones_like(self.high))
        else:
            state = np.clip(state, self.low, self.high)
        
        return state

    def get_dynamic_reward(self):
        # Compute the reward for the dynamic environment
        # This updates the turbine velocities and computes the power output

        # For each turbine, update velocities using stored dynamic values
        if self.update_rule == 'momentum':
            self._momentum_turbine_update()
            total_power = np.sum(self.floris_interface.get_turbine_power())
        elif self.update_rule == 'observation_points':
            powers = self._op_turbine_update()
            total_power = np.sum(powers)
        else:
            raise NotImplementedError(f"Update rule {self.update_rule} not implemented")

        # Get power, sum it up and return rewards and info
        if np.isnan(total_power): total_power = 0
        reward = total_power * self._reward_scaling_factor
        info = {'power_output': np.sum(total_power)}
        return reward, info

    # Dynamic environment update rules
    def _momentum_update(self, steady_state, steady_state_plot=None):
        # Update the dynamic state using a momentum update rule

        # The first state is just the steady state
        if self.dynamic_state is None:
            next_dynamic_state = steady_state
            next_dynamic_state_plot = steady_state_plot
        else:
            # Update the dynamic state
            next_dynamic_state = {}
            for k, steady_state_df in steady_state.items():
                next_dynamic_state[k] = self.dynamic_state[k]*self.momentum_alpha + steady_state_df*self.momentum_beta
            if steady_state_plot is not None:
                next_dynamic_state_plot = self.dynamic_state_plot*self.momentum_alpha + steady_state_plot*self.momentum_beta

        return next_dynamic_state, next_dynamic_state_plot
    
    def _momentum_turbine_update(self):
        for i, (coord, turbine) in enumerate(self.floris_interface.floris.farm.flow_field.turbine_map.items):
            # Force recomputing the closest points given the flow points
            turbine.flow_field_point_indices = None
            turbine_wind = self.dynamic_state[f"turbine_{i}"]["u"].values
            wind_coordinates = self.recorded_points[f"turbine_{i}"]
            turbine.velocities = turbine.calculate_swept_area_velocities(turbine_wind, coord, wind_coordinates[:,0], wind_coordinates[:,1], wind_coordinates[:,2])

    def _momentum_get_mast_state(self):
        # Get dynamic mast observations from the stored values
        assert self.mast_observations == ('wind_speed', 'wind_direction'), "Only wind speed and direction are supported as mast observations"
        # For each mast, get the wind speed and direction
        mast_state = []
        for i in range(len(self.mast_layout[0])):
            #x, y = self.mast_layout[0][i], self.mast_layout[1][i]
            mast_df = self.dynamic_state[f"mast_{i}"]
            #print(f"Printing mast_{i}", mast_df, self.step_count)
            assert len(mast_df) == 1, "Mast should have only one point"
            u, v = mast_df["u"].values, mast_df["v"].values
            wind_speed = (u ** 2 + v ** 2) ** 0.5
            wind_direction = (np.degrees(np.arctan2(v, u)) + self._farm.wind_map.input_direction[0]) % 360
            mast_state.extend([wind_speed, wind_direction])
        return mast_state

    def _op_update(self, steady_state, steady_state_plot=None):
        # Update the dynamic state using the observation points update rule
        # pseudocode for each step

        # For each turbine, generate a set of new observation points
        #   - These points inherit the turbine yaw and info about the wake
        
        # Compute the change in point positions due to the wind 
        # This is the same for all points, so we precompute it
        wind_direction, wind_speed = self.wind_process.wind_direction, self.wind_process.wind_speed
        wind_direction_rad = np.deg2rad(-wind_direction - 90)
        d_pos = np.array([wind_speed*np.cos(wind_direction_rad), wind_speed*np.sin(wind_direction_rad), 0])
        d_pos = d_pos * self.time_delta

        sampling_points = []
        new_ops = []
        for i, (coord, turbine) in enumerate(self.floris_interface.floris.farm.flow_field.turbine_map.items):
            rotor_radius = turbine.rotor_radius
            yaw = self.yaws_from_north[i]
            yaw = np.deg2rad(-yaw)

            # Make a line of observation points at the turbine
            for j, dr in enumerate(np.linspace(-rotor_radius, rotor_radius, self.op_per_turbine)):
                c = np.array([coord.x1, coord.x2, coord.x3]) #xyz point
                # Get the point moved from the center c by dr in the direction of the yaw
                point = c + dr*np.array([np.cos(yaw), np.sin(yaw), 0])
                
                new_ops.append({
                    "x": point[0],
                    "y": point[1],
                    "z": point[2],
                    "yaw": yaw,
                    # "represented_speed_u": 0,
                    # "represented_speed_v": 0,
                    # "represented_speed_w": 0,
                    "wind_direction": wind_direction,
                    "wind_speed": wind_speed,
                    "source": f"turbine_{i}",
                    "t": self.step_count,
                    "age": 0,
                })
                # For each observation point, save the line of points to sample
                # to build the wake matrix
                sampling_points.append(np.vstack((np.arange(self.op_wake_matrix_horizon), ) * 3).T*d_pos + point)
        
        self.observation_points = pd.concat((self.observation_points, pd.DataFrame(new_ops)), ignore_index=True)
        
        sampling_points = np.array(sampling_points)
        # (N_TURBINES x OP_PER_TURBINE, OP_WAKE_HORIZON, 3) (5,20,3)
        wake_matrix = self.floris_interface.get_set_of_points(sampling_points[:,:,0].flatten(), sampling_points[:,:,1].flatten(), sampling_points[:,:,2].flatten())
        wake_matrix = wake_matrix.groupby(["x", "y", "z"]).mean().reset_index()
        self.wake_matrices[self.step_count] = {
            "wake_matrix": wake_matrix,	
            "angle": wind_direction, # UNUSED
        }

        # Move the point according to the freestream wind (FloriDyn 2.3.3)
        self.observation_points["x"] += d_pos[0]
        self.observation_points["y"] += d_pos[1]
        self.observation_points["z"] += d_pos[2]
        self.observation_points["age"] += 1
        # Filter old points and points out of bounds
        self.observation_points = self.observation_points[self.observation_points["age"] < self.op_wake_matrix_horizon]
        self.observation_points = self.observation_points[self.observation_points["x"] > self.wf_bounds[0][0] - self.op_dynamic_state_margin]
        self.observation_points = self.observation_points[self.observation_points["x"] < self.wf_bounds[1][0] + self.op_dynamic_state_margin]
        self.observation_points = self.observation_points[self.observation_points["y"] > self.wf_bounds[0][1] - self.op_dynamic_state_margin]
        self.observation_points = self.observation_points[self.observation_points["y"] < self.wf_bounds[1][1] + self.op_dynamic_state_margin]
        # Update the speed that each point represents
        self.observation_points = self.observation_points.apply(self._get_closest_represented_speed, axis=1)

        # Compute the dynamic state
        t0 = time.time()
        new_dynamic_state = self._op_to_dynamic_state()
        if self.verbose: print("Dynamic state computation time", time.time()-t0)

        return new_dynamic_state, self.steady_state_plot

    def _op_to_dynamic_state(self):
        x = np.copy(self.observation_points["x"])
        y = np.copy(self.observation_points["y"])
        u = np.copy(self.observation_points["u"])
        age = np.copy(self.observation_points["age"])
        v = np.copy(self.observation_points["v"])
        w = np.copy(self.observation_points["w"])
        wind_direction = np.copy(self.observation_points["wind_direction"])
        mag = np.sqrt(u**2 + v**2 + w**2)

        # Computing it transposed, and then transposing it back
        # , to make sure the indexing is correct like the plotting
        wind_direction = -(wind_direction + 90) % 360
        uv_angle = np.degrees(np.arctan2(v, u)) % 360
        # uv_angle[uv_angle > 90] = uv_angle[uv_angle > 90] - 180
        # uv_angle[uv_angle < -90] = uv_angle[uv_angle < -90] + 180
        uv_angle = (wind_direction + uv_angle) % 360
        wind_direction = (wind_direction + uv_angle) % 360
        # Detransposing
        wind_direction = -wind_direction - 90
        # Angle wrt the old wind direction
        # local_angle = np.arctan2(v, u)
        # Absolute angle represented by the op
        # wind_direction_rad = wind_direction_rad + local_angle

        # Size of the wind farm
        (x_min, y_min), (x_max, y_max) = self.wf_bounds
        r = int(self.turbines[0].rotor_radius * self.len_ratio / self.op_per_turbine)
        assert r > 0, "Radius is 0, probably caused by a too small dynamic state shape"    
        # For each observation point, apply a gaussian filter
        # Average of the gaussian filters is the dynamic state
        
        # How does the wind speed change wrt freestream?
        #mag = mag + self.wind_process.wind_speed
        wind_speed_change = mag - self.wind_process.wind_speed
        # Note that this ignores the directionality of the wind....
        # The correct implementation is (u, v) -> (h, v)
        # Then from the absolute wind directions you can actually do sums
        #   (subtractions) of the vectors, like up here

        # For each point, apply a gaussian filter and sum them up
        # This allows for multiple points to influence the same location
        # , but may cause negative values in wind...
        # Something like a p-mean with high p could be better, to weigh the
        #   bigger reductions more, but this is a good start
        t0 = time.time()
        # Jax is used here to speed up the computation   
        smooth_wake_shape = self.dynamic_state_shape
        xyval = np.array([(x+self.op_dynamic_state_margin)*self.len_ratio, (y+self.op_dynamic_state_margin)*self.len_ratio, wind_speed_change]).T
        if self.parallel_dynamic_computations:
            # Parallelized filter computation
            smooth_wind_speed = self.pool.map(functools.partial(_op_get_gaussian_filter, r=r, smooth_wake_shape=smooth_wake_shape), xyval)
            smooth_wind_speed = np.sum(smooth_wind_speed, axis=0)
            dynamic_state_speed = smooth_wind_speed + np.ones_like(smooth_wind_speed)*self.wind_process.wind_speed
            # Computation not parallelized if it cant be used
            # but only parallelize the gaussian filter computation
            # and vectorize the rest with numpy
            smooth_wake_shape = self.dynamic_state_shape
            xyval = np.array([(x+self.op_dynamic_state_margin)*self.len_ratio, (y+self.op_dynamic_state_margin)*self.len_ratio, wind_speed_change]).T
            # Make len(x) cirlces of radius r, centered in (x, y)
            Ax = np.arange(-r, int(smooth_wake_shape[0] + r))
            Ay = np.arange(-r, int(smooth_wake_shape[1] + r))
            Ax = np.tile(Ax, (len(x), 1))
            Ay = np.tile(Ay, (len(x), 1))
            Ax = Ax.T - xyval[:,0]
            Ay = Ay.T - xyval[:,1]
            A = Ax**2 + Ay[:,None]**2
            circles = ((A - r**2) <= 0).astype(int)
            circles = circles * wind_speed_change

            circles = np.transpose(circles, (2, 1, 0))
            circles = circles[:, r:-r, r:-r]

            # smooth_wake = self.pool.map(functools.partial(gaussian_filter, sigma=r), circles)
            smooth_wake = np.array([gaussian_filter(c, sigma=r) for c in circles])
            smooth_wake = np.sum(smooth_wake, axis=0)
            dynamic_state_speed = smooth_wake + np.ones_like(smooth_wake)*self.wind_process.wind_speed  
        else:
            wake_gaussians = _op_centered_gaussians(smooth_wake_shape, xyval[:,0], xyval[:,1], r, wind_speed_change)
            smooth_wake = np.sum(wake_gaussians, axis=0)
            dynamic_state_speed = smooth_wake + np.ones_like(smooth_wake)*self.wind_process.wind_speed
        if self.verbose: print("Speed computation time", time.time()-t0)
        # This doesnt allow for points to overlap, overlappign points are overwritten,
        #   , here by the more recent points, but this wouldnt allow for wakes to cross 
        # wake = np.zeros((int(x_max-x_min) + 2*self.op_dynamic_state_margin + 2*r, int(y_max-y_min) + 2*self.op_dynamic_state_margin + 2*r))
        # # Overwrite the older points with the newer ones
        # for i in np.argsort(-age):
        #     # set the value to u
        #     _x = int(x[i] + self.op_dynamic_state_margin)
        #     _y = int(y[i] + self.op_dynamic_state_margin)
        #     circle = get_circle_idxs(r) * wind_speed_change[i]
        #     wake[_x:_x+2*r+1, _y:_y+2*r+1] = circle
        # wake = wake[r:-r, r:-r]
        # dynamic_state_speed = gaussian_filter(wake, sigma=r) + np.ones_like(wake)*self.wind_process.wind_speed


        # How does the wind direction change wrt freestream?
        t0 = time.time()
        wind_direction_change = (wind_direction - self.wind_process.wind_direction+180)%360 - 180
        smooth_wake_shape = self.dynamic_state_shape
        xyval = np.array([(x+self.op_dynamic_state_margin)*self.len_ratio, (y+self.op_dynamic_state_margin)*self.len_ratio, wind_direction_change]).T
        if self.parallel_dynamic_computations:
            # Parallelized
            smooth_wind_direction = self.pool.map(functools.partial(_op_get_gaussian_filter, r=r, smooth_wake_shape=smooth_wake_shape), xyval)
            smooth_wind_direction = np.sum(smooth_wind_direction, axis=0)
            dynamic_state_direction = smooth_wind_direction + np.ones_like(smooth_wind_direction)*np.deg2rad(self.wind_process.wind_direction)
        else:
            # Vectorized (just a gaussian, no gaussian filter)
            wake_gaussians = _op_centered_gaussians(smooth_wake_shape, xyval[:,0], xyval[:,1], r, wind_direction_change)
            smooth_wind_direction = np.sum(wake_gaussians, axis=0)
            dynamic_state_direction = smooth_wind_direction + np.ones_like(smooth_wind_direction)*np.deg2rad(self.wind_process.wind_direction)

        if self.verbose: print("Direction computation time", time.time()-t0)

        dynamic_state = {
            "speed": dynamic_state_speed,
            "direction": dynamic_state_direction,
        }
        if self.verbose: print("Dynamic state shape: ", dynamic_state["speed"].shape)
        if self.verbose: print("Dynamic direction shape: ", dynamic_state["direction"].shape)
        return dynamic_state
    
    def _op_get_gaussian_filter(self, xyv, r=None, smooth_wake_shape=None):
        x, y, val = xyv
        temp = np.zeros(np.array(smooth_wake_shape) + 2*r)
        # for each point in a circle of radius 100, set the value to u
        wake_point = get_circle_idxs(r) * val

        x0, y0 = x+self.op_dynamic_state_margin*self.len_ratio, y+self.op_dynamic_state_margin*self.len_ratio
        x0, y0 = int(x0), int(y0)
        try:
            temp[x0:x0+2*r+1, y0:y0+2*r+1] = wake_point
        except:
            print("Error in setting wake point")
            print(x0, y0, temp.shape, wake_point.shape, smooth_wake_shape)
        temp = temp[r:-r, r:-r]
        return gaussian_filter(temp, sigma=r)

    def _op_turbine_update(self):
        self.t_power_log = []
        for i, ((coord, turbine), yaw) in enumerate(zip(self.floris_interface.floris.farm.flow_field.turbine_map.items, self.yaws_from_north)):
            # Force recomputing the closest points given the flow points
            turbine.flow_field_point_indices = None
            x, y = coord.x1, coord.x2
            r = turbine.rotor_radius

            # Points along the turbine
            yaw = np.deg2rad(-yaw)
            # Add 5 points behind the turbine
            turbine_wind_measurement_points = []
            for dr in np.linspace(-r, r, 5):
                c = np.array([coord.x1, coord.x2, coord.x3])
                point = c + dr*np.array([np.cos(yaw), np.sin(yaw), 0])
                # Offset the point in the direction perpendicular to the yaw
                # by a length of 10
                angle = (yaw + np.pi/2)
                turbine_wind_point = point + 40*np.array([np.cos(angle), np.sin(angle), 0])
                # check if the points are behind of the turbine
                _x = int(turbine_wind_point[0] + self.op_dynamic_state_margin)
                _y = int(turbine_wind_point[1] + self.op_dynamic_state_margin)
                turbine_wind_measurement_points.append([_x, _y])

                if self.verbose:
                    t_back_wind_point = point - 40*np.array([np.cos(angle), np.sin(angle), 0])
                    print("Trubine speed [Front | Back] (front should be more than back usually) ")
                    print(self.dynamic_state["speed"].T[_x, _y], self.dynamic_state["speed"].T[int(t_back_wind_point[0] + self.op_dynamic_state_margin), int(t_back_wind_point[1] + self.op_dynamic_state_margin)])
            
            # Transposing because the computations are the same as 
            # in plotting, so this is a good way to check if the points are correct
            turbine_wind = np.array([self.dynamic_state["speed"][int(x*self.len_ratio),int(y*self.len_ratio)] for x, y in turbine_wind_measurement_points])
            turbine_wind[turbine_wind < 1] = 1
            turbine_wind_angles = np.array([self.dynamic_state["direction"][int(x*self.len_ratio),int(y*self.len_ratio)] for x, y in turbine_wind_measurement_points])
            wind_coordinates = np.array([[x-self.op_dynamic_state_margin,y-self.op_dynamic_state_margin,coord.x3] for x, y in turbine_wind_measurement_points])
            turbine.velocities = turbine.calculate_swept_area_velocities(turbine_wind, coord, wind_coordinates[:,0], wind_coordinates[:,1], wind_coordinates[:,2])
            
            incident_angles = np.rad2deg(turbine_wind_angles) - self.yaws_from_north[i]
            incident_angles_corrected = (incident_angles + 180) % 360 - 180
            mean_incident_angle = np.mean(incident_angles_corrected)
            if self.verbose: 
                print("Turbine wind: ", turbine_wind)
                print("Turbine wind angles: ", np.rad2deg(turbine_wind_angles)%360)
                print("Turbine yaws", self.yaws_from_north[i])
                print("Incident angles: ", incident_angles)
                print("Incident angles corrected: ", incident_angles)

            # Compute power
            turbine = self.turbines[i]
            # Computing power here because original code considers the direction
            # at which the wind hits the turbine.
            pW = turbine.pP / 3.0 
            cos_incident_angle = np.cos(np.deg2rad(mean_incident_angle))
            cos_incident_angle = np.clip(cos_incident_angle, 0, 1) # Clip to avoid numerical errors
            if self.verbose: print("Cos incident angle: ", cos_incident_angle)
            yaw_effective_velocity = turbine.average_velocity * cos_incident_angle ** pW
            #power = turbine.air_density * turbine.powInterp(yaw_effective_velocity) * turbine.turbulence_parameter
            power = turbine.air_density * turbine.powInterp(yaw_effective_velocity) * turbine.turbulence_parameter
            if self.verbose:
                print("Turbine: ", i)
                print("Pp, Pw, v: ", turbine.pP, pW, turbine.average_velocity)
                # c, c, _
                print(f"Yaw effective velocity: {yaw_effective_velocity}")
                print("Air density: ", turbine.air_density)
                # Constant = 1.255
                print("Turbulence: ", turbine.turbulence_parameter)
                # Constant = 1
                print("Power: ", turbine.power)
                print("My power: ", power)
                print("pwinterp: ", turbine.powInterp(yaw_effective_velocity))
            self.t_power_log.append(power)
        return self.t_power_log

    def _op_get_mast_state(self):
        wind_info = []
        for x, y in np.array(self.mast_layout).T:
            x = int((x + self.op_dynamic_state_margin) * self.len_ratio)
            y = int((y + self.op_dynamic_state_margin) * self.len_ratio)
            wind_speed = self.dynamic_state["speed"][x, y]
            wind_direction = self.dynamic_state["direction"][x, y]
            wind_info.extend([wind_speed, wind_direction])
        return wind_info
    
    def _op_render(self):
        fig = plt.figure(frameon=False)
        # Render has to be transposed to match the original plots
        # Wind farm shape:
        shape = (self.wf_bounds[1][0] - self.wf_bounds[0][0], self.wf_bounds[1][1] - self.wf_bounds[0][1])
        shape = (shape[0] + 2*self.op_dynamic_state_margin, shape[1] + 2*self.op_dynamic_state_margin)
        dynamic_plot_shape = (int(shape[0]), int(shape[1]))
        state_img = cv2.resize(self.dynamic_state["speed"], dynamic_plot_shape)
        im = plt.imshow(state_img.T)
        l = 20
        x = np.linspace(self.op_dynamic_state_margin/4, dynamic_plot_shape[0]-1, l)
        y = np.linspace(self.op_dynamic_state_margin/4, dynamic_plot_shape[1]-1, l)
        X, Y = np.meshgrid(x, y)
        dirs = [self.dynamic_state["direction"][int(x*self.len_ratio), int(y*self.len_ratio)] for x, y in zip(X.flatten(), Y.flatten())]
        dirs = [np.deg2rad((d-90)) for d in dirs]
        U = -np.sin(dirs).reshape(X.shape)*20
        V = -np.cos(dirs).reshape(Y.shape)*20
        plt.quiver(X, Y, U, V, units='xy', scale=1, color='lightblue')
        # Text at every xy with direction value
        # for x, y, d in zip(X.flatten(), Y.flatten(), dirs):
        #     plt.text(x, y, f"{int(d)}", color='black', fontsize=4) 
        # Add line for each turbine in the plot
        for (coord, turbine), yaw in zip(self.floris_interface.floris.farm.flow_field.turbine_map.items, self.yaws_from_north):
            x, y = coord.x1, coord.x2
            r = turbine.rotor_radius
            yaw = np.deg2rad(-yaw)
            x0 = int(x + r*np.cos(yaw) + self.op_dynamic_state_margin)
            y0 = int(y + r*np.sin(yaw) + self.op_dynamic_state_margin)
            x1 = int(x - r*np.cos(yaw) + self.op_dynamic_state_margin)
            y1 = int(y - r*np.sin(yaw) + self.op_dynamic_state_margin)
            plt.plot([x0, x1], [y0, y1], 'black', lw=2)

            # Add 5 points behind the turbine
            for j, dr in enumerate(np.linspace(-r, r, 5)):
                c = np.array([coord.x1, coord.x2, coord.x3])
                point = c + dr*np.array([np.cos(yaw), np.sin(yaw), 0])
                # Offset the point in the direction perpendicular to the yaw
                angle = (yaw + np.pi/2)
                offset = r / self.op_per_turbine
                point = point + offset*np.array([np.cos(angle), np.sin(angle), 0])
                # check if the points are behind of the turbine
                _x = int(point[0] + self.op_dynamic_state_margin)
                _y = int(point[1] + self.op_dynamic_state_margin)
                plt.plot(_x, _y, 'o', markersize=3, color='grey')

        # For each mast plot a cross
        for x, y in np.array(self.mast_layout).T:
            x = int(x + self.op_dynamic_state_margin)
            y = int(y + self.op_dynamic_state_margin)
            plt.plot(x, y, 'rx', markersize=5)
        
        # For each observation point plot an arrow in the point direction
        for x, y, u, v, angle in self.observation_points[["x", "y", "u", "v", "wind_direction"]].values:
            x = int(x + self.op_dynamic_state_margin)
            y = int(y + self.op_dynamic_state_margin)
            angle = -(angle + 90)
            arrow_l = 10
            # Only plot if it fits in the plot
            if x > dynamic_plot_shape[0] - 2*arrow_l or y > dynamic_plot_shape[1] - 2*arrow_l or x < 2*arrow_l or y < 2*arrow_l:
                continue
            plt.arrow(x, y, arrow_l*np.cos(np.deg2rad(angle)), arrow_l*np.sin(np.deg2rad(angle)), head_width=2, head_length=2, fc='k', ec='k')
            uv_angle = np.degrees(np.arctan2(v, u))
            # This should be within +- 90 degrees
            if uv_angle > 90: uv_angle = uv_angle - 180
            if uv_angle < -90: uv_angle = uv_angle + 180
            if self.verbose: 
                if np.abs(uv_angle) > 20: print("UV ANGLE", uv_angle)
            uv_angle = (angle + uv_angle) % 360
            plt.arrow(x, y, arrow_l*np.cos(np.deg2rad(uv_angle)), arrow_l*np.sin(np.deg2rad(uv_angle)), head_width=2, head_length=2, fc='orange', ec='orange')
        plt.axis('off')
        plt.box(False)
        #plt.colorbar(im)
        plt.gca().invert_yaxis()
        fig.subplots_adjust(bottom = 0)
        fig.subplots_adjust(top = 1)
        fig.subplots_adjust(right = 1)
        fig.subplots_adjust(left = 0)
        #plt.title("Dynamic state")
        
        canvas = plt.gca().figure.canvas
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        op_plt = data.reshape(canvas.get_width_height()[::-1] + (3,))
        plt.show()
        return op_plt

    def _get_closest_represented_speed(self, op):
        mat = self.wake_matrices[op["t"]]["wake_matrix"]
        dists = mat[["x", "y", "z"]].values - np.array([op["x"], op["y"], op["z"]])
        dists = np.square(dists).sum(axis=1)
        closest = mat.iloc[np.argmin(dists)]
        op["u"] = closest["u"]
        op["v"] = closest["v"]
        op["w"] = closest["w"]

        # The represented wind direction is constant throughout the wake
        # And the direction change because of the wake is encoded in u,v
        return op

def _op_get_gaussian_filter(xyv, r=None, smooth_wake_shape=None):
        x, y, val = xyv
        temp = np.zeros(np.array(smooth_wake_shape) + 2*r)
        # for each point in a circle of radius 100, set the value to u
        wake_point = get_circle_idxs(r) * val
        x0, y0 = int(x), int(y)
        try:
            temp[x0:x0+2*r+1, y0:y0+2*r+1] = wake_point
        except:
            print("Error in setting wake point")
            print(x0, y0, temp.shape, wake_point.shape, smooth_wake_shape)
        temp = temp[r:-r, r:-r]
        return gaussian_filter(temp, sigma=r)

#@functools.partial(jax.jit, static_argnums=(2,3,4))
def _op_get_gaussian_filter_jax(xyv, temp, r=None, smooth_wake_shape=None, margin=None):
    x, y, val = xyv
    # jax set x to int
    x = x.astype(int)
    y = y.astype(int)
    # for each point in a circle of radius 100, set the value to u
    # wake_point = get_circle_idxs(r) * val
    
    A = jnp.arange(-r,r+1)**2
    dists = jnp.sqrt(A[:,None] + A)
    circle = ((dists-r)<=0).astype(int)
    wake_point = circle * val

    x0, y0 = x+margin, y+margin
    # try:
    # temp[x0:x0+2*r+1, y0:y0+2*r+1] = wake_point
    temp = jax.lax.dynamic_update_slice(temp, wake_point, (x0, y0))
    # except:
    #     print("Error in setting wake point")
    #     print(x0, y0, temp.shape, wake_point.shape, smooth_wake_shape)
    #temp = temp[r:-r, r:-r]
    temp = jax.lax.dynamic_slice(temp, (r, r), smooth_wake_shape)
    filter_size = jnp.linspace(-r, r, 2*r+1)
    gaussian_filter = jax.scipy.stats.norm.pdf(filter_size,scale=r) * jax.scipy.stats.norm.pdf(filter_size[:, None],scale=r)
    smooth_wake = jax.scipy.signal.convolve(temp, gaussian_filter, mode='same')
    return smooth_wake

def _op_centered_gaussians(shape, x, y, r, value):
    Ax = np.arange(-r, int(shape[0] + r))
    Ay = np.arange(-r, int(shape[1] + r))
    Ax = np.tile(Ax, (len(x), 1))
    Ay = np.tile(Ay, (len(x), 1))
    Ax = Ax.T - x
    Ay = Ay.T - y
    Asumsq = Ax**2 + Ay[:,None]**2

    # Make gaussians and sum them
    gaussians = np.exp(-4*np.log(2) * Asumsq / (2.5*r)**2)
    gaussians = gaussians * value
    gaussians = np.transpose(gaussians, (2, 1, 0))
    gaussians = gaussians[:, r:-r, r:-r]
    return gaussians    