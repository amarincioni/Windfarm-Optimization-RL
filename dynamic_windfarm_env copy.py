from wind_farm_env.wind_farm_gym import WindFarmEnv
from wind_processes import RandomResetWindProcess, SetSequenceWindProcess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


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
        load_pyglet_visualization=False,
        update_rule=None,
        momentum_alpha=0.95,
        momentum_beta=1,
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

        # Saving states for wind momentum
        self.current_wind_state = None
        self.current_plot_state = None

        # Defines the points to sample the wind from
        self.recorded_points = self.initialize_recorded_points()
        if self.verbose:
            for k in self.recorded_points.keys():
                print(f"Points for {k}: {self.recorded_points[k].shape}")
        self.steady_state = None
        self.steady_state_plot = None
        self.dynamic_state = None
        self.dynamic_state_plot = None

        # Initialize visualization if enabled
        if self.load_pyglet_visualization:
            self.visualization = self.FarmVisualization(self.floris_interface, flow_points=self._flow_points())

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
            if False:
                num_points = num_points * 2
                N_ANGLES = 16
                # For each angle record a square grid of points
                for angle in np.linspace(0, 360, N_ANGLES):
                    # Record num_points points in a square grid
                    for r in np.linspace(-pt, pt, num_points):
                        # record the point
                        x = coord.x1 + r * np.cos(np.deg2rad(angle))
                        y = coord.x2 + r * np.sin(np.deg2rad(angle))

                        # Ignore points too close to the turbine
                        if x**2 + y**2 < 9:
                            continue

                        # record at different z heights
                        for h in np.linspace(-pt, pt, num_points):
                            assert coord.x3 == turbine.hub_height, "Turbine hub height is not the same as the coordinate"
                            z = coord.x3 + h
                            recorded_points[f"turbine_{i}"].append((x, y, z))
            else:
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

    def reset(self, wind_direction=None, **kwargs):
        obs = super().reset()

        # Initialize renderable directions
        if wind_direction is not None:
            self.wind_process.wind_direction = wind_direction
        #self.wind_process.wind_direction = 155 + (int(self.runs * (295-155)/8) % (295-155))
        
        # Update environment after reset (wind speed and direction are logged and used later, so this is necessary)
        self.floris_interface.reinitialize_flow_field(wind_speed=self.wind_process.wind_speed, wind_direction=self.wind_process.wind_direction)

        # Initialize turbine directions randomly, but towards the wind
        for turbine in self.turbines:
            turbine.yaw_angle = self._np_random.uniform(self.desired_min_yaw, self.desired_max_yaw)
            #turbine.yaw_angle = 0

        power_output = np.sum(self.floris_interface.get_turbine_power())
        info = {'power_output': np.nan_to_num(power_output, nan=0)}

        self.step_count = 0
        self.runs += 1
        if self.verbose:
            print("Run count:", self.runs)
            print("Wind speed:", self.wind_process.wind_speed)
            print("Wind direction:", self.wind_process.wind_direction)

        # if self.update_rule is not None:
        #     self.current_wind_state = self.get_wind_state()
            
        #     if self.load_pyglet_visualization:
        #         if self.current_plot_state is None:
        #             self.current_plot_state = self.visualization.get_cut_plane().df
        if self.update_rule is not None:
            self.steady_state = self.get_steady_state()
            if self.load_pyglet_visualization:
                self.steady_state_plot = self.visualization.get_cut_plane().df

        return obs, info

    def step(self, action):

        if self.update_rule is None:
            obs, reward, done, info = super().step(action)

            # Add power output to info
            power_output = np.sum(self.floris_interface.get_turbine_power())
            info['power_output'] = np.nan_to_num(power_output, nan=0)
        else:
            obs, reward, done, info = self.modified_step(action)
        

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

    def get_wind_state(self):
        for coord, turbine in self.floris_interface.floris.farm.flow_field.turbine_map.items:
            print(f"Turbine {coord}: {turbine.yaw_angle}")
        1/0
        return {
            "u": self.floris_interface.floris.farm.flow_field.u,
            "v": self.floris_interface.floris.farm.flow_field.v,
            "w": self.floris_interface.floris.farm.flow_field.w,
            "u_initial": self.floris_interface.floris.farm.flow_field.u_initial,
        }
    
    def set_wind_state(self, wind_state):
        self.floris_interface.floris.farm.flow_field.u_initial = wind_state["u_initial"]
        self.floris_interface.floris.farm.flow_field.u = wind_state["u"]
        self.floris_interface.floris.farm.flow_field.v = wind_state["v"]
        self.floris_interface.floris.farm.flow_field.w = wind_state["w"]

    def set_modified_wind_state(self, wind_state):
        self.floris_interface.floris.farm.flow_field.u_initial = wind_state["u"]
        self.floris_interface.floris.farm.flow_field.u = wind_state["u"]
        self.floris_interface.floris.farm.flow_field.v = wind_state["v"]
        self.floris_interface.floris.farm.flow_field.w = wind_state["w"]

    def update_turbine_velocities_with_wind_state(self, wind_state):
        
        for coord, turbine in self.floris_interface.floris.farm.flow_field.turbine_map.items:
            # Force floris to find the closest points 
            # from the wind flow we provide
            turbine.flow_field_point_indices = None 

            # Get the perpendicular wind to the turbine
            # TODO: rotate to the turbines yaw
            u_at_turbine = wind_state["u"]
            turbine_location = coord
            x, y, z = self.wind_sampling_points
            turbine.velocities = turbine.calculate_swept_area_velocities(u_at_turbine, turbine_location, x, y, z)
            



    def get_reward(self):
        # Idea to set turbines and get power
        # 1) Set flow_field variables to the stored self.current_wind_state
        # 2) for each turbine, t.update_velocities(u_wake=0, coord, flow_field, rotated_xyz)
        #   2a) unfortunately, turbulence computations are done in the wake calculations
        #       and affect the power output (reward) so that also would need to be fixed
        # 3) sum the t.power for each turbine 
        #  [info, rewards can be computed from it]

        # Since this idea is not feasible we set u_initial to u

        # Store floris values and set flow_field to our values
        temp_wind_state = self.get_wind_state()
        self.set_modified_wind_state(self.current_wind_state)
        
        # Compute reward
        # Update flow field and turbine velocities/turbulence
        self.floris_interface.calculate_wake()
        power = np.sum(self.floris_interface.get_turbine_power())
        if np.isnan(power): 
            power = 0
        reward = power * self._reward_scaling_factor

        # Reset flow field to the original state
        self.set_wind_state(temp_wind_state)

        # How to get observation
        # 1) Set flow_field from self.current_wind_state
        # 2) So something like wfgym
        #  a) take the flow_points() and get the u,v,w values like _get_measurement_point_data()
        #  a2) Will need to REWRITE floris_interface.get_set_of_points() to get the u,v,w values
        #       without recomputing the wake and flow field (should be pretty easy)
        #  b) take the yaws just like wfgym

        # self.set_wind_state(self.current_wind_state)
        # power = [turbine.power for turbine in self.floris.farm.turbines]
        info = {'power_output': np.sum(power)}

        # reward = np.sum(self.floris_interface.get_turbine_power(no_wake=True))
        return reward, info

    def get_dynamic_reward(self):
        # For each turbine, update velocities using stored dynamic values
        for i, (coord, turbine) in enumerate(self.floris_interface.floris.farm.flow_field.turbine_map.items):
            # Force recomputing the closest points given the flow points
            turbine.flow_field_point_indices = None
            turbine_wind = self.dynamic_state[f"turbine_{i}"]["u"].values
            wind_coordinates = self.recorded_points[f"turbine_{i}"]
            turbine.velocities = turbine.calculate_swept_area_velocities(turbine_wind, coord, wind_coordinates[:,0], wind_coordinates[:,1], wind_coordinates[:,2])
        
        # Get power, sum it up and return rewards and info
        total_power = np.sum(self.floris_interface.get_turbine_power())
        if np.isnan(total_power): total_power = 0
        reward = total_power * self._reward_scaling_factor
        info = {'power_output': np.sum(total_power)}
        return reward, info


    def _get_modified_state(self):
        print("Getting modified state", self._has_states)
        if self._has_states:
            self.current_flow_points = self._flow_points()
            if self.verbose: print(f"Flow points: {self.current_flow_points}")
            if len(self.current_flow_points[0]) > 0:
                # Store floris values and set flow_field to our values
                temp_wind_state = self.get_wind_state()
                self.set_modified_wind_state(self.current_wind_state)

                # This calls compute wake
                self._current_flow = self.floris_interface.get_set_of_points(*self.current_flow_points)
                if self.verbose: print(f"Flow: {self._current_flow}")

                # Reset flow field to the original state
                self.set_wind_state(temp_wind_state)
            
            if self.verbose: print(f"Observed variables: {self.observed_variables}")
            state = [self._get_measurement_point_data(d) for d in self.observed_variables]

            # inject noise
            # if self._perturbed_observations is not None:
            #     added_noise = np.zeros_like(state)
            #     added_noise.put(self._perturbed_observations, self._noise.step()['value'], mode='raise')
            #     added_noise = added_noise * self._perturbation_scale
            #     state = state + added_noise

            # rescale and clip off
            if self._normalize_observations:
                state = (np.array(state) - self.low) / self.state_delta
                state = np.clip(state, np.zeros_like(self.low), np.ones_like(self.high))
            else:
                state = np.clip(state, self.low, self.high)
            return list(state)
        else:
            return 0

    def get_dynamic_state(self):
        
        # Code taken from original environment _get_state()
        self.current_flow_points = self._flow_points()
        if len(self.current_flow_points[0]) > 0:
            self._current_flow = self.floris_interface.get_set_of_points(*self.current_flow_points)

        # Now only used for non mast observations
        state = [self._get_measurement_point_data(d) for d in self.observed_variables if "mast_" not in d["name"]]

        # Get dynamic mast observations from the stored values
        assert self.mast_observations == ('wind_speed', 'wind_direction'), "Only wind speed and direction are supported as mast observations"
        # For each mast, get the wind speed and direction
        for i in range(len(self.mast_layout[0])):
            #x, y = self.mast_layout[0][i], self.mast_layout[1][i]
            mast_df = self.dynamic_state[f"mast_{i}"]
            #print(f"Printing mast_{i}", mast_df, self.step_count)
            assert len(mast_df) == 1, "Mast should have only one point"
            u, v = mast_df["u"].values, mast_df["v"].values
            wind_speed = (u ** 2 + v ** 2) ** 0.5
            wind_direction = (np.degrees(np.arctan2(v, u)) + self._farm.wind_map.input_direction[0]) % 360
            state.extend([wind_speed, wind_direction])

        state = np.array([float(s) for s in state])
        # rescale and clip off
        if self._normalize_observations:
            state = (np.array(state) - self.low) / self.state_delta
            state = np.clip(state, np.zeros_like(self.low), np.ones_like(self.high))
        else:
            state = np.clip(state, self.low, self.high)
        
        return state

    def render(self, mode='human'):
        
        assert self.load_pyglet_visualization, "Visualization is disabled, pyglet is not loaded"

        if self.state is None:
            return None

        return self.visualization.render(return_rgb_array=mode == 'rgb_array', wind_state=self.dynamic_state_plot)

    def get_steady_state(self):
        # This becomes a dictionary of dataframes with columns (u, v, w, x1, x2, x3)
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

    def modified_step(self, action):
        # Adjust yaws of the environment
        obs, reward, done, info = super().step(action)

        ##### Modified dynamics ########
        # Get Floris state
        # floris_next_state = self.get_wind_state()
        # if self.load_pyglet_visualization:
        #     floris_next_plot_state = self.visualization.get_cut_plane().df
        # else:
        #     floris_next_plot_state = None
        self.steady_state = self.get_steady_state()
        if self.load_pyglet_visualization:
            self.steady_state_plot = self.visualization.get_cut_plane().df

        # Update state according to the update rule
        # self.current_wind_state, self.current_plot_state = self.update_state(floris_next_state, floris_next_plot_state)
        self.dynamic_state, self.dynamic_state_plot = self.update_dynamic_state(self.steady_state, self.steady_state_plot)
        ##### End modified dynamics #####

        # Compute rewards, info, observations given the new state
        #reward, info = self.get_reward()
        reward, info = self.get_dynamic_reward()
        #obs = self._get_modified_state()
        obs = self.get_dynamic_state()
        done = False

        return obs, reward, done, info

    # def update_state(self, floris_state, floris_plot_state=None):
    #     # Use the correct update rule
    #     if self.update_rule == 'momentum':
    #         return self._momentum_update(floris_state, floris_plot_state)
    #     elif self.update_rule == 'observation_points':
    #         raise NotImplementedError("Observation points update rule not implemented")
    #     else:
    #         raise NotImplementedError(f"Update rule {self.update_rule} not implemented")
        
    def update_dynamic_state(self, steady_state, steady_state_plot=None):
        # Use the correct update rule
        if self.update_rule == 'momentum':
            return self._momentum_update(steady_state, steady_state_plot)
        elif self.update_rule == 'observation_points':
            raise NotImplementedError("Observation points update rule not implemented")
        else:
            raise NotImplementedError(f"Update rule {self.update_rule} not implemented")

    def _momentum_update(self, steady_state, steady_state_plot=None):
        # next_state = {}
        # for k in floris_state.keys():
        #     if "x" not in k:
        #         next_state[k] = self.current_wind_state[k]*self.momentum_alpha + floris_state[k]*self.momentum_beta
        #     else:
        #         next_state[k] = floris_state[k]
        #     print(f"Floris {k}", next_state[k].shape)

        # if floris_plot_state is not None:
        #     next_plot_state = {}
        #     for k in floris_plot_state.keys():
        #         if "x" not in k:
        #             next_plot_state[k] = self.current_plot_state[k]*self.momentum_alpha + floris_plot_state[k]*self.momentum_beta
        #         else:
        #             next_plot_state[k] = floris_plot_state[k]
        #         print(f"Floris plot {k}", next_plot_state[k].shape)
        # else:
        #     next_plot_state = None

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
