import numpy as np

class SR_ProportionalController:
    def __init__(self, wind_directions=None, optimal_yaws=None):
        self.wind_directions = wind_directions if wind_directions is not None else np.linspace(0, 360, 120)
        self.optimal_yaws = optimal_yaws if optimal_yaws is not None else np.load("data/serial_refine/yaw_angles_opt_sr.npy")

    def predict(self, wind_direction, current_yaws):
        # Find the closest wind direction and aim for the optimal yaw
        # TODO: interpolate between the two closest wind directions
        # TODO: Distance should take into account the periodicity of the wind direction
        closest_idx = np.argmin(np.abs(self.wind_directions - wind_direction))
        target_yaw = self.optimal_yaws[closest_idx]
        
        yaw_diff = target_yaw - current_yaws
        #print(target_yaw, current_yaws, yaw_diff)
        action = yaw_diff * 0.5
        return action

if __name__ == "__main__":
    optimal_yaws = np.load("data/serial_refine/yaw_angles_opt_sr.npy")
    wind_directions = np.linspace(0, 360, 120)
    agent = SR_ProportionalController(wind_directions, optimal_yaws)