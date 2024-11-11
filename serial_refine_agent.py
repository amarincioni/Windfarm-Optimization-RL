import numpy as np
import matplotlib.pyplot as plt

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

def plot_yaw_angles(name, wind_directions, optimal_yaws):

    for i in range(optimal_yaws.shape[1]):
        plt.plot(wind_directions, optimal_yaws[:, i], label=f"Optimal yaw {i}")
    plt.title("Optimal yaw angles for different wind directions")
    plt.legend()
    plt.xlabel("Wind direction")
    plt.ylabel("Optimal yaw angle")
    plt.savefig(f"data/serial_refine/{name}_optimal_yaws.png")
    plt.show()

    # Show scatterplot of all optimal yaw angles
    for i in range(optimal_yaws.shape[1]):
        plt.scatter(wind_directions, optimal_yaws[:, i], label=f"Optimal yaw {i}")
    plt.title("Optimal yaw angles for different wind directions")
    plt.xlabel("Wind direction")
    plt.ylabel("Optimal yaw angle")
    plt.savefig(f"data/serial_refine/{name}_optimal_yaws_scatter.png")
    plt.show()

    # Show the difference between adjacent optimal yaws
    for i in range(optimal_yaws.shape[1]):
        plt.plot(wind_directions[:-1], np.diff(optimal_yaws[:, i]), label=f"Optimal yaw {i}")
    plt.title("Difference between adjacent optimal yaw angles")
    plt.xlabel("Wind direction")
    plt.ylabel("Yaw angle difference")
    plt.legend()
    plt.savefig(f"data/serial_refine/{name}_optimal_yaws_diff.png")
    plt.show()

    # Print the angles with the highest differences
    for i in range(optimal_yaws.shape[1]):
        diff = np.diff(optimal_yaws[:, i])
        max_diff_idx = np.argmax(diff)
        print(f"Max difference for yaw {i} at wind direction {wind_directions[max_diff_idx]}: {diff[max_diff_idx]}")




if __name__ == "__main__":
    optimal_yaws = np.load("data/serial_refine/yaw_angles_opt_sr.npy")
    wind_directions = np.linspace(0, 360, 120)
    plot_yaw_angles("serial_refine", wind_directions, optimal_yaws)

    optimal_yaws = np.load("data/serial_refine/lhs16_yaw_angles_opt.npy")
    wind_directions = np.linspace(0, 360, 120)
    plot_yaw_angles("lhs16", wind_directions, optimal_yaws)

    optimal_yaws = np.load("data/serial_refine/lhs8_yaw_angles_opt.npy")
    wind_directions = np.linspace(0, 360, 120)
    plot_yaw_angles("lhs8", wind_directions, optimal_yaws)

