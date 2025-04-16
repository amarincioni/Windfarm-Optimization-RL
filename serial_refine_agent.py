import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
        action = yaw_diff * 0.1
        return action

def plot_yaw_angles(name, wind_directions, optimal_yaws, title_name=""):

    for i in range(optimal_yaws.shape[1]):
        plt.plot(wind_directions, optimal_yaws[:, i], label=f"Turbine {i}")
    plt.title("Optimal yaw angles for different wind directions" + title_name, wrap=True)
    plt.legend()
    plt.xlabel("Wind direction (degrees)")
    plt.ylabel("Optimal yaw angle (degrees)")
    plt.savefig(f"data/serial_refine/{name}_optimal_yaws.png")
    plt.show()

    # Show scatterplot of all optimal yaw angles
    # for i in range(optimal_yaws.shape[1]):
    #     plt.scatter(wind_directions, optimal_yaws[:, i], label=f"Turbine {i}")
    # plt.title("Optimal yaw angles for different wind directions")
    # plt.xlabel("Wind direction (degrees)")
    # plt.ylabel("Optimal yaw angle (degrees)")
    # plt.savefig(f"data/serial_refine/{name}_optimal_yaws_scatter.png")
    # plt.show()

    # Show the difference between adjacent optimal yaws
    for i in range(optimal_yaws.shape[1]):
        plt.plot(wind_directions[:-1], np.diff(optimal_yaws[:, i]), label=f"Turbine {i}")
    plt.title("Difference between adjacent optimal yaw angles" + title_name, wrap=True)
    plt.xlabel("Wind direction (degrees)")
    plt.ylabel("Yaw angle difference (degrees)")
    plt.legend()
    plt.savefig(f"data/serial_refine/{name}_optimal_yaws_diff.png")
    plt.show()

    # Show the absolute difference
    for i in range(optimal_yaws.shape[1]):
        plt.plot(wind_directions[:-1], np.abs(np.diff(optimal_yaws[:, i])), label=f"Turbine {i}")
        
    # plt.gca().set_title('Normalized occupied \n Neighbors')
    plt.title("Absolute difference between adjacent \noptimal yaw angles" + title_name, wrap=True)
    plt.xlabel("Wind direction (degrees)")
    plt.ylabel("Yaw angle difference (degrees)")
    plt.legend()
    plt.savefig(f"data/serial_refine/{name}_optimal_yaws_abs_diff.png")
    plt.show()

    # Plot max of abs of all turbines
    max_diff = np.max(np.abs(np.diff(optimal_yaws, axis=0)), axis=1)
    # plt.plot(wind_directions[:-1], max_diff)
    # plt.title("Max difference between adjacent optimal yaw angles")
    # plt.xlabel("Wind direction (degrees)")
    # plt.ylabel("Max yaw angle difference (degrees)")
    # plt.savefig(f"data/serial_refine/{name}_optimal_yaws_max_diff.png")
    # plt.show()

    # Print the angles with the highest differences
    for i in range(optimal_yaws.shape[1]):
        diff = np.diff(optimal_yaws[:, i])
        max_diff_idx = np.argmax(diff)
        print(f"Max difference for yaw {i} at wind direction {wind_directions[max_diff_idx]}: {diff[max_diff_idx]}")

# Plot max abs diff for all three layouts
def plot_max_diff():
    # make 3 different subplots
    
    optimal_yaws = np.load("data/serial_refine/yaw_angles_opt_sr.npy")
    wind_directions = np.linspace(0, 360, 120)
    max_diff4 = np.max(np.abs(np.diff(optimal_yaws, axis=0)), axis=1)
    plot_yaw_angles("serial_refine", wind_directions, optimal_yaws)

    optimal_yaws = np.load("data/serial_refine/lhs16_yaw_angles_opt.npy")
    wind_directions = np.linspace(0, 360, 120)
    max_diff8 = np.max(np.abs(np.diff(optimal_yaws, axis=0)), axis=1)
    plot_yaw_angles("serial_refine", wind_directions, optimal_yaws)

    optimal_yaws = np.load("data/serial_refine/lhs8_yaw_angles_opt.npy")
    wind_directions = np.linspace(0, 360, 120)
    max_diff16 = np.max(np.abs(np.diff(optimal_yaws, axis=0)), axis=1)
    plot_yaw_angles("serial_refine", wind_directions, optimal_yaws)

    plt.plot(wind_directions[:-1], max_diff4, label="4Symm")
    plt.plot(wind_directions[:-1], max_diff8, label="LHS8")
    plt.plot(wind_directions[:-1], max_diff16, label="LHS16")
    plt.title("Max difference between adjacent optimal yaw angles")
    plt.xlabel("Wind direction (degrees)")
    plt.ylabel("Max yaw angle difference (degrees)")
    plt.legend()
    plt.savefig(f"data/serial_refine/max_diff.png")
    plt.show()

    threshold_count4 = [np.sum(max_diff4 > i) / max_diff4.shape[0] for i in np.linspace(0, 25, 30)]
    threshold_count8 = [np.sum(max_diff8 > i) / max_diff8.shape[0] for i in np.linspace(0, 25, 30)]
    threshold_count16 = [np.sum(max_diff16 > i) / max_diff16.shape[0] for i in np.linspace(0, 25, 30)]
    plt.plot(np.linspace(0, 25, 30), threshold_count4, label="4Symm")
    plt.plot(np.linspace(0, 25, 30), threshold_count8, label="LHS8")
    plt.plot(np.linspace(0, 25, 30), threshold_count16, label="LHS16")
    plt.title("Number of wind directions with max diff above threshold")
    plt.xlabel("Max yaw angle difference (degrees)")
    plt.ylabel("Fraction of wind directions")
    plt.legend()
    plt.savefig(f"data/serial_refine/max_diff_threshold.png")
    plt.show()



if __name__ == "__main__":
    # optimal_yaws = np.load("data/serial_refine/yaw_angles_opt_sr.npy")
    # wind_directions = np.linspace(0, 360, 120)
    # plot_yaw_angles("serial_refine", wind_directions, optimal_yaws)

    # optimal_yaws = np.load("data/serial_refine/lhs16_yaw_angles_opt.npy")
    # wind_directions = np.linspace(0, 360, 120)
    # plot_yaw_angles("lhs16", wind_directions, optimal_yaws)

    # optimal_yaws = np.load("data/serial_refine/lhs8_yaw_angles_opt.npy")
    # wind_directions = np.linspace(0, 360, 120)
    # plot_yaw_angles("lhs8", wind_directions, optimal_yaws)

    plot_max_diff()