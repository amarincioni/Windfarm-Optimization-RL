from wind_farm_gym.wind_process.wind_process import WindProcess
import numpy as np
from config import WIND_SPEED_RANGE, WIND_DIRECTION_CHANGE_RANGE, WIND_SPEED_CHANGE_RANGE

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
            self.wind_speed += np.random.uniform(*WIND_SPEED_CHANGE_RANGE)
            self.wind_direction += np.random.uniform(*WIND_DIRECTION_CHANGE_RANGE)

            # Clip values
            self.wind_speed = np.clip(self.wind_speed, *WIND_SPEED_RANGE)
            self.wind_direction = int(self.wind_direction) % 360

        return {'wind_speed': self.wind_speed, 'wind_direction': self.wind_direction}
    
    def reset(self):
        if self.set_wind_speed is not None:
            self.wind_speed = self.set_wind_speed
        else:
            self.wind_speed = np.random.uniform(*WIND_SPEED_RANGE)

        if self.sorted_wind:
            self.wind_direction = (self.wind_direction + 1) % 360
        else:
            self.wind_direction = int(np.random.uniform(0, 360)) 
            # 297 is correct, 296 works, 297 does not
            # 154 is ok, 153 is not
            # so input range is [154,297)
        return self.step()

class SetSequenceWindProcess(WindProcess):
    def __init__(self, wind_speeds, wind_directions):
        self.wind_speeds = wind_speeds
        self.wind_directions = wind_directions
        self.idx = 0

        self.wind_speed = 8.
        self.wind_direction = 270

    def step(self):
        wind_speed = self.wind_speeds[self.idx]
        wind_direction = self.wind_directions[self.idx]
        self.idx = (self.idx + 1) % len(self.wind_speeds)

        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        return {'wind_speed': wind_speed, 'wind_direction': wind_direction}
    
    def reset(self):
        self.idx = 0
        return self.step()