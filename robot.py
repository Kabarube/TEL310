

class Robot:
    def __init__(self):
        super().__init__()
        self.x = 0
        self.y = 0
        self.theta = 0
        self.v = 0
        self.omega = 0
        self.sensor_readings = []
        self.sensor_noise = 0.1
        self.motion_noise = 0.1

    @property
    def pose(self):
        return (self.x, self.y, self.theta)

    @pose.setter
    def set_pose(self, pose):
        self.x, self.y, self.theta = pose

    @property
    def velocity(self):
        return (self.v, self.omega)

    @velocity.setter
    def set_velocity(self, velocity):
        self.v, self.omega = velocity

    @property
    def sensor_readings(self):
        return self.sensor_readings

    @sensor_readings.setter
    def set_sensor_readings(self, readings):
        self.sensor_readings = readings

    @property
    def sensor_noise(self):
        return self.sensor_noise

    @sensor_noise.setter
    def set_sensor_noise(self, noise):
        self.sensor_noise = noise

    @property
    def motion_noise(self):
        return self.motion_noise

    @motion_noise.setter
    def set_motion_noise(self, noise):
        self.motion_noise = noise