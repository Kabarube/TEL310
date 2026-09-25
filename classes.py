from abc import ABC, abstractmethod
from motion_models import sample_motion_model_velocity, sample_motion_model_odometry

class Robot:
    def __init__(self, pose, motion_model): 
        self.pose = tuple(pose)
        self.motion_model = motion_model

    def step(self, control, dt):
        self.pose = self.motion_model.sample(
            self.pose,
            control,
            dt,
        )

class MotionModel:
    @abstractmethod
    def sample(self, pose, control, dt):
        raise NotImplementedError


class VelocityMotionModel(MotionModel):
    def __init__(self, alpha):
        self.alpha = alpha

    def sample(self, pose, control, dt):
        return sample_motion_model_velocity(
            control,
            pose,
            self.alpha,
            dt,
        )

class OdometryMotionModel(MotionModel):
    def __init__(self, alpha):
        self.alpha = alpha

    def sample(self, pose, control):
        return sample_motion_model_odometry(
            control,
            pose,
            self.alpha,
        )