from classes import Robot, VelocityMotionModel, OdometryMotionModel

alpha = (
    0.005,
    0.005,
    0.05,
    0.05,
    0.01,
    0.01
)

initial_pose = (0, 0, 0)
steps = 40

motion_model = VelocityMotionModel(alpha)


robot = Robot(pose=initial_pose, motion_model=motion_model)

for _ in range(40):
    robot.step_velocity(
        control=(0.8, -0.7),
        dt=0.1
    )

print(robot.pose)