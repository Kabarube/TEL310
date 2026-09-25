import random
import numpy as np
import matplotlib.pyplot as plt
from motion_models import sample_motion_model_velocity, sample_motion_model_odometry


random.seed(10)

# Motion-model noise parameters
alpha = (
    0.001,
    0.001,
    0.05,
    0.1,
    0.01,
    0.01,
)

# Initial robot pose: x, y, theta
initial_pose = (0.0, 0.0, 0.0)

odometry = (
    (0.0, 0.0, 0.0),
    (0.5, 0.0, 0.4),
)

# Motion command
control = (0.8, -0.5)  # velocity, angular velocity
dt = 0.1

number_of_particles = 500
number_of_steps = 40


# Create the particles that start at the initial pose
particles = [initial_pose for _ in range(number_of_particles)]

# Update particles each step
for step in range(number_of_steps):
    new_particles = []

    # For each step update all particles
    for particle in particles:
        new_pose = sample_motion_model_velocity(
            control,
            particle,
            alpha,
            dt
        )
        new_particles.append(new_pose)

    # Update set of particles with new probable positions
    particles = np.array(new_particles)

# Plot final particle distribution
plt.figure(figsize=(8, 6))

# Plot particles
plt.scatter(
    particles[:, 0],
    particles[:, 1],
    s=5,
    alpha=0.35,
    label="Particles",
)

# Plot initial position
plt.scatter(
    initial_pose[0],
    initial_pose[1],
    color='blue',
    s=80,
    label='Initial position'
)

# Mean estimated position
mean_x = np.mean(particles[:, 0])
mean_y = np.mean(particles[:, 1])

plt.scatter(
    mean_x,
    mean_y,
    color="red",
    s=80,
    label="Mean pose",
)

plt.xlabel("x position")
plt.ylabel("y position")
plt.title("Particle distribution after motion")
plt.axis("equal")
plt.legend()
plt.show()
