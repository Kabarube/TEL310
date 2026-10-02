import numpy as np
import matplotlib.pyplot as plt
from motion_models import sample_motion_model_velocity, motion_model_velocity, sample_motion_model_odometry, motion_model_odometry


# Motion-model noise parameters
alpha_vel = (
    0.01,   # translational error due to v
    0.01,   # translational error due to omega
    0.02,   # rotational error due to v
    0.02,   # rotational error due to omega
    0.01,   # final rotation error due to v
    0.01,   # final rotation error due to omega
)

# Motion parameters
initial_pose = (0.0, 0.0, 0.0)  # Initial robot pose: x, y, theta
control = (1.0, 0.5)            # velocity, angular velocity
dt = 1.0 


# Simulation parameters
number_of_particles = 500
number_of_steps = 40


# Particle generator
def generate_particles(sampling_function, number_of_particles, *args):
    """Generate a set of particles by repeatedly sampling from a provided model.
    Args:
        sampling_function (callable): A function that generates a single particle
            when called with the supplied positional arguments.
        number_of_particles (int): The number of particles to generate.
        *args: Additional positional arguments forwarded to
            `sampling_function`.
    Returns:
        numpy.ndarray: A NumPy array containing all generated particles.
    """
    return np.array([
        sampling_function(*args)
        for _ in range(number_of_particles)
    ])


# =================================
# Sample motion velocity simulation
# =================================

# Generate particles of possible posterior poses
velocity_particles = generate_particles(
    sample_motion_model_velocity,
    number_of_particles,
    control,
    initial_pose,
    alpha_vel,
    dt
)

# ========================
# Noise free expected pose
# ========================
v, omega = control
x, y, theta = initial_pose

r = v / omega
theta_expected = theta + omega * dt

x_expected = x - r * (
    np.sin(theta) - np.sin(theta_expected)
)

y_expected = y + r * (
    np.cos(theta) - np.cos(theta_expected)
)


# ============================
# Motion model velocity simulation
# - Finding probability of different candidate poses
# ============================


# Building a grid of candidate poses
x_values = np.linspace(0.0, 1.5, 100)
y_values = np.linspace(-0.3, 0.8, 100)

theta_candidate = initial_pose[2] + control[1] * dt

likelihood = np.zeros(
    (len(y_values), len(x_values))
)

# Evaluation probability of each candidate pose
for iy, y in enumerate(y_values):
    for ix, x in enumerate(x_values):

        candidate_pose = (
            x,
            y,
            theta_candidate
        )

        p = motion_model_velocity(
            candidate_pose,
            initial_pose,
            control,
            alpha_vel,
            dt
        )

        if np.isfinite(p):
            likelihood[iy, ix] = p

# ================================
# Sample Odometry Model Simulation 
# ================================

# Odometry parameters
# - In this case odometry sensor says the robot moved from x_t-1=(0, 0, 0) to xt=(0.5, 0.0, 0.4)
odometry = (
    (0.0, 0.0, 0.0),
    (0.5, 0.7, 0.4),
)

# Odometry noise parameters
alpha_odom = (
    0.01,   # translational error due to v
    0.01,   # translational error due to omega
    0.02,   # rotational error due to v
    0.02,   # rotational error due to omega
    0.01,   # final rotation error due to v
    0.01,   # final rotation error due to omega
)

# Generate particles
odometry_particles = generate_particles(
    sample_motion_model_odometry,
    number_of_particles,
    odometry,
    initial_pose,
    alpha_odom
)

theta_candidate = odometry[1][2]

# Building a grid of candidate poses
x_values_odom = np.linspace(-0.2, 1.2, 100)
y_values_odom = np.linspace(-0.5, 1.0, 100)

odom_likelihood = np.zeros(
    (len(y_values_odom), len(x_values_odom))
)

x_bar_prev, x_bar_t = odometry

for iy, y in enumerate(y_values_odom):
    for ix, x in enumerate(x_values_odom):

        candidate_pose = (
            x,
            y,
            theta_candidate
        )

        p = motion_model_odometry(
            initial_pose,
            candidate_pose,
            x_bar_prev,
            x_bar_t,
            alpha_odom
        )

        if np.isfinite(p):
            odom_likelihood[iy, ix] = p

# ========
# Plotting
# ========

fig, ax = plt.subplots(2, 2, figsize=(10, 10))

# Starting position
ax[0, 0].scatter(
    initial_pose[0],
    initial_pose[1],
    s=80,
    c='orange',
    label='Initial pose',
)

# Plot posterior probable poses
ax[0, 0].scatter(
    velocity_particles[:, 0],
    velocity_particles[:, 1],
    alpha = 0.40,
    s=6,
    label='Particles'
)

# Noise free pose
ax[0, 0].scatter(
    x_expected,
    y_expected,
    s=100,
    marker='o',
    c='r',
    label='Expected pose'
)

im = ax[0, 1].imshow(
    likelihood,
    aspect='auto',
    origin='lower',
    extent=[
        x_values[0],
        x_values[-1],
        y_values[0],
        y_values[-1]
    ],
)

fig.colorbar(im, ax=ax[0, 1], label='Probability Density')

ax[0, 1].scatter(
    initial_pose[0],
    initial_pose[1],
    marker='o',
    c='orange',
    label='Initial Pose'
)

ax[1, 0].scatter(
    odometry_particles[:, 0],
    odometry_particles[:, 1],
    s=6,
    alpha=0.4,
    label='Odometry particles'
)

ax[1, 0].scatter(
    initial_pose[0],
    initial_pose[1],
    s=80,
    label='Initial pose'
)

im_odom = ax[1, 1].imshow(
    odom_likelihood,
    origin='lower',
    extent=[
        x_values_odom[0],
        x_values_odom[-1],
        y_values_odom[0],
        y_values_odom[-1]
    ],
    aspect='auto'
)

ax[1, 1].scatter(
    initial_pose[0],
    initial_pose[1],
    marker='o',
    c='orange',
    label='Initial pose'
)



fig.colorbar(
    im_odom,
    ax=ax[1, 1],
    label='Probability Density'
)

ax[0, 0].set(xlabel='x', ylabel='y', title='Motion Sampling Model')
ax[0, 1].set(xlabel='x', ylabel='y', title='Motion Probability Model')
ax[1, 0].set(xlabel='x', ylabel='y', title='Odometry Sampling Model')
ax[1, 1].set(xlabel='x', ylabel='y', title='Odometry Probability Model')
ax[0, 0].legend()
ax[0, 1].legend()
ax[1, 0].legend()
ax[1, 1].legend()


plt.show()
