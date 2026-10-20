
"""
TEL310 - Probabilistic Robotics
Lab 2: Beam Range Finder Simulation

Uses the provided SET_Prob.py environment.
"""

import numpy as np
import matplotlib.pyplot as plt

from SET_Prob import Simulator
from sensor_models import BeamModelParams, beam_range_finder_model


# --------------------------------------------------
# Configuration
# --------------------------------------------------

N_BEAMS = 36
MAX_RANGE = 10.0
ROBOT_POSE = np.array([-2.0, -2.5, 0.0])

rng = np.random.default_rng(42)


# --------------------------------------------------
# Environment
# --------------------------------------------------

def create_map(include_cat=False):
    """Create the map in the format expected by SET_Prob.py."""

    walls = [
        # Outer boundary
        ((-5.0, -5.0), ( 5.0, -5.0)),
        (( 5.0, -5.0), ( 5.0,  5.0)),
        (( 5.0,  5.0), (-5.0,  5.0)),
        ((-5.0,  5.0), (-5.0, -5.0)),

        # Starting area
        ((-5.0, -1.0), (-3.0, -1.0)),
        ((-1.0, -1.0), (0.0, -1.0)),
        ((0.0, -1.0), (0.0, -5.0)),
        ((-5.0, -4.0), (-4.0, -4.0)),
        ((-4.0, -4.0), (-4.0, -5.0)),

        # Upper right area
        ((0.0, 5.0), (2.0, 1.0)),  # Diagonal
        ((3.0, 5.0), (3.0, 3.0)),  # Vertical
        ((4.0, 1.0), (5.0, 1.0)),  # Horizontas

        # Upper cross
        ((-3.5, 2.5), (-1.5, 2.5)),
        ((-2.5, 1.5), (-2.5, 3.5)),

        # Lower-right crossed walls
        ((1.5, -1.5), (3.5, -3.5)),
        ((1.5, -3.5), (3.5, -1.5)),

    ]

    world_map = {"lines": [], "circles": []}

    for p1, p2 in walls:
        world_map["lines"].append({
            "p1": p1,
            "p2": p2,
            "color": np.array([0, 0, 0]),
        })


    if include_cat:
        # Unknown obstacle: exists in the real environment
        # but is not included in the robot's map.
        world_map["circles"].append({
            "center": np.array([0.0, 4.0]),
            "radius": 0.30,
            "color": np.array([0, 0, 255]),
        })

    return world_map


# --------------------------------------------------
# Simulator setup
# --------------------------------------------------

def create_simulator(world_map):
    """Configure the original simulator for eight beams."""

    sim = Simulator(
        robot_init_pose={
            "initial_pose": ROBOT_POSE.copy(),
        },
        cam_params={
            "width": 800,
            "height": 800,
            "meters_per_pixel": 0.01,
        },
        time_step=0.1,
        world_map=world_map,
    )

    sim.lidar_max_range = MAX_RANGE

    # Eight beams evenly distributed around the robot
    sim.lidar_angles = np.linspace(
        -np.pi, np.pi, N_BEAMS, endpoint=False
    )

    return sim


def get_full_scan(sim):
    """
    Convert SET_Prob.py detections into a complete range array.

    The original get_lidar() omits beams that hit nothing.
    We represent these as maximum-range measurements.
    """

    scan = np.full(
        len(sim.lidar_angles),
        sim.lidar_max_range,
        dtype=float,
    )

    for reading in sim.get_lidar():
        index = np.argmin(
            np.abs(sim.lidar_angles - reading["angle"])
        )
        scan[index] = reading["distance"]

    return scan


# --------------------------------------------------
# Simulate measurements
# --------------------------------------------------

known_map = create_map(include_cat=False)
actual_map = create_map(include_cat=True)

expected_sim = create_simulator(known_map)
actual_sim = create_simulator(actual_map)

# Expected geometric ranges from the known map
z_expected = get_full_scan(expected_sim)

# Actual geometric ranges, including the unexpected cat
z_actual = get_full_scan(actual_sim)

# Add Gaussian sensor noise to non-maximum readings
z_observed = z_actual.copy()

valid = z_actual < MAX_RANGE
z_observed[valid] += rng.normal(
    0.0, 0.04, size=np.count_nonzero(valid)
)

z_observed = np.clip(z_observed, 0.0, MAX_RANGE)


# --------------------------------------------------
# Evaluate beam model
# --------------------------------------------------

params = BeamModelParams()

log_likelihood, beam_likelihoods = beam_range_finder_model(
    z_observed,
    z_expected,
    MAX_RANGE,
    params,
)

print("\nBEAM RANGE FINDER MODEL")
print("-" * 62)

for k, (z_star, z, p) in enumerate(
    zip(z_expected, z_observed, beam_likelihoods),
    start=1,
):
    print(
        f"Beam {k}: "
        f"expected={z_star:.3f} m, "
        f"observed={z:.3f} m, "
        f"likelihood={p:.5f}"
    )

print("-" * 62)
print(f"Total log-likelihood: {log_likelihood:.6f}")


# --------------------------------------------------
# Visualization
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]

# Draw known walls
for wall in known_map["lines"]:
    p1, p2 = wall["p1"], wall["p2"]
    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        color="black",
        linewidth=2,
    )

# Draw obstacles
for obstacle in actual_map["circles"]:
    center = obstacle["center"]
    radius = obstacle["radius"]

    is_cat = np.allclose(center, [1.5, 0.0])

    circle = plt.Circle(
        center,
        radius,
        color="royalblue" if is_cat else "gray",
        alpha=0.7,
    )
    ax.add_patch(circle)





# Draw expected and observed beams
for angle, z_star, z in zip(
    actual_sim.lidar_angles,
    z_expected,
    z_observed,
):
    global_angle = ROBOT_POSE[2] + angle

    direction = np.array([
        np.cos(global_angle),
        np.sin(global_angle),
    ])

    expected_endpoint = ROBOT_POSE[:2] + z_star * direction
    observed_endpoint = ROBOT_POSE[:2] + z * direction
    
    ax.plot(
        [ROBOT_POSE[0], expected_endpoint[0]],
        [ROBOT_POSE[1], expected_endpoint[1]],
        color="gray",
        linestyle="--",
        alpha=0.5,
    )

    ax.plot(
        [ROBOT_POSE[0], observed_endpoint[0]],
        [ROBOT_POSE[1], observed_endpoint[1]],
        color="red",
        linestyle="--",
        linewidth=1.5,
    )

ax.scatter(
    ROBOT_POSE[0],
    ROBOT_POSE[1],
    s=130,
    color="green",
    zorder=5,
    label="Robot",
)

ax.set_aspect("equal")
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title("Robot and LiDAR Measurements")
ax.grid(alpha=0.2)

# Compare expected and observed distances
ax = axes[1]
beam_numbers = np.arange(1, N_BEAMS + 1)

ax.plot(
    beam_numbers,
    z_expected,
    "o--",
    label="Expected range",
)

ax.plot(
    beam_numbers,
    z_observed,
    "o-",
    label="Observed range",
)

ax.set_xticks(beam_numbers)
ax.set_xlabel("Beam number")
ax.set_ylabel("Distance [m]")
ax.set_title("Expected vs. Observed LiDAR Scan")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
