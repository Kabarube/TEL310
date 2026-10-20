
"""
TEL310 - Probabilistic Robotics
Lab 2: Initial beam sensor model experiment.

Run:
    python run_lab2.py
"""

import numpy as np
import matplotlib.pyplot as plt

from SET_Prob import Simulator
from sensor_models import (
    BeamModelParams,
    ray_cast,
    beam_range_finder_model
)


def make_map(include_cat=False):
    """
    Lab 2 map, approximated from the assignment figure.

    Static walls are known to the robot.
    Cats are dynamic obstacles absent from the known map.
    """
    walls = [
        # Outer boundary
        ((-5.0, -5.0), ( 5.0, -5.0)),
        (( 5.0, -5.0), ( 5.0,  5.0)),
        (( 5.0,  5.0), (-5.0,  5.0)),
        ((-5.0,  5.0), (-5.0, -5.0)),

        # Upper-left region
        ((-3.5,  2.5), (-1.5,  2.5)),
        ((-2.5,  2.0), (-2.5,  3.5)),

        # Upper-center diagonal wall
        (( 0.0,  5.0), ( 2.2,  1.0)),

        # Upper-right vertical wall
        (( 3.0,  5.0), ( 3.0,  3.0)),

        # Right wall segment
        (( 4.0, -0.2), ( 5.0, -0.2)),

        # Left-middle wall
        ((-5.0, -1.0), (-3.0, -1.0)),

        # Lower-middle partition
        ((-0.5, -1.0), ( 0.0, -1.0)),
        (( 0.0, -1.0), ( 0.0, -5.0)),

        # Lower-right crossed walls
        (( 1.5, -3.0), ( 3.5, -4.3)),
        (( 1.5, -4.5), ( 3.5, -2.7)),

        # Lower-left corner
        ((-5.0, -4.1), (-4.0, -4.1)),
        ((-4.0, -4.1), (-4.0, -5.0)),
    ]

    world_map = {"lines": [], "circles": []}

    for p1, p2 in walls:
        world_map["lines"].append({
            "p1": np.array(p1, dtype=float),
            "p2": np.array(p2, dtype=float),
            "color": np.array([0, 0, 0])
        })

    if include_cat:
        cats = [
            # Approximate cat positions in assignment
            (0.3, 4.3, 0.18),
            (2.0, -4.6, 0.18),
        ]

        for x, y, radius in cats:
            world_map["circles"].append({
                "center": np.array([x, y], dtype=float),
                "radius": radius,
                "color": np.array([0, 0, 255])
            })

    return world_map



def main():
    rng = np.random.default_rng(42)

    # Eight equally spaced beams around the robot.
    angles = np.linspace(
        0, 2 * np.pi, 8, endpoint=False
    )

    pose = np.array([0.0, 0.0, 0.0])

    known_map = make_map(include_cat=False)
    real_world = make_map(include_cat=True)

    sim = Simulator(
        robot_init_pose={"initial_pose": pose.copy()},
        cam_params={
            "width": 640,
            "height": 640,
            "meters_per_pixel": 0.005
        },
        time_step=0.1,
        world_map=real_world
    )

    sim.lidar_max_range = 2.0

    # Expected ranges from the known static map.
    expected = ray_cast(
        sim, pose, known_map, angles
    )

    # Actual geometrical ranges, including the cat.
    actual = ray_cast(
        sim, pose, real_world, angles
    )

    # Add Gaussian measurement noise.
    measured = np.clip(
        actual + rng.normal(0.0, 0.025, size=len(angles)),
        0.0,
        sim.lidar_max_range
    )

    params = BeamModelParams()

    log_likelihood = beam_range_finder_model(
        measured, expected, sim.lidar_max_range, params
    )

    print(f"Scan log-likelihood: {log_likelihood:.4f}")
    print("\nBeam | Expected | Actual | Measured")

    for k, (ze, za, zm) in enumerate(
        zip(expected, actual, measured), start=1
    ):
        print(f"{k:4} | {ze:8.3f} | {za:6.3f} | {zm:8.3f}")

    # Plot the known walls, cat and LiDAR beams.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for wall in known_map["lines"]:
        p1, p2 = wall["p1"], wall["p2"]
        ax1.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            "k-", linewidth=2
        )

    cat = real_world["circles"][0]
    ax1.add_patch(plt.Circle(
        cat["center"], cat["radius"],
        color="tab:blue", alpha=0.6
    ))

    ax1.plot(pose[0], pose[1], "go", markersize=10)

    for k, (angle, distance) in enumerate(zip(angles, measured)):
        direction = np.array([
            np.cos(pose[2] + angle),
            np.sin(pose[2] + angle)
        ])
        endpoint = pose[:2] + distance * direction

        ax1.plot(
            [pose[0], endpoint[0]],
            [pose[1], endpoint[1]],
            "r--", alpha=0.7
        )
        ax1.text(endpoint[0], endpoint[1], str(k + 1))

    ax1.set_title("Eight-beam LiDAR simulation")
    ax1.set_aspect("equal")
    ax1.set_xlim(-1.8, 1.8)
    ax1.set_ylim(-1.8, 1.8)
    ax1.grid(True)

    beam_numbers = np.arange(1, len(angles) + 1)

    ax2.plot(
        beam_numbers, expected,
        "o-", label="Expected (known map)"
    )
    ax2.plot(
        beam_numbers, measured,
        "s--", label="Measured (with cat)"
    )

    ax2.set_xlabel("Beam number")
    ax2.set_ylabel("Range [m]")
    ax2.set_xticks(beam_numbers)
    ax2.set_title("Expected vs measured distances")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
