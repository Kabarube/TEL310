from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
import cv2
import matplotlib.pyplot as plt
import argparse

from SET_Prob import Simulator
from ray_cast import expected_scan, measured_scan
from sensor_models import (
    beam_range_finder_model,
    likelihood_field_range_finder_model,
    learn_intrinsic_parameters,
    landmark_model_known_correspondence,
    sample_landmark_model_known_correspondence
)


def create_simulator() -> Simulator:
    """
    Create the lab simulator with the assignment map and starting pose.

    Returns
    -------
    Simulator
        Simulator with an 800-by-800 camera, a 0.1-second time step,
        an 8-metre maximum sensor range, and pose [-2, -3, pi / 2].
    """

    # Map from assignment
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

    # Initialize parameters and simulation
    world_map = {'lines': [], 'circles': []}

    for p1, p2 in walls:
        world_map['lines'].append({
            'p1': np.array(p1, dtype=float),
            'p2': np.array(p2, dtype=float),
            'color': np.array([0, 0, 0])
        })

    # world_map["circles"].append({
    #     "center": np.array([2.0, 0.0]),
    #     "radius": 0.5,
    #     "color": np.array([255, 0, 0])
    # })

    robot_params = {
        "initial_pose": np.array([-2.0, -3.0, np.pi / 2]),
        "a_max": 0.5,
        "alpha_max": 0.5,
        "dt": 0.1,
        "radius": 0.15
    }

    camera_params = {
        "width": 800,
        "height": 800,
        "meters_per_pixel": 0.015
    }

    # Init simulator
    sim = Simulator(
        robot_init_pose=robot_params,
        cam_params=camera_params,
        time_step=0.1,
        world_map=world_map
    )

    sim.lidar_max_range = 8.0

    return sim

# Draw the sensor scan on the simulator image
def draw_scan(
    image: NDArray[np.uint8],
    sim: Simulator,
    ranges: Sequence[float] | NDArray[np.float64],
    color: tuple[int, int, int] = (255, 0, 0),
) -> NDArray[np.uint8]:
    """
    Draw evenly spaced range beams and endpoints on a copy of an image.

    Parameters
    ----------
    image : np.ndarray of shape (height, width, 3)
        Source image, which is copied before drawing.
    sim : Simulator
        Simulator providing the current robot pose and camera projection.
    ranges : sequence of float
        Beam ranges over a full rotation, starting at the robot's heading.
    color : tuple of int, optional
        Three-channel color passed directly to OpenCV. Default is (255, 0, 0).

    Returns
    -------
    np.ndarray
        Copy of image with beam lines and endpoint markers.
    """

    image = image.copy()
    pose = sim.robot.get_pose()
    x, y, theta = pose

    angles = np.linspace(0, 2*np.pi, len(ranges), endpoint=False)

    # Convert robot's world position into image pixels
    start = sim.camera.project_point([x, y])

    for angle, distance in zip(angles, ranges):
        global_angle = theta + angle

        # Calculate beam endpoint in world coordinates
        end_x = x + distance * np.cos(global_angle)
        end_y = y + distance * np.sin(global_angle)

        # Convert endpoint to pixel coordinates
        end = sim.camera.project_point([end_x, end_y])

        # Draw beam and intersection point
        cv2.line(image, start, end, color, 2)
        cv2.circle(image, end, 4, color, -1)

    return image

def plot_parameter_learning(history: NDArray[np.float64]) -> None:
    """
    Display the evolution of the six beam-model parameters during EM.

    Parameters
    ----------
    history : np.ndarray of shape (iterations + 1, 6)
        Initial and updated [z_hit, z_short, z_max, z_rand, sigma_hit,
        lambda_short] values, one row per iteration.
    """
    names = [
        r"$z_{\mathrm{hit}}$",
        r"$z_{\mathrm{short}}$",
        r"$z_{\mathrm{max}}$",
        r"$z_{\mathrm{rand}}$",
        r"$\sigma_{\mathrm{hit}}$",
        r"$\lambda_{\mathrm{short}}$"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    fig.suptitle(
        "Intrinsic Sensor Parameter Learning\n"
        "Expectation-Maximization (EM)",
        fontsize=15,
        fontweight="bold"
    )

    for i, ax in enumerate(axes.flat):
        ax.plot(history[:, i], marker=".")
        ax.set_title(names[i])
        ax.set_xlabel("EM iteration")
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def experiment_parameter_learning(
    sim: Simulator, rng: np.random.Generator,
) -> NDArray[np.float64]:
    """
    Collect 200 scans, fit beam-model parameters, and print the estimates.

    Parameters
    ----------
    sim : Simulator
        Simulator providing the fixed robot pose, map, and sensor range.
    rng : np.random.Generator
        Random number generator used to generate noisy training scans.

    Returns
    -------
    np.ndarray of shape (iterations + 1, 6)
        Initial parameters and subsequent EM estimates in the order
        [z_hit, z_short, z_max, z_rand, sigma_hit, lambda_short].
    """

    cats = [
        {"center": np.array([-1.0, -2.0]), "radius": 0.25},
        {"center": np.array([2.5, 0.0]), "radius": 0.25},
    ]

    z_measured, z_expected = collect_training_data(
        sim, cats, rng, num_scans=200
    )

    initial_params = [0.7, 0.1, 0.1, 0.1, 0.2, 1.0]

    learned_params, history = learn_intrinsic_parameters(
        z_measured,
        z_expected,
        sim.lidar_max_range,
        initial_params
    )

    names = [
        "z_hit", "z_short", "z_max", "z_rand",
        "sigma_hit", "lambda_short"
    ]

    for name, initial, learned in zip(
        names, initial_params, learned_params
    ):
        print(f"{name:15} {initial:.4f} -> {learned:.4f}")

    print(f"EM iterations: {len(history) - 1}")

    return history

def run_landmark_demo() -> None:
    """
    Display 500 robot positions sampled from a known landmark observation.

    Evaluate the observation at the true pose and include its likelihood
    in the plot title. Use a fixed random seed for reproducibility.
    """

    rng = np.random.default_rng(42)

    # [x, y, signature]
    landmarks = {0: np.array([2.0, 2.0, 1.0])}

    true_pose = np.array([0.0, 0.0, 0.0])

    # Simulated observation of landmark 0
    feature = np.array([np.sqrt(8), np.pi / 4, 1.0])

    sigma_r = 0.1
    sigma_phi = 0.05
    sigma_s = 0.1

    likelihood = landmark_model_known_correspondence(
        feature, 0, true_pose, landmarks,
        sigma_r, sigma_phi, sigma_s
    )

    samples = np.array([
        sample_landmark_model_known_correspondence(
            feature, 0, landmarks,
            sigma_r, sigma_phi, rng
        )
        for _ in range(500)
    ])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        samples[:, 0], samples[:, 1],
        s=8, alpha=0.4, label="Sampled robot poses"
    )
    ax.scatter(2, 2, marker="*", s=180, label="Known Landmark")
    ax.scatter(0, 0, marker="x", color='red', s=100, label="True robot pose")

    ax.set_aspect("equal")
    ax.set_title(
        "Landmark Model with Known Correspondence\n"
        f"Pose Sampling (Table 6.5) | "
        f"Measurement Likelihood: {likelihood:.3g}",
        fontsize=12
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc='upper right')
    plt.show()

def run_sensor_simulation(
    sim: Simulator,
    params: Sequence[float] | NDArray[np.float64],
    rng: np.random.Generator,
) -> None:
    """
    Run the interactive sensor simulation until the Q key is pressed.

    Display expected and noisy scans, moving cats, and beam and likelihood
    field scores. W/S adjust linear velocity, A/D adjust angular velocity,
    and H stops the robot. Close OpenCV windows on exit.

    Parameters
    ----------
    sim : Simulator
        Simulator advanced and rendered on each loop iteration.
    params : array_like of shape (6,)
        [z_hit, z_short, z_max, z_rand, sigma_hit, lambda_short].
    rng : np.random.Generator
        Random number generator used for noisy range measurements.
    """

    v, omega = 0.0, 0.0
    sensor_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)

    cats = [
        {"center": np.array([-1.0, -2.0]), "radius": 0.25},
        {"center": np.array([2.5, 0.0]), "radius": 0.25},
    ]

    t = 0.0
    while True:

        image = sim.step(v, omega)

        z_expected = expected_scan(sim)

        update_cats(cats, t)
        t += sim.dt

        z_measured = measured_scan(sim, cats, rng)
                
        q_beam = beam_range_finder_model(
            z_expected,
            z_measured,
            sim.lidar_max_range,
            params
        )

        q_field = likelihood_field_range_finder_model(
            z_measured,
            sim.robot.get_pose(),
            sim.world_map,
            sensor_angles,
            sim.lidar_max_range,
            z_hit=params[0],
            z_rand=params[3],
            sigma_hit=params[4]
        )

        for cat in cats:
            center_px = sim.camera.project_point(cat["center"])
            radius_px = int(cat["radius"] / sim.camera.meters_per_pixel)

            cv2.circle(
                image,
                center_px,
                radius_px,
                (255, 165, 0),
                -1
            )

        # 6. Draw expected and measured scans
        image = draw_scan(
            image, sim, z_expected, color=(0, 160, 0)
        )

        image = draw_scan(
            image, sim, z_measured, color=(255, 0, 0)
        )

        # 7. Display the likelihoods
        cv2.putText(
            image, f"Beam likelihood: {q_beam:.3g}",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 0), 2
        )

        cv2.putText(
            image, f"Field likelihood: {q_field:.3g}",
            (15, 60), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 0), 2
        )

        cv2.putText(
            image,
            "Green: Expected | Red: Measured | Orange: Cats",
            (350, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )

        cv2.imshow(
            "TEL310 - Ray Casting and Probabilistic Range Sensor Models",
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )

        # 8. Keyboard controls
        key = cv2.waitKey(30) & 0xFF

        if key == ord("w"):
            v += 0.1
        elif key == ord("s"):
            v -= 0.1
        elif key == ord("a"):
            omega += 0.1
        elif key == ord("d"):
            omega -= 0.1
        elif key == ord("h"):
            v, omega = 0.0, 0.0
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

def update_cats(cats: list[dict], t: float) -> None:
    """
    Update the first two cats' centers along fixed periodic trajectories.

    Parameters
    ----------
    cats : list of dict
        At least two circular obstacles. The first two "center" entries
        are replaced in place; other entries are left unchanged.
    t : float
        Simulation time in seconds.
    """

    cats[0]["center"] = np.array([
        -1.0 + 0.6 * np.sin(0.8 * t),
        -2.0 + 0.4 * np.cos(0.8 * t)
    ])

    cats[1]["center"] = np.array([
        2.5 + 0.5 * np.sin(0.5 * t),
        0.5 * np.cos(0.5 * t)
    ])

def collect_training_data(
    sim: Simulator,
    cats: list[dict],
    rng: np.random.Generator,
    num_scans: int = 200,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Collect paired beam ranges while moving cats around a fixed robot.

    Parameters
    ----------
    sim : Simulator
        Simulator providing the current pose, map, range limit, and dt.
        The robot pose is not advanced during collection.
    cats : list of dict
        At least two circular obstacles with "center" and "radius" entries.
        Their centers are updated in place for each scan.
    rng : np.random.Generator
        Random number generator used for measurement noise.
    num_scans : int, optional
        Number of eight-beam scans to collect. Default is 200.

    Returns
    -------
    z_measured : np.ndarray of shape (8 * num_scans,)
        Flattened noisy ranges including the cats.
    z_expected : np.ndarray of shape (8 * num_scans,)
        Matching flattened ranges predicted from the static map.
    """

    all_expected = []
    all_measured = []

    for i in range(num_scans):
        # Move cats through their trajectories
        t = i * sim.dt
        update_cats(cats, t)

        # Static-map prediction
        z_expected = expected_scan(sim)

        # Sensor observation including cats
        z_measured = measured_scan(sim, cats, rng)

        all_expected.extend(z_expected)
        all_measured.extend(z_measured)

    return (
        np.array(all_measured),
        np.array(all_expected)
    )

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="TEL310 Labwork 2",
        description=(
            "TEL310 - Probabilistic Robotics\n"
            "Labwork 2: Robot Perception and Sensor Models\n\n"
            "Python simulations of probabilistic range sensors, "
            "intrinsic parameter learning, and landmark observations."
        ),
        epilog=(
            "Demonstrations:\n"
            "  sensor    Interactive robot simulation with ray casting,\n"
            "            moving cats, and range sensor likelihoods.\n\n"
            "  learning  Estimate intrinsic sensor parameters using the\n"
            "            Expectation-Maximization (EM) algorithm.\n\n"
            "  landmark  Evaluate a landmark observation and visualize\n"
            "            sampled robot poses with known correspondence.\n\n"
            "Examples:\n"
            "  python main.py sensor\n"
            "  python main.py learning\n"
            "  python main.py landmark\n\n"
            "Sensor controls:\n"
            "  W / S    Increase / decrease linear velocity\n"
            "  A / D    Increase / decrease angular velocity\n"
            "  H        Stop robot\n"
            "  Q        Quit simulation"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'demo',
        choices=['sensor', 'learning', 'landmark'],
        help='Select demonstration'
    )
    args = parser.parse_args()

    if args.demo == 'sensor':
        # Create simulation environment
        sim = create_simulator()
        params = [0.7, 0.1, 0.1, 0.1, 0.05, 1.0]
        rng = np.random.default_rng(42)
        run_sensor_simulation(sim, params, rng)

    elif args.demo == 'learning':
        sim = create_simulator()
        rng = np.random.default_rng(42)
        history = experiment_parameter_learning(sim, rng)
        plot_parameter_learning(history)

    elif args.demo == 'landmark':
        run_landmark_demo()

if __name__ == '__main__':
    main()
