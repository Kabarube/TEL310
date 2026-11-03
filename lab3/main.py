import numpy as np
import cv2
from simulation_env import Simulator
from ekf import correction_step, prediction_step, wrap_angle, associate_landmark
from ukf import ukf_correction_step, ukf_prediction_step, ukf_measurement_sigma_points
import argparse

def create_simulator() -> Simulator:
    """Create a simulator with boundary walls and default robot and camera settings.

    Returns:
        Simulator: The initialized simulation environment.
    """

    world_map = {
        'lines': [],
        'circles': []
    }

    walls = [
        ((8, -8), (-8, -8), (0, 0, 0)),
        ((-8, -8), (-8, 8), (0, 0, 0)),
        ((-8, 8), (8, 8), (0, 0, 0)),
        ((8, 8), (8, -8), (0, 0, 0))
    ]

    for p1_world, p2_world, color in walls:
        world_map['lines'].append({
            'p1': np.array(p1_world),
            'p2': np.array(p2_world),
            'color': np.array(color)
        })

    robot_params = {
        'initial_pose': np.array([0.0, 0.0, 0.0]),
        'a_max': 0.5,
        'alpha_max': 0.5,
        'dt': 0.1,
        'radius': 0.3,
    }

    camera_params = {
        'width': 800,
        'height': 800,
        'meters_per_pixel': 0.025
    }

    sim = Simulator(
        robot_init_pose=robot_params,
        cam_params=camera_params,
        time_step=0.1,
        world_map=world_map,
    )

    return sim

def create_landmark(sim: Simulator) -> dict:
    """Add known landmarks to the simulator's map.

    Args:
        sim (Simulator): The simulation environment to add landmarks to.

    Returns:
        dict: Landmark IDs mapped to arrays containing [x, y, signature].
    """

    coordinates = [
        # Bottom landmarks
        (-8, -8), (-4, -8), (0, -8), (4, -8), (8, -8),
        # Top landmarks
        (-8, 8), (-4, 8), (0, 8), (4, 8), (8, 8),
        # Left side landmarks
        (-8, -4), (-8, 0), (-8, 4),
        # Right siden landmarks
        (8, -4), (8, 0), (8, 4)
    ]

    # rng for colors
    rng = np.random.default_rng(42)
    landmarks = {}

    for i, (x, y) in enumerate(coordinates):
        landmarks[i] = np.array([x, y, i])
        # Draw with simulation environment
        sim.world_map['circles'].append({
            'center': np.array([x, y]),
            'radius': 0.4,
            'color': rng.integers(low=0, high=255, size=3)
        })

    return landmarks

def observe_landmark(
    true_pose: np.ndarray,
    landmark: np.ndarray,
    Q: np.ndarray,
    rng: np.random.Generator
) -> np.ndarray:
    """Generate a noisy observation of a landmark.

    Args:
        true_pose (np.ndarray): Robot pose [x, y, theta].
        landmark (np.ndarray): Landmark data [x, y, signature].
        Q (np.ndarray): Measurement noise covariance.
        rng (np.random.Generator): Random number generator for measurement noise.

    Returns:
        np.ndarray: Noisy [range, bearing, signature], with bearing in radians.
    """
    x, y, theta = true_pose
    mx, my, signature = landmark

    # Relative pos to landmark
    dx = mx - x
    dy = my - y

    # Expected range
    r = np.hypot(dx, dy)
    phi = np.arctan2(dy, dx) - theta

    # Ideal observation
    z_expected = np.array([r, phi, signature])

    # Add noise
    noise = rng.multivariate_normal(
        mean=np.zeros(3),
        cov=Q
    )

    # Observation with noise
    z = z_expected + noise
    z[1] = wrap_angle(z[1]) # Normalize to [-pi, pi]

    return z

def is_landmark_visible(
        true_pose: np.ndarray,
        landmark: np.ndarray,
        sensor_range: float,
        sensor_fov: float
    ) -> bool:

    """Check whether a landmark is within sensor range and field of view.

    Args:
        true_pose (np.ndarray): Robot pose [x, y, theta].
        landmark (np.ndarray): Landmark data [x, y, signature].
        sensor_range (float): Maximum detection distance in meters.
        sensor_fov (float): Sensor field of view in radians.

    Returns:
        bool: True if the landmark is within both limits, otherwise False.
    """

    x, y, theta = true_pose
    mx, my, _ = landmark

    dx = mx - x
    dy = my - y

    # Distance to landmark
    distance = np.hypot(dx, dy)

    if distance > sensor_range:
        return False

    # Bearing relative to robot heading
    bearing = wrap_angle(
        np.arctan2(dy, dx) - theta
    )

    # Field-of-view test
    if abs(bearing) > sensor_fov / 2:
        return False

    return True

def draw_path(
    image: np.ndarray,
    sim: Simulator,
    path: list,
    color: tuple
) -> np.ndarray:
    """Draw a trajectory on the simulation image.

    Args:
        image (np.ndarray): Image to draw on.
        sim (Simulator): Simulator used to convert world positions to pixels.
        path (list): Sequence of [x, y] positions in world coordinates.
        color (tuple): Line color in BGR order.

    Returns:
        np.ndarray: The input image with the trajectory drawn on it.
    """

    if len(path) < 2:
        return image

    points = np.array([
        sim.camera.project_point(position)
        for position in path
    ], dtype=np.int32)

    cv2.polylines(
        image,
        [points],
        isClosed=False,
        color=color,
        thickness=2
    )

    return image

def draw_estimated_robot(image: np.ndarray, sim: Simulator, mu: np.ndarray) -> np.ndarray:
    """Draw the estimated robot position and heading in red.

    Args:
        image (np.ndarray): Image to draw on.
        sim (Simulator): Simulator used to convert world positions to pixels.
        mu (np.ndarray): Estimated robot pose [x, y, theta].

    Returns:
        np.ndarray: The input image with the estimated robot drawn on it.
    """

    estimated_pixel = sim.camera.project_point(mu[:2])

    # Estimated position (red)
    cv2.circle(
        image,
        estimated_pixel,
        6,
        (0, 0, 255),
        -1
    )

    # Estimated heading
    heading_length = 0.4

    heading_endpoint = mu[:2] + heading_length * np.array([
        np.cos(mu[2]),
        np.sin(mu[2])
    ])

    heading_pixel = sim.camera.project_point(heading_endpoint)

    cv2.line(
        image,
        estimated_pixel,
        heading_pixel,
        (0, 0, 255),
        2
    )

    return image

def draw_covariance_ellipse(
    image: np.ndarray,
    sim: Simulator,
    mu: np.ndarray,
    sigma: np.ndarray
) -> np.ndarray:
    """Draw a 95% confidence ellipse for the estimated robot position.

    Args:
        image (np.ndarray): Image to draw on.
        sim (Simulator): Simulator used to convert world coordinates to pixels.
        mu (np.ndarray): Estimated robot pose [x, y, theta].
        sigma (np.ndarray): Pose covariance matrix, with shape (3, 3).

    Returns:
        np.ndarray: The input image with the confidence ellipse drawn on it.
    """

    # 1. Extract the 2D position covariance
    covariance_xy = sigma[:2, :2]

    # 2. Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_xy)

    # 3. Sort eigenvalues from largest to smallest
    order = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # 4. Compute ellipse semiaxes (95% confidence)
    chi2_95 = 5.991

    axes_meters = np.sqrt(
        chi2_95 * np.maximum(eigenvalues, 0.0)
    )

    # 5. Convert metres to pixels
    axes_pixels = axes_meters / sim.camera.meters_per_pixel

    # 6. Calculate major-axis direction
    major_axis = eigenvectors[:, 0]

    angle = np.degrees(
        np.arctan2(major_axis[1], major_axis[0])
    )

    # Image y-axis points downward
    angle = -angle

    # 7. Convert estimated position to image pixels
    center = sim.camera.project_point(mu[:2])

    # 8. Draw ellipse
    cv2.ellipse(
        image,
        center,
        tuple(np.maximum(1, np.round(axes_pixels)).astype(int)),
        angle,
        0,
        360,
        (255, 0, 0),
        2
    )

    return image

def parse_args() -> argparse.Namespace:
    """Parse command-line options for the localization simulation.

    Returns:
        argparse.Namespace: Selected algorithm, sensor range in meters,
            and field of view in degrees.
    """
    parser = argparse.ArgumentParser(
        description='TEL310 Lab3 - Mobile Robot Localization'
    )

    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['ekf-known', 'ekf-unknown', 'ukf'],
        default='ekf-known',
        help='Localization algorithm'
    )

    parser.add_argument(
        '--sensor-range',
        type=float,
        default=9.0,
        help='Maximum landmark detection range in meter'
    )

    parser.add_argument(
        '--fov',
        type=float,
        default=90.0,
        help='Sensor field of view in deg.'
    )
    return parser.parse_args()


def main() -> None:
    sim = create_simulator()
    landmarks = create_landmark(sim)

    args = parse_args()

    algorithm = args.algorithm
    sensor_range = args.sensor_range
    sensor_fov = np.deg2rad(args.fov)


    # Initial estimate of robot pose
    mu = sim.robot.get_pose().copy()

    # Initial uncertainty in the estimated pose
    sigma = np.diag([
        0.5**2,
        0.5**2,
        np.deg2rad(5)**2
    ])

    # Motion-noise parameters
    alpha = np.array([0.1, 0.1, 0.1, 0.1])

    # Measurement-noise covariance
    Q = np.diag([
        0.1**2,
        0.05**2,
        0.1**2
    ])

    # Random number generator
    rng = np.random.default_rng(42)

    # Robot controls
    desired_v = 0.5
    desired_omega = 0.2

    # Trajectory history
    true_path = []
    estimated_path = []

    landmark = landmarks[0]
    while True:

        # Move simulated robot
        image = sim.step(desired_v, desired_omega)

        # Get true pose and applied control
        true_pose = sim.robot.get_pose().copy()
        u = np.array(sim.robot.get_vel())

        log_pzt = 0.0

        # Prediction
        if algorithm in ("ekf-known", "ekf-unknown"):
            mu, sigma = prediction_step(
                mu, u, sigma, alpha, sim.dt
            )

        elif algorithm == "ukf":
            mu, sigma, X_pred, X_aug, weights = ukf_prediction_step(
                mu, sigma, u, alpha, Q, sim.dt
            )

        mu[2] = wrap_angle(mu[2])

        
        # EKF correction
        observed_landmarks = 0

        for landmark_id, landmark in landmarks.items():

            # Check sensor visibility
            if not is_landmark_visible(
                true_pose, landmark, sensor_range, sensor_fov
            ):
                continue

            # Generate measurement
            z = observe_landmark(
                true_pose, landmark, Q, rng
            )

            # Determine which landmark to use
            if algorithm == "ekf-known":
                selected_landmark = landmark

            elif algorithm == "ekf-unknown":
                associated_id = associate_landmark(
                    mu, sigma, z, landmarks, Q
                )
                selected_landmark = landmarks[associated_id]

                if associated_id != landmark_id:
                    print(
                        f"Wrong association! "
                        f"Expected: {landmark_id}, "
                        f"Selected: {associated_id}"
                    )

            if algorithm in ("ekf-known", "ekf-unknown"):
                mu, sigma, log_pz = correction_step(
                    mu, sigma, z, selected_landmark, Q
                )

            elif algorithm == "ukf":
                # Regenerate sigma points from the current corrected belief
                X_state, X_noise, w = ukf_measurement_sigma_points(
                    mu, sigma, Q
                )

                mu, sigma, log_pz = ukf_correction_step(
                    mu, sigma, X_state, X_noise, w, z[:2], landmark
                )

            log_pzt += log_pz

            mu[2] = wrap_angle(mu[2])

            observed_landmarks += 1

            # Draw detected landmark and sensor line
            robot_pixel = sim.camera.project_point(
                true_pose[:2]
            )

            landmark_pixel = sim.camera.project_point(
                landmark[:2]
            )

            cv2.line(
                image,
                robot_pixel,
                landmark_pixel,
                (0, 180, 0),
                1
            )

            cv2.circle(
                image,
                landmark_pixel,
                9,
                (0, 200, 0),
                2
            )

        pzt = np.exp(log_pzt)
        print(
            f"Algorithm: {algorithm} | "
            f"Log-likelihood: {log_pzt:.4f} | "
            f"Likelihood: {pzt:.6e}"
        )
        # Save trajectories
        true_path.append(true_pose[:2].copy())
        estimated_path.append(mu[:2].copy())

        # Draw true trajectory (blue)
        image = draw_path(
            image, sim, true_path, (255, 0, 0)
        )

        # Draw estimated trajectory (red)
        image = draw_path(
            image, sim, estimated_path, (0, 0, 255)
        )

        # 8. Draw estimated robot
        image = draw_estimated_robot(image, sim, mu)
        # Draw EKF uncertainty
        image = draw_covariance_ellipse(image, sim, mu, sigma)

        # 9. Calculate position error
        position_error = np.linalg.norm(
            mu[:2] - true_pose[:2]
        )

        # 10. Display diagnostics
        cv2.putText(
            image,
            f"Position error: {position_error:.3f} m",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

        cv2.putText(
            image,
            f"Detected landmarks: {observed_landmarks}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

        print(
            f"Landmarks: {observed_landmarks:2d} | "
            f"Position error: {position_error:.3f} m | "
            f"Sigma x: {np.sqrt(sigma[0, 0]):.3f} m | "
            f"Sigma y: {np.sqrt(sigma[1, 1]):.3f} m"
        )

        # 11. Display simulator
        cv2.imshow("TEL310 - Lab 3", image)

        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
