import numpy as np
import cv2
from simulation_env import Simulator
from ekf import correction_step, prediction_step, wrap_angle

def create_simulator() -> Simulator:
    """Initialize simulation environment"""

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
    """Add knows landmarks to map"""

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

def observe_landmark(true_pose, landmark, Q, rng):
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
        true_pose,
        landmark,
        sensor_range,
        sensor_fov
    ) -> bool:

    """Check if landmark is withing FoV"""

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

def draw_path(image, sim, path, color):
    """Draw trajectory"""

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

def draw_estimated_robot(image, sim, mu):
    """Draw the estimated robot pose and heading."""

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

def draw_covariance_ellipse(image, sim, mu, sigma):
    """Draw a 95% confidence ellipse for robot position."""

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

def main():
    sim = create_simulator()
    landmarks = create_landmark(sim)

    # Sensor config
    sensor_range = 9.0
    sensor_fov = np.deg2rad(90)

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

        # EKF prediction
        mu, sigma = prediction_step(
            mu,
            u,
            sigma,
            alpha,
            sim.dt
        )

        # EKF correction
        observed_landmarks = 0

        for landmark_id, landmark in landmarks.items():

            # Check sensor visibility
            if not is_landmark_visible(
                true_pose,
                landmark,
                sensor_range,
                sensor_fov
            ):
                continue

            # Generate measurement
            z = observe_landmark(
                true_pose,
                landmark,
                Q,
                rng
            )

            # Correct pose and covariance
            mu, sigma = correction_step(
                mu,
                sigma,
                z,
                landmark,
                Q
            )

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

        # Save trajectories
        true_path.append(true_pose[:2].copy())
        estimated_path.append(mu[:2].copy())

        # Draw true trajectory (blue)
        image = draw_path(
            image,
            sim,
            true_path,
            (255, 0, 0)
        )

        # Draw estimated trajectory (red)
        image = draw_path(
            image,
            sim,
            estimated_path,
            (0, 0, 255)
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