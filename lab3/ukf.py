import numpy as np
from ekf import gaussian_log_likelihood


def generate_sigma_points(mu_aug: np.ndarray, sigma_aug: np.ndarray, gamma: float) -> np.ndarray:
    """Generate sigma points from an augmented Gaussian distribution.

    Args:
        mu_aug (np.ndarray): Augmented mean, with shape (n,).
        sigma_aug (np.ndarray): Augmented covariance, with shape (n, n).
        gamma (float): Scale factor for the distance of points from the mean.

    Returns:
        np.ndarray: Sigma points stored as columns, with shape (n, 2*n + 1).
    """
    # Number of dimensions
    n = len(mu_aug)

    # Matrix square root of covariance
    sqrt_sigma = np.linalg.cholesky(sigma_aug)

    # Allocate output matrix
    sigma_points = np.zeros((n, 2 * n + 1))

    # First sigma point is the mean
    sigma_points[:, 0] = mu_aug

    # Remaining sigma points
    for i in range(n):
        offset = gamma * sqrt_sigma[:, i]

        sigma_points[:, i + 1] = mu_aug + offset
        sigma_points[:, i + n + 1] = mu_aug - offset

    return sigma_points

def motion_model(pose: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """Propagate one robot pose using the velocity motion model.

    Args:
        pose (np.ndarray): Robot pose [x, y, theta].
        u (np.ndarray): Linear and angular velocities [v, omega].
        dt (float): Time interval in seconds.

    Returns:
        np.ndarray: Predicted robot pose [x, y, theta].
    """

    x, y, theta = pose
    v, omega = u

    if np.isclose(omega, 0.0):
        return np.array([
            x + v * np.cos(theta) * dt,
            y + v * np.sin(theta) * dt,
            theta
        ])

    theta_new = theta + omega * dt
    radius = v / omega

    return np.array([
        x + radius * (np.sin(theta_new) - np.sin(theta)),
        y + radius * (np.cos(theta) - np.cos(theta_new)),
        theta_new
    ])

def ukf_prediction_step(
    mu: np.ndarray,
    sigma: np.ndarray,
    u: np.ndarray,
    alpha: np.ndarray,
    Q: np.ndarray,
    dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Predict the robot pose and covariance using augmented sigma points.

    Args:
        mu (np.ndarray): Current robot pose [x, y, theta].
        sigma (np.ndarray): Current pose covariance, with shape (3, 3).
        u (np.ndarray): Linear and angular velocities [v, omega].
        alpha (np.ndarray): Four motion noise coefficients.
        Q (np.ndarray): Measurement noise covariance; uses range and bearing entries.
        dt (float): Time interval in seconds.

    Returns:
        tuple: Predicted pose, predicted covariance, predicted pose sigma points,
            augmented sigma points, and sigma point weights.
    """

    v, omega = u
    a1, a2, a3, a4 = alpha

    # 1. Motion noise covariance
    M = np.diag([
        a1 * v**2 + a2 * omega**2,
        a3 * v**2 + a4 * omega**2
    ])

    # 2. Augmented mean: [x, y, theta, noise_v,
    #                     noise_omega, noise_r, noise_phi]
    mu_aug = np.concatenate([mu, np.zeros(4)])

    # 3. Augmented covariance
    sigma_aug = np.zeros((7, 7))
    sigma_aug[:3, :3] = sigma
    sigma_aug[3:5, 3:5] = M
    sigma_aug[5:7, 5:7] = Q[:2, :2]

    # 4. Sigma point parameters
    L = len(mu_aug)
    lambda_ = 1.0
    gamma = np.sqrt(L + lambda_)

    # 5. Sigma point weights
    weights = np.full(2 * L + 1, 1.0 / (2 * (L + lambda_)))
    weights[0] = lambda_ / (L + lambda_)

    # 6. Generate augmented sigma points
    X_aug = generate_sigma_points(mu_aug, sigma_aug, gamma)

    # 7. Propagate each sigma point
    X_pred = np.zeros((3, 2 * L + 1))

    for i in range(2 * L + 1):
        pose_i = X_aug[:3, i]
        noise_u = X_aug[3:5, i]

        X_pred[:, i] = motion_model(pose_i, u + noise_u, dt)

    # 8. Calculate predicted mean
    mu_pred = X_pred @ weights

    # Circular mean for heading
    mu_pred[2] = np.arctan2(
        np.sum(weights * np.sin(X_pred[2, :])),
        np.sum(weights * np.cos(X_pred[2, :]))
    )

    # 9. Calculate predicted covariance
    sigma_pred = np.zeros((3, 3))

    for i in range(2 * L + 1):
        diff = X_pred[:, i] - mu_pred

        # Wrap heading difference
        diff[2] = (diff[2] + np.pi) % (2 * np.pi) - np.pi

        sigma_pred += weights[i] * np.outer(diff, diff)

    return mu_pred, sigma_pred, X_pred, X_aug, weights

def measurement_model(pose: np.ndarray, landmark: np.ndarray) -> np.ndarray:
    """Predict range and bearing to a landmark.

    Args:
        pose (np.ndarray): Robot pose [x, y, theta].
        landmark (np.ndarray): Landmark data whose first two entries are [x, y].

    Returns:
        np.ndarray: Expected [range, bearing], with bearing in radians in [-pi, pi).
    """

    x, y, theta = pose
    mx, my = landmark[:2]

    dx = mx - x
    dy = my - y

    r = np.hypot(dx, dy)
    phi = np.arctan2(dy, dx) - theta

    phi = (phi + np.pi) % (2 * np.pi) - np.pi

    return np.array([r, phi])

def ukf_correction_step(
    mu_pred: np.ndarray,
    sigma_pred: np.ndarray,
    X_pred: np.ndarray,
    X_aug: np.ndarray,
    weights: np.ndarray,
    z: np.ndarray,
    landmark: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Correct the predicted pose and covariance using one landmark observation.

    Args:
        mu_pred (np.ndarray): Predicted robot pose [x, y, theta].
        sigma_pred (np.ndarray): Predicted pose covariance, with shape (3, 3).
        X_pred (np.ndarray): Pose sigma points stored as columns.
        X_aug (np.ndarray): Matching augmented sigma points, with range and
            bearing noise in rows 5 and 6.
        weights (np.ndarray): Weight for each sigma point.
        z (np.ndarray): Observation whose first two entries are [range, bearing].
        landmark (np.ndarray): Landmark data whose first two entries are [x, y].

    Returns:
        tuple: Corrected pose, corrected covariance, and measurement log-likelihood.
    """

    n_points = X_pred.shape[1]

    # 1. Transform predicted sigma points into measurements
    Z_pred = np.zeros((2, n_points))

    for i in range(n_points):

        Z_pred[:, i] = measurement_model(
            X_pred[:, i],
            landmark
        )

        # Measurement noise from augmented sigma points
        Z_pred[:, i] += X_aug[5:7, i]

        # Normalize bearing
        Z_pred[1, i] = (
            (Z_pred[1, i] + np.pi) % (2 * np.pi)
        ) - np.pi

    # 2. Calculate expected measurement
    z_expected = Z_pred @ weights

    # Circular mean for bearing
    z_expected[1] = np.arctan2(
        np.sum(weights * np.sin(Z_pred[1, :])),
        np.sum(weights * np.cos(Z_pred[1, :]))
    )

    # 3. Calculate innovation covariance S
    S = np.zeros((2, 2))

    # 4. Calculate cross-covariance
    sigma_xz = np.zeros((3, 2))

    for i in range(n_points):

        dz = Z_pred[:, i] - z_expected
        dx = X_pred[:, i] - mu_pred

        # Wrap angular differences
        dz[1] = (dz[1] + np.pi) % (2 * np.pi) - np.pi
        dx[2] = (dx[2] + np.pi) % (2 * np.pi) - np.pi

        S += weights[i] * np.outer(dz, dz)
        sigma_xz += weights[i] * np.outer(dx, dz)

    # 5. Kalman gain
    K = np.linalg.solve(S, sigma_xz.T).T

    # 6. Measurement innovation
    innovation = z[:2] - z_expected
    innovation[1] = (
        (innovation[1] + np.pi) % (2 * np.pi)
    ) - np.pi

    # Measurement log-likelihood
    log_pz = gaussian_log_likelihood(innovation, S)


    # 7. Correct pose
    mu = mu_pred + K @ innovation
    mu[2] = (mu[2] + np.pi) % (2 * np.pi) - np.pi

    # 8. Correct covariance
    sigma = sigma_pred - K @ S @ K.T

    # Enforce numerical symmetry
    sigma = 0.5 * (sigma + sigma.T)

    return mu, sigma, log_pz

def ukf_measurement_sigma_points(
    mu: np.ndarray,
    sigma: np.ndarray,
    Q: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sigma points for a measurement update from the current belief.

    Args:
        mu (np.ndarray): Current robot pose [x, y, theta].
        sigma (np.ndarray): Current pose covariance, with shape (3, 3).
        Q (np.ndarray): Measurement noise covariance; uses range and bearing entries.

    Returns:
        tuple: Pose sigma points, compatible augmented sigma points with range
            and bearing noise in rows 5 and 6, and sigma point weights.
    """

    L = 5  # 3 pose variables + 2 measurement noise variables
    lambda_ = 1.0
    gamma = np.sqrt(L + lambda_)

    mu_aug = np.concatenate([mu, np.zeros(2)])

    sigma_aug = np.zeros((L, L))
    sigma_aug[:3, :3] = sigma
    sigma_aug[3:5, 3:5] = Q[:2, :2]

    weights = np.full(2 * L + 1, 1.0 / (2 * (L + lambda_)))
    weights[0] = lambda_ / (L + lambda_)

    X_aug = generate_sigma_points(mu_aug, sigma_aug, gamma)

    # The correction function expects measurement noise at indices 5:7.
    # Construct a compatible augmented array.
    X_compatible = np.zeros((7, 2 * L + 1))
    X_compatible[:3, :] = X_aug[:3, :]
    X_compatible[5:7, :] = X_aug[3:5, :]

    return X_aug[:3, :], X_compatible, weights

if __name__ == "__main__":

    mu = np.array([0.0, 0.0, 0.0])
    sigma = np.diag([
        0.5**2,
        0.5**2,
        np.deg2rad(5)**2
    ])

    u = np.array([0.5, 0.2])
    alpha = np.array([0.1, 0.1, 0.1, 0.1])
    Q = np.diag([0.1**2, 0.05**2])

    # UKF prediction
    mu_pred, sigma_pred, X_pred, X_aug, weights = (
        ukf_prediction_step(mu, sigma, u, alpha, Q, 0.1)
    )

    # Landmark and simulated measurement
    landmark = np.array([4.0, 4.0, 0.0])

    z = measurement_model(mu_pred, landmark)

    # UKF correction
    mu_corr, sigma_corr = ukf_correction_step(
        mu_pred, sigma_pred, X_pred, X_aug, weights, z, landmark
    )

    print("Predicted pose:", mu_pred)
    print("Corrected pose:", mu_corr)

    print("\nPosition covariance before correction:")
    print(sigma_pred[:2, :2])

    print("\nPosition covariance after correction:")
    print(sigma_corr[:2, :2])

    print("\nCovariance symmetric:",
          np.allclose(sigma_corr, sigma_corr.T))

    print("Covariance PSD:",
          np.all(np.linalg.eigvalsh(sigma_corr) >= -1e-10))
