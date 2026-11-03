import numpy as np


def prediction_step(
        mu: np.ndarray,
        u: np.ndarray,
        sigma: np.ndarray,
        alpha: np.ndarray,
        dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
    """Predict the robot pose and covariance using a velocity motion model.

    Args:
        mu (np.ndarray): Current pose [x, y, theta], with shape (3,).
        u (np.ndarray): Linear and angular velocities [v, omega].
        sigma (np.ndarray): Current pose covariance, with shape (3, 3).
        alpha (np.ndarray): Four motion noise coefficients.
        dt (float): Time interval over which to apply the motion.

    Returns:
        tuple[np.ndarray, np.ndarray]: Predicted pose, with shape (3,),
            and predicted covariance, with shape (3, 3).
    """
    x, y, theta = mu
    v, omega = u
    a1, a2, a3, a4 = alpha

    V = np.zeros((3, 2))
    G = np.eye(3)

    # Predict mean pose
    if np.isclose(omega, 0.0):  # Straight line case
        mu_pred = np.array([
            x + v * np.cos(theta) * dt,
            y + v * np.sin(theta) * dt,
            theta
        ])

        # Jacobian G: derivatives with respect to pose
        G[0, 2] = -v * dt * np.sin(theta)
        G[1, 2] =  v * dt * np.cos(theta)

        # Jacobian V: derivatives with respect to controls
        V[0, 0] = dt * np.cos(theta)
        V[1, 0] = dt * np.sin(theta)

        V[0, 1] = -0.5 * v * dt**2 * np.sin(theta)
        V[1, 1] =  0.5 * v * dt**2 * np.cos(theta)
        V[2, 1] = dt

    else:
        theta_new: float = theta + omega * dt
        radius: float = v / omega

        mu_pred = np.array([
            x + radius * (np.sin(theta_new) - np.sin(theta)),
            y + radius * (np.cos(theta) - np.cos(theta_new)),
            theta_new
        ])

        # Jacobian G
        G[0, 2] = radius * (np.cos(theta_new) - np.cos(theta))
        G[1, 2] = radius * (np.sin(theta_new) - np.sin(theta))

        # Jacobian V
        V[0, 0] = (np.sin(theta_new) - np.sin(theta)) / omega
        V[1, 0] = (np.cos(theta) - np.cos(theta_new)) / omega

        V[0, 1] = v * (((np.sin(theta) - np.sin(theta_new)) / omega**2) +
                    ((np.cos(theta_new) * dt) / omega))
        
        V[1, 1] = v * ((np.sin(theta_new) * dt / omega) -
                    (np.cos(theta) - np.cos(theta_new)) / omega**2)

        V[2, 1] = dt

    # Covariance matrix M
    M = np.diag([
        a1 * v**2 + a2 * omega**2,
        a3 * v**2 + a4 * omega**2
    ])

    # Predicted covariance
    sigma_pred = G @ sigma @ G.T + V @ M @ V.T

    return mu_pred, sigma_pred

def correction_step(
        mu_pred: np.ndarray,
        sigma_pred: np.ndarray,
        z: np.ndarray,
        landmark: np.ndarray,
        Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Correct the predicted pose and covariance using one landmark observation.

    Args:
        mu_pred (np.ndarray): Predicted robot pose [x, y, theta].
        sigma_pred (np.ndarray): Predicted pose covariance.
        z (np.ndarray): Observed [range, bearing, signature].
        landmark (np.ndarray): Known landmark data [x, y, signature].
        Q (np.ndarray): Measurement noise covariance.

    Returns:
        tuple: Corrected pose, corrected covariance, and measurement log-likelihood.
    """

    x, y, theta = mu_pred
    mx, my, ms = landmark

    # Relative position to landmark
    dx = mx - x
    dy = my - y
    q = dx**2 + dy**2

    if q <= 0:
        raise ValueError("Predited robot position coincides with landmark")

    z_expected = np.array([
        np.sqrt(q),
        np.arctan2(dy, dx) - theta,
        ms
    ])

    # Measurement Jacobian H
    H = np.array([
        [-dx / np.sqrt(q), -dy / np.sqrt(q), 0],
        [dy / q, -dx / q, -1],
        [0, 0, 0]
    ])

    # Measurement prediction uncertainty
    S = H @ sigma_pred @ H.T + Q

    # Kalman gain
    K = sigma_pred @ H.T @ np.linalg.inv(S)

    # Kalman innovation step
    innovation = z - z_expected

    innovation[1] = wrap_angle(innovation[1])

    # Measurement log-likelihood
    log_pz = gaussian_log_likelihood(
        innovation,
        S
    )

    # Correct mean
    mu = mu_pred + K @ innovation

    # Correct covariance
    I = np.eye(3)
    sigma = (I - K @ H) @ sigma_pred

    return mu, sigma, log_pz

def associate_landmark(
    mu_pred: np.ndarray,
    sigma_pred: np.ndarray,
    z: np.ndarray,
    landmarks: dict,
    Q: np.ndarray
) -> int:
    """Choose the most likely landmark for an observation.

    Args:
        mu_pred (np.ndarray): Predicted robot pose [x, y, theta].
        sigma_pred (np.ndarray): Predicted pose covariance.
        z (np.ndarray): Observed [range, bearing, signature].
        landmarks (dict): Landmark IDs mapped to [x, y, signature] arrays.
        Q (np.ndarray): Measurement noise covariance.

    Returns:
        int: ID of the landmark with the highest measurement likelihood.

    Raises:
        ValueError: If no valid landmark association is found.
    """

    x, y, theta = mu_pred

    best_id = None
    best_log_likelihood = -np.inf

    for landmark_id, landmark in landmarks.items():
        mx, my, signature = landmark

        dx = mx - x
        dy = my - y
        q = dx**2 + dy**2

        if q <= 1e-12:
            continue

        # Expected observation for candidate landmark
        z_expected = np.array([
            np.sqrt(q),
            wrap_angle(np.arctan2(dy, dx) - theta),
            signature
        ])

        # Measurement Jacobian
        H = np.array([
            [-dx / np.sqrt(q), -dy / np.sqrt(q), 0.0],
            [dy / q,          -dx / q,         -1.0],
            [0.0,              0.0,             0.0]
        ])

        # Innovation covariance
        S = H @ sigma_pred @ H.T + Q

        # Innovation
        innovation = z - z_expected
        innovation[1] = wrap_angle(innovation[1])

        # Gaussian log-likelihood
        sign, logdet = np.linalg.slogdet(S)

        if sign <= 0:
            continue

        mahalanobis_sq = innovation @ np.linalg.solve(
            S, innovation
        )

        log_likelihood = -0.5 * (
            mahalanobis_sq + logdet + len(z) * np.log(2 * np.pi)
        )

        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_id = landmark_id

    if best_id is None:
        raise ValueError("No valid landmark association found")

    return best_id

def gaussian_log_likelihood(
    innovation: np.ndarray,
    S: np.ndarray
) -> float:
    """Calculate the Gaussian log-likelihood of a measurement innovation.

    Args:
        innovation (np.ndarray): Difference between observed and expected measurements.
        S (np.ndarray): Innovation covariance matrix.

    Returns:
        float: Log-likelihood of the innovation.

    Raises:
        ValueError: If the innovation covariance has a nonpositive determinant.
    """

    n = len(innovation)

    sign, logdet = np.linalg.slogdet(S)

    if sign <= 0:
        raise ValueError("Innovation covariance must be positive definite")

    mahalanobis_sq = innovation @ np.linalg.solve(
        S, innovation
    )

    log_likelihood = -0.5 * (
        mahalanobis_sq
        + logdet
        + n * np.log(2 * np.pi)
    )

    return float(log_likelihood)


def wrap_angle(angle: float) -> float:
    """Wrap an angle to the range [-pi, pi).

    Args:
        angle (float): Angle in radians.

    Returns:
        float: Equivalent angle in radians between -pi (included) and pi (excluded).
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi

