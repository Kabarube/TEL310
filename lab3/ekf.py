import numpy as np
#from lab1.motion_models import prob_normal_distribution
#from SET_Prob import Simulator

def EKF_localisation_known_correspondences(mu:np.ndarray, cov, u, z, c, map):
    x_prev, y_prev, theta_prev = None
    pass



def prediction_step(
        mu: np.ndarray,
        u: np.ndarray,
        sigma: np.ndarray,
        alpha: np.ndarray,
        dt: float
    ) -> np.ndarray:
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


# # Test prediction step
# mu = np.array([0.0, 0.0, 0.0])

# Sigma = np.diag([0.01, 0.01, 0.01])

# u = np.array([1.0, 0.5])

# alpha = (0.01, 0.01, 0.02, 0.02)

# dt = 1.0

# print(prediction_step(
#     mu,
#     u,
#     Sigma,
#     alpha,
#     dt
# ))


def correction_step(
        mu_pred: np.ndarray,
        sigma_pred: np.ndarray,
        z: np.ndarray,
        landmark: np.ndarray,
        Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

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

    # Correct mean
    mu = mu_pred + K @ innovation

    # Correct covariance
    I = np.eye(3)
    sigma = (I - K @ H) @ sigma_pred

    return mu, sigma

def wrap_angle(angle):
    """Wrap an angle to the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi



#TEST CORRECTION
mu_pred = np.array([0.0, 0.0, 0.0])

sigma_pred = np.diag([
    0.2**2,
    0.2**2,
    np.deg2rad(10)**2
])

# Known landmark at (3, 2) with signature 1
landmark = np.array([3.0, 2.0, 1.0])

# Simulated measurement
z = np.array([
    3.5,
    0.65,
    1.0
])

Q = np.diag([
    0.1**2,
    0.05**2,
    0.1**2
])

mu, sigma = correction_step(
    mu_pred,
    sigma_pred,
    z,
    landmark,
    Q
)

print("Predicted pose:", mu_pred)
print("Corrected pose:", mu)
print("Predicted covariance:\n", sigma_pred)
print("Corrected covariance:\n", sigma, "\n")
print(np.diag(sigma_pred))
print(np.diag(sigma))