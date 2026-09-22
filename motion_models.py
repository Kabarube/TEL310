import math
import random



def motion_model_velocity(x_t, x_prev, u_t, alpha):
    """
    Compute the likelihood of a motion from x_prev to x_t under a velocity motion model.
    """
    x, y, theta = x_t
    xp, yp, thetap = x_prev
    v_hat, omega_hat, dt = u_t
    a1, a2, a3, a4, a5, a6 = alpha

    if dt <= 0:
        raise ValueError("dt must be positive")

    numerator = (x - xp) * math.cos(theta) + (y - yp) * math.sin(theta)
    denominator = (y - yp) * math.cos(theta) - (x - xp) * math.sin(theta)

    if abs(denominator) < 1e-12:
        mu = 0.0
    else:
        mu = 0.5 * numerator / denominator

    x_star = (x + xp) / 2.0 + mu * (y - yp)
    y_star = (y + yp) / 2.0 + mu * (xp - x)
    r_star = math.hypot(x - x_star, y - y_star)

    dtheta = math.atan2(yp - y_star, xp - x_star) - math.atan2(y - y_star, x - x_star)
    dtheta = (dtheta + math.pi) % (2.0 * math.pi) - math.pi

    omega = dtheta / dt
    v = omega * r_star
    gamma = (thetap - theta) / dt - omega

    def prob(error, variance):
        if variance <= 0:
            return 0.0
        return (1.0 / math.sqrt(2.0 * math.pi * variance)) * math.exp(
            -(error ** 2) / (2.0 * variance)
        )

    var1 = a1 * v**2 + a2 * omega**2
    var2 = a3 * v**2 + a4 * omega**2
    var3 = a5 * v**2 + a6 * omega**2

    p1 = prob(v - v_hat, var1)
    p2 = prob(omega - omega_hat, var2)
    p3 = prob(gamma, var3)

    return p1 * p2 * p3


def prob_triangular_distribution(a, b):
    """
    Evaluate the triangular distribution from the pseudocode.
    """
    if b <= 0:
        raise ValueError("b must be positive")

    support = math.sqrt(6.0) * b
    if abs(a) > support:
        return 0.0

    return 1.0 / (math.sqrt(6.0) * b) - abs(a) / (6.0 * b**2)


def prob_normal_distribution(a, b):
    """
    Evaluate the Gaussian density with mean 0 and standard deviation b.
    """
    if b <= 0:
        raise ValueError("b must be positive")

    return (1.0 / (math.sqrt(2.0 * math.pi) * b)) * math.exp(-(a**2) / (2.0 * b**2))


def sample_normal_distribution(b):
    """
    Sample from N(0, b^2).
    """
    if b <= 0:
        raise ValueError("b must be positive")
    return random.gauss(0.0, b)


def sample_triangular_distribution(b):
    """
    Sample from the triangular distribution used in the pseudocode.
    This follows the assignment's simplified form:
        return (sqrt(6) / 2) * rand(-b, b)
    """
    if b <= 0:
        raise ValueError("b must be positive")
    return (math.sqrt(6.0) / 2.0) * random.uniform(-b, b)


def sample_motion_model_velocity(x_prev, u_t, alpha):
    """
    Sample a new pose from the velocity motion model.
    """
    x, y, theta = x_prev
    v, omega, dt = u_t
    a1, a2, a3, a4, a5, a6 = alpha

    if dt <= 0:
        raise ValueError("dt must be positive")

    v_hat = v + random.gauss(0.0, math.sqrt(a1 * v**2 + a2 * omega**2))
    omega_hat = omega + random.gauss(0.0, math.sqrt(a3 * v**2 + a4 * omega**2))
    gamma = random.gauss(0.0, math.sqrt(a5 * v**2 + a6 * omega**2))

    if abs(omega_hat) < 1e-12:
        theta_hat = theta + omega_hat * dt
        x_prime = x + v_hat * dt * math.cos(theta)
        y_prime = y + v_hat * dt * math.sin(theta)
        theta_prime = theta_hat + gamma * dt
        return (x_prime, y_prime, theta_prime)

    r_hat = v_hat / omega_hat
    theta_hat = theta + omega_hat * dt
    x_prime = x - r_hat * (math.sin(theta) - math.sin(theta_hat))
    y_prime = y + r_hat * (math.cos(theta) - math.cos(theta_hat))
    theta_prime = theta_hat + gamma * dt
    return (x_prime, y_prime, theta_prime)


def motion_model_odometry(x_t, x_prev, alpha):
    """
    Evaluate the odometry motion model likelihood.

    x_t : tuple[float, float, float]
        New pose (x', y', theta')
    x_prev : tuple[float, float, float]
        Previous pose (x, y, theta)
    alpha : tuple[float, float, float, float, float, float]
        Noise parameters.
    """
    x, y, theta = x_prev
    xp, yp, thetap = x_t
    a1, a2, a3, a4, a5, a6 = alpha

    delta_rot1 = math.atan2(yp - y, xp - x) - theta
    delta_trans = math.hypot(xp - x, yp - y)
    delta_rot2 = thetap - theta - delta_rot1

    delta_rot1_hat = delta_rot1
    delta_trans_hat = delta_trans
    delta_rot2_hat = delta_rot2

    p1 = prob_normal_distribution(delta_rot1 - delta_rot1_hat, math.sqrt(a1 * delta_rot1_hat**2 + a2 * delta_trans_hat**2))
    p2 = prob_normal_distribution(delta_trans - delta_trans_hat, math.sqrt(a3 * delta_trans_hat**2 + a4 * (delta_rot1_hat**2 + delta_rot2_hat**2)))
    p3 = prob_normal_distribution(delta_rot2 - delta_rot2_hat, math.sqrt(a5 * delta_trans_hat**2 + a6 * (delta_rot1_hat**2 + delta_rot2_hat**2)))

    return p1 * p2 * p3


def sample_motion_model_odometry(x_prev, u_t, alpha):
    """
    Sample a new pose using the odometry motion model.
    """
    x, y, theta = x_prev
    v, omega, dt = u_t
    a1, a2, a3, a4, a5, a6 = alpha

    if dt <= 0:
        raise ValueError("dt must be positive")

    x_prime = x + v * dt * math.cos(theta)
    y_prime = y + v * dt * math.sin(theta)
    theta_prime = theta + omega * dt
    x_t = (x_prime, y_prime, theta_prime)

    delta_rot1 = math.atan2(y_prime - y, x_prime - x) - theta
    delta_trans = math.hypot(x_prime - x, y_prime - y)
    delta_rot2 = theta_prime - theta - delta_rot1

    delta_rot1_hat = delta_rot1 - random.gauss(0.0, math.sqrt(a1 * delta_rot1**2 + a2 * delta_trans**2))
    delta_trans_hat = delta_trans - random.gauss(0.0, math.sqrt(a3 * delta_trans**2 + a4 * (delta_rot1**2 + delta_rot2**2)))
    delta_rot2_hat = delta_rot2 - random.gauss(0.0, math.sqrt(a5 * delta_trans**2 + a6 * (delta_rot1**2 + delta_rot2**2)))

    x_sample = x + delta_trans_hat * math.cos(theta + delta_rot1_hat)
    y_sample = y + delta_trans_hat * math.sin(theta + delta_rot1_hat)
    theta_sample = theta + delta_rot1_hat + delta_rot2_hat

    return (x_sample, y_sample, theta_sample)


if __name__ == "__main__":
    random.seed(0)

    x_t = (2.0, 1.0, 0.3)
    x_prev = (1.0, 0.0, 0.1)
    u_t = (0.5, 0.2, 0.1)
    alpha = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1)

    p = motion_model_velocity(x_t, x_prev, u_t, alpha)
    print(f"Probability: {p}")

    sampled = sample_motion_model_velocity((1.0, 0.0, 0.0), (0.6, 0.2, 0.1), alpha)
    print(f"Sampled: {sampled}")

    print(f"Triangular density at 0.0: {prob_triangular_distribution(0.0, 1.0)}")
    print(f"Normal density at 0.0: {prob_normal_distribution(0.0, 1.0)}")
    print(f"Sample normal: {sample_normal_distribution(1.0)}")
    print(f"Sample triangular: {sample_triangular_distribution(1.0)}")
