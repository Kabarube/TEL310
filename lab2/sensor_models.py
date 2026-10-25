# =============================================================================
# Author:       Kai Ruben Hamlander Enerhaugen
# Email:        kai.r.h.enerhaugen@nmbu.no
# Created:      2026-10-23
# Description:  Sensor models for probabilistic robotics (TEL310 NMBU). 
# =============================================================================
import numpy as np
from scipy.stats import truncnorm, norm
from geometry import beam_endpoint, nearest_obstacle_distance


##########################################
# Sensor models for probabilistic robotics
##########################################


def p_hit(z, z_expected, sigma_hit, max_range):
    """
    Probability density of an ordinary range measurement.

    Uses a Gaussian distribution truncated to [0, max_range].
    """

    if not 0 <= z <= max_range:
        return 0.0

    a = (0 - z_expected) / sigma_hit
    b = (max_range - z_expected) / sigma_hit

    return truncnorm.pdf(
        z,
        a=a,
        b=b,
        loc=z_expected,
        scale=sigma_hit
    )

def p_short(z, z_expected, lambda_short):
    if not 0 <= z <= z_expected:
        return 0.0

    eta = 1 / (1 - np.exp(-lambda_short * z_expected))

    return eta * lambda_short * np.exp(-lambda_short * z)

def p_max(z, max_range):
    return 1.0 if z == max_range else 0.0

def p_rand(z, max_range):
    if 0 <= z < max_range:
        return 1 / max_range
    return 0.0

def likelihood_field_range_finder_model(
    z_measured,
    pose,
    world_map,
    sensor_angles,
    max_range,
    z_hit=0.9,
    z_rand=0.1,
    sigma_hit=0.2
):
    """Likelihood of a range scan using distances to mapped obstacles."""
    q = 1.0

    for z, angle in zip(z_measured, sensor_angles):

        # Skip maximum-range measurements (Table 6.3)
        if z >= max_range:
            continue

        # Transform measured range into world coordinates
        endpoint = beam_endpoint(pose, z, angle)

        # Find distance to nearest mapped obstacle
        distance = nearest_obstacle_distance(endpoint, world_map)

        # Gaussian likelihood of endpoint distance
        p_hit = (
            1 / (np.sqrt(2 * np.pi) * sigma_hit)
            * np.exp(-0.5 * (distance / sigma_hit) ** 2)
        )

        # Mixture of hit and random measurements
        p = z_hit * p_hit + z_rand / max_range

        q *= p

    return q
    
def beam_range_finder_model(
        z_expected,
        z_measured,
        max_range,
        params
):
    """
    Evaluate the likelihood of an entire range scan.

    Parameters
    ----------
    z_expected : np.ndarray
        Expected ranges calculated using ray casting.

    z_measured : np.ndarray
        Actual simulated sensor measurements.

    max_range : float
        Maximum sensor range.

    params : list
        [z_hit, z_short, z_max, z_rand, sigma_hit, lambda_short]

    Returns
    -------
    float
        Likelihood of the complete scan.
    """

    z_hit, z_short, z_max, z_rand, sigma_hit, lambda_short = params

    if len(z_expected) != len(z_measured):
        raise ValueError("Expected and measured scans must have equal length")

    q = 1.0

    for expected, measured in zip(z_expected, z_measured):

        # Individual probability components
        hit = p_hit(measured, expected, sigma_hit, max_range)
        short = p_short(measured, expected, lambda_short)
        max_return = p_max(measured, max_range)
        random = p_rand(measured, max_range)

        # Weighted mixture
        p = (z_hit * hit + z_short * short + z_max * max_return + z_rand * random)

        q *= p

    return q


def beam_responsibilities(z, z_expected, max_range, params):
    z_hit, z_short, z_max, z_rand, sigma_hit, lambda_short = params

    components = np.array([
        p_hit(z, z_expected, sigma_hit, max_range),
        p_short(z, z_expected, lambda_short),
        p_max(z, max_range),
        p_rand(z, max_range)
    ])

    weights = np.array([z_hit, z_short, z_max, z_rand])

    weighted_components = weights * components
    total = np.sum(weighted_components)

    if total <= 0:
        raise ValueError("Measurement has zero mixture likelihood")

    return weighted_components / total


def learn_intrinsic_parameters(
    z_measured,
    z_expected,
    max_range,
    initial_params,
    max_iterations=50,
    tolerance=1e-5
):
    """
    Learn the six intrinsic beam-model parameters using EM.

    z_measured and z_expected are matching 1D arrays.
    Each element represents one individual beam measurement.
    """

    z_measured = np.asarray(z_measured, dtype=float)
    z_expected = np.asarray(z_expected, dtype=float)
    params = np.asarray(initial_params, dtype=float).copy()

    if z_measured.shape != z_expected.shape:
        raise ValueError("Measurement arrays must have equal shapes")

    if z_measured.ndim != 1 or z_measured.size == 0:
        raise ValueError("Training data must be a nonempty 1D array")

    history = [params.copy()]
    eps = 1e-10

    for iteration in range(max_iterations):

        # E-step: calculate responsibilities
        responsibilities = np.array([
            beam_responsibilities(z, z_star, max_range, params)
            for z, z_star in zip(z_measured, z_expected)
        ])

        # M-step: update mixture weights
        weights = np.mean(responsibilities, axis=0)

        # M-step: update sigma_hit
        e_hit = responsibilities[:, 0]
        squared_errors = (z_measured - z_expected) ** 2

        sigma_hit = np.sqrt(
            np.sum(e_hit * squared_errors)
            / max(np.sum(e_hit), eps)
        )

        # M-step: update lambda_short
        e_short = responsibilities[:, 1]

        lambda_short = (
            np.sum(e_short)
            / max(np.sum(e_short * z_measured), eps)
        )

        # Avoid degenerate distribution parameters
        sigma_hit = max(sigma_hit, 1e-3)
        lambda_short = max(lambda_short, 1e-3)

        new_params = np.array([
            *weights,
            sigma_hit,
            lambda_short
        ])

        history.append(new_params.copy())

        # Check convergence
        if np.max(np.abs(new_params - params)) < tolerance:
            params = new_params
            break

        params = new_params

    return params, np.array(history)

def wrap_angle(angle):
    """Normalize an angle to [-pi, pi)."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def landmark_model_known_correspondence(
    feature, correspondence, pose, landmarks,
    sigma_r, sigma_phi, sigma_s
):
    # Known landmark identity
    landmark = landmarks[correspondence]

    lx, ly, signature = landmark
    x, y, theta = pose
    r, phi, s = feature

    # Expected landmark observation
    dx = lx - x
    dy = ly - y

    r_expected = np.hypot(dx, dy)
    phi_expected = wrap_angle(np.arctan2(dy, dx) - theta)

    # Measurement errors
    error_r = r - r_expected
    error_phi = wrap_angle(phi - phi_expected)
    error_s = s - signature

    # Independent Gaussian measurement errors
    q = (
        norm.pdf(error_r, loc=0, scale=sigma_r)
        * norm.pdf(error_phi, loc=0, scale=sigma_phi)
        * norm.pdf(error_s, loc=0, scale=sigma_s)
    )

    return q


def sample_landmark_model_known_correspondence(
    feature, correspondence, landmarks,
    sigma_r, sigma_phi, rng
):
    lx, ly, _ = landmarks[correspondence]
    r, phi, _ = feature

    gamma = rng.uniform(0, 2 * np.pi)

    sampled_r = r + rng.normal(0, sigma_r)
    sampled_phi = phi + rng.normal(0, sigma_phi)

    x = lx + sampled_r * np.cos(gamma)
    y = ly + sampled_r * np.sin(gamma)
    theta = wrap_angle(gamma - np.pi - sampled_phi)

    return np.array([x, y, theta])