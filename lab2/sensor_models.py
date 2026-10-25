# =============================================================================
# Author:       Kai Ruben Hamlander Enerhaugen
# Email:        kai.r.h.enerhaugen@nmbu.no
# Created:      2026-10-23
# Description:  Sensor models for probabilistic robotics (TEL310 NMBU). 
# =============================================================================
from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import truncnorm, norm
from ray_cast import beam_endpoint, nearest_obstacle_distance


##########################################
# Sensor models for probabilistic robotics
##########################################


def p_hit(z: float, z_expected: float, sigma_hit: float, max_range: float) -> float:
    """
    Evaluate the hit component's probability density for a range measurement.

    Uses a Gaussian centered at z_expected and truncated to [0, max_range].

    Parameters
    ----------
    z : float
        Measured range.
    z_expected : float
        Expected range to the nearest mapped obstacle along the beam.
    sigma_hit : float
        Positive standard deviation of the Gaussian before truncation.
    max_range : float
        Maximum sensor range.

    Returns
    -------
    float
        Probability density at z, or 0.0 if z lies outside [0, max_range].
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

def p_short(z: float, z_expected: float, lambda_short: float) -> float:
    """
    Evaluate the short-return component's probability density.

    Uses an exponential distribution normalized over [0, z_expected] to
    model unexpectedly short ranges caused by unmapped obstacles.

    Parameters
    ----------
    z : float
        Measured range.
    z_expected : float
        Positive expected range to the nearest mapped obstacle.
    lambda_short : float
        Positive rate parameter of the exponential distribution.

    Returns
    -------
    float
        Probability density at z, or 0.0 if z lies outside [0, z_expected].
    """
    if not 0 <= z <= z_expected:
        return 0.0

    eta = 1 / (1 - np.exp(-lambda_short * z_expected))

    return eta * lambda_short * np.exp(-lambda_short * z)

def p_max(z: float, max_range: float) -> float:
    """
    Evaluate the maximum-range return indicator.

    Parameters
    ----------
    z : float
        Measured range.
    max_range : float
        Maximum sensor range.

    Returns
    -------
    float
        1.0 if z equals max_range exactly, otherwise 0.0.
    """
    return 1.0 if z == max_range else 0.0

def p_rand(z: float, max_range: float) -> float:
    """
    Evaluate the uniform random-return component's probability density.

    Parameters
    ----------
    z : float
        Measured range.
    max_range : float
        Positive maximum sensor range.

    Returns
    -------
    float
        1 / max_range for 0 <= z < max_range, otherwise 0.0.
    """
    if 0 <= z < max_range:
        return 1 / max_range
    return 0.0

def likelihood_field_range_finder_model(
    z_measured: Sequence[float] | NDArray[np.float64],
    pose: Sequence[float] | NDArray[np.float64],
    world_map: dict[str, list[dict]],
    sensor_angles: Sequence[float] | NDArray[np.float64],
    max_range: float,
    z_hit: float = 0.9,
    z_rand: float = 0.1,
    sigma_hit: float = 0.2,
) -> float:
    """
    Evaluate a scan using beam endpoints' distances to mapped walls.

    Parameters
    ----------
    z_measured : array_like of shape (n,)
        Measured ranges. Values at or above max_range are skipped.
    pose : array_like of shape (3,)
        Robot pose [x, y, theta], with theta in radians.
    world_map : dict
        Map containing wall segments in its "lines" entry.
    sensor_angles : array_like of shape (n,)
        Beam angles relative to the robot's heading, in radians.
    max_range : float
        Positive maximum sensor range.
    z_hit : float, optional
        Weight of the Gaussian endpoint-distance component. Default is 0.9.
    z_rand : float, optional
        Weight of the uniform random-return component. Default is 0.1.
    sigma_hit : float, optional
        Positive Gaussian standard deviation. Default is 0.2.

    Returns
    -------
    float
        Product of the per-beam mixture values, or 1.0 if no beams are used.

    Notes
    -----
    Circular obstacles are ignored. Ranges and angles are paired with zip,
    so any unmatched trailing entries are ignored.
    """
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
        z_expected: Sequence[float] | NDArray[np.float64],
        z_measured: Sequence[float] | NDArray[np.float64],
        max_range: float,
        params: Sequence[float] | NDArray[np.float64],
) -> float:
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
        Product of the per-beam mixture values, or 1.0 for empty scans.

    Raises
    ------
    ValueError
        If the expected and measured scans have different lengths.
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


def beam_responsibilities(
    z: float,
    z_expected: float,
    max_range: float,
    params: Sequence[float] | NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculate the posterior mixture responsibilities for one beam.

    Parameters
    ----------
    z : float
        Measured range.
    z_expected : float
        Expected range from ray casting.
    max_range : float
        Maximum sensor range.
    params : array_like of shape (6,)
        [z_hit, z_short, z_max, z_rand, sigma_hit, lambda_short].

    Returns
    -------
    np.ndarray of shape (4,)
        Normalized weighted components in hit, short, max, random order.

    Raises
    ------
    ValueError
        If the total weighted mixture value is nonpositive.
    """
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
    z_measured: ArrayLike,
    z_expected: ArrayLike,
    max_range: float,
    initial_params: ArrayLike,
    max_iterations: int = 50,
    tolerance: float = 1e-5,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Learn the six intrinsic beam-model parameters using EM.

    Parameters
    ----------
    z_measured : array_like of shape (n,)
        Nonempty training data containing individual measured beam ranges.
    z_expected : array_like of shape (n,)
        Matching expected beam ranges from ray casting.
    max_range : float
        Maximum sensor range.
    initial_params : array_like of shape (6,)
        Initial [z_hit, z_short, z_max, z_rand, sigma_hit, lambda_short].
        Copied before updating.
    max_iterations : int, optional
        Maximum number of EM updates. Default is 50.
    tolerance : float, optional
        Stop when the largest absolute parameter change is below this
        threshold. Default is 1e-5.

    Returns
    -------
    params : np.ndarray of shape (6,)
        Final parameter estimates in the same order as initial_params.
    history : np.ndarray of shape (iterations + 1, 6)
        Initial parameters followed by the estimates after each update.

    Raises
    ------
    ValueError
        If training arrays have different shapes, are not nonempty 1D
        arrays, or a beam has nonpositive mixture likelihood.

    Notes
    -----
    Updated sigma_hit and lambda_short are bounded below by 1e-3.
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

def wrap_angle(
    angle: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """
    Normalize angles to [-pi, pi).

    Parameters
    ----------
    angle : float or np.ndarray
        Angle or angles in radians.

    Returns
    -------
    float or np.ndarray
        Wrapped angles in radians, preserving the input shape.
    """
    return (angle + np.pi) % (2 * np.pi) - np.pi


def landmark_model_known_correspondence(
    feature: Sequence[float] | NDArray[np.float64],
    correspondence: Hashable,
    pose: Sequence[float] | NDArray[np.float64],
    landmarks: Mapping[Hashable, Sequence[float] | NDArray[np.float64]]
    | Sequence[Sequence[float] | NDArray[np.float64]],
    sigma_r: float,
    sigma_phi: float,
    sigma_s: float,
) -> float:
    """
    Evaluate an observation of a known landmark at a candidate robot pose.

    Parameters
    ----------
    feature : array_like of shape (3,)
        Observed [range, bearing, signature], with bearing in radians.
    correspondence : hashable or int
        Key or index identifying the observed landmark.
    pose : array_like of shape (3,)
        Robot pose [x, y, theta], with theta in radians.
    landmarks : mapping or sequence
        Landmark entries [x, y, signature], indexed by correspondence.
    sigma_r : float
        Positive standard deviation of range error.
    sigma_phi : float
        Positive standard deviation of bearing error, in radians.
    sigma_s : float
        Positive standard deviation of signature error.

    Returns
    -------
    float
        Product of independent Gaussian densities for range, wrapped
        bearing, and signature errors.
    """
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
    feature: Sequence[float] | NDArray[np.float64],
    correspondence: Hashable,
    landmarks: Mapping[Hashable, Sequence[float] | NDArray[np.float64]]
    | Sequence[Sequence[float] | NDArray[np.float64]],
    sigma_r: float,
    sigma_phi: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """
    Sample a robot pose from an observation of a known landmark.

    Parameters
    ----------
    feature : array_like of shape (3,)
        Observed [range, bearing, signature], with bearing in radians.
        The signature is unused.
    correspondence : hashable or int
        Key or index identifying the observed landmark.
    landmarks : mapping or sequence
        Landmark entries [x, y, signature], indexed by correspondence.
    sigma_r : float
        Standard deviation of the Gaussian range perturbation.
    sigma_phi : float
        Standard deviation of the Gaussian bearing perturbation, in radians.
    rng : np.random.Generator
        Random number generator for the direction and measurement noise.

    Returns
    -------
    np.ndarray of shape (3,)
        Sampled pose [x, y, theta], with theta in [-pi, pi).

    Notes
    -----
    Position direction around the landmark is sampled uniformly. Gaussian
    range perturbations are not clipped to nonnegative values.
    """
    lx, ly, _ = landmarks[correspondence]
    r, phi, _ = feature

    gamma = rng.uniform(0, 2 * np.pi)

    sampled_r = r + rng.normal(0, sigma_r)
    sampled_phi = phi + rng.normal(0, sigma_phi)

    x = lx + sampled_r * np.cos(gamma)
    y = ly + sampled_r * np.sin(gamma)
    theta = wrap_angle(gamma - np.pi - sampled_phi)

    return np.array([x, y, theta])
