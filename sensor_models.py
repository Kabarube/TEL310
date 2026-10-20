
"""
TEL310 - Probabilistic Robotics
Lab 2: Sensor Models

Ray casting and beam-based range finder model.
Reference: Thrun et al., Chapter 6, Table 6.1.
"""

from dataclasses import dataclass
from math import erf

import numpy as np



@dataclass
class BeamModelParams:
    """Intrinsic parameters of the beam-based sensor model."""
    z_hit: float = 0.70
    z_short: float = 0.15
    z_max: float = 0.10
    z_rand: float = 0.05

    sigma_hit: float = 0.10
    lambda_short: float = 1.0



def normal_cdf(x):
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def p_hit(z, z_expected, z_max, sigma):
    """
    Truncated Gaussian model for accurate measurements.

    z: observed distance
    z_expected: distance predicted by ray casting
    """
    if not 0 <= z <= z_max:
        return 0.0

    upper = normal_cdf((z_max - z_expected) / sigma)
    lower = normal_cdf(-z_expected / sigma)
    normalization = upper - lower

    if normalization <= 0:
        return 0.0

    error = (z - z_expected) / sigma

    return (
        np.exp(-0.5 * error**2)
        / (sigma * np.sqrt(2 * np.pi) * normalization)
    )


def p_short(z, z_expected, lambda_short):
    """Exponential distribution for unexpected short measurements."""
    if z_expected <= 0 or not 0 <= z <= z_expected:
        return 0.0

    normalization = -np.expm1(-lambda_short * z_expected)

    return (
        lambda_short * np.exp(-lambda_short * z)
        / normalization
    )


def p_max(z, z_max):
    """Probability mass at the maximum sensor range."""
    return 1.0 if np.isclose(z, z_max, atol=1e-9) else 0.0


def p_rand(z, z_max):
    """Uniform distribution for random measurements."""
    if 0 <= z < z_max:
        return 1.0 / z_max

    return 0.0


def beam_probability(z, z_expected, z_max, params):
    """Weighted mixture of the four measurement models."""

    return (
        params.z_hit
        * p_hit(z, z_expected, z_max, params.sigma_hit)

        + params.z_short
        * p_short(z, z_expected, params.lambda_short)

        + params.z_max
        * p_max(z, z_max)

        + params.z_rand
        * p_rand(z, z_max)
    )


def beam_range_finder_model(
    z_observed,
    z_expected,
    z_max,
    params,
):
    """
    Calculate the log-likelihood of an entire LiDAR scan.

    z_observed: array of measured distances
    z_expected: array of ray-cast predicted distances
    z_max: maximum sensor range
    params: BeamModelParams instance

    Returns:
        log_likelihood: log p(z_t | x_t, m)
        beam_likelihoods: likelihood for each beam
    """

    probabilities = np.array([
        beam_probability(z, z_star, z_max, params)
        for z, z_star in zip(z_observed, z_expected)
    ])

    if np.any(probabilities <= 0):
        log_likelihood = np.isinf()
    else:
        log_likelihood = float(np.sum(np.log(probabilities)))

    return log_likelihood, probabilities


def learn_intrinsic_parameters(
    z_observed,
    z_expected,
    z_max,
    initial_params=None,
    max_iterations=200,
    tolerance=1e-6,
):
    """
    Learn beam-model parameters using Expectation-Maximization.

    Reference:
        Thrun, Burgard & Fox, Table 6.2.

    Parameters
    ----------
    z_observed : ndarray
        Collected LiDAR measurements.
    z_expected : ndarray
        Expected distances from ray casting.
    z_max : float
        Maximum LiDAR range.
    initial_params : BeamModelParams
        Initial parameter estimates.
    max_iterations : int
        Maximum number of EM iterations.
    tolerance : float
        Convergence tolerance on parameter changes.

    Returns
    -------
    params : BeamModelParams
        Learned parameters.
    history : list
        Parameter estimates at each iteration.
    """

    z_observed = np.asarray(z_observed, dtype=float).ravel()
    z_expected = np.asarray(z_expected, dtype=float).ravel()


    if initial_params is None:
        params = BeamModelParams()
    else:
        params = BeamModelParams(**vars(initial_params))

    history = []

    def parameter_vector(p):
        return np.array([
            p.z_hit, p.z_short, p.z_max, p.z_rand,
            p.sigma_hit, p.lambda_short,
        ])

    for iteration in range(max_iterations):

        # ------------------------------------------
        # E-step: calculate responsibilities
        # ------------------------------------------

        responsibilities = np.zeros(
            (len(z_observed), 4),
            dtype=float,
        )

        for i, (z, z_star) in enumerate(
            zip(z_observed, z_expected)
        ):

            components = np.array([
                p_hit(z, z_star, z_max, params.sigma_hit),
                p_short(z, z_star, params.lambda_short),
                p_max(z, z_max),
                p_rand(z, z_max),
            ])

            weights = np.array([
                params.z_hit,
                params.z_short,
                params.z_max,
                params.z_rand,
            ])

            weighted_components = weights * components
            total = np.sum(weighted_components)

            if total <= 0:
                raise ValueError(
                    "An observation has zero likelihood."
                )

            responsibilities[i] = weighted_components / total

        # ------------------------------------------
        # M-step: update parameters
        # ------------------------------------------

        old_parameters = parameter_vector(params)

        new_weights = responsibilities.mean(axis=0)

        hit_resp = responsibilities[:, 0]
        short_resp = responsibilities[:, 1]

        hit_sum = np.sum(hit_resp)
        short_sum = np.sum(short_resp)

        sigma_hit = params.sigma_hit
        lambda_short = params.lambda_short

        if hit_sum > 1e-12:
            squared_errors = (z_observed - z_expected)**2

            sigma_hit = np.sqrt(
                np.sum(hit_resp * squared_errors) / hit_sum
            )

        if short_sum > 1e-12:
            weighted_distances = np.sum(
                short_resp * z_observed
            )

            if weighted_distances > 1e-12:
                lambda_short = short_sum / weighted_distances

        # Keep noise parameters strictly positive.
        sigma_hit = max(float(sigma_hit), 1e-5)
        lambda_short = max(float(lambda_short), 1e-5)

        params = BeamModelParams(
            z_hit=float(new_weights[0]),
            z_short=float(new_weights[1]),
            z_max=float(new_weights[2]),
            z_rand=float(new_weights[3]),
            sigma_hit=sigma_hit,
            lambda_short=lambda_short,
        )

        history.append(parameter_vector(params).copy())

        # ------------------------------------------
        # Convergence
        # ------------------------------------------

        change = np.max(
            np.abs(parameter_vector(params) - old_parameters)
        )

        if change < tolerance:
            print(f"EM converged after {iteration + 1} iterations.")
            break

    return params, np.array(history)

