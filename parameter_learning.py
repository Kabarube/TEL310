
"""
TEL310 - Probabilistic Robotics
Lab 2: Learning Intrinsic Sensor Parameters

Synthetic-data validation of the EM algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
from sensor_models import BeamModelParams, learn_intrinsic_parameters

# ---------------------------------------------
# Generate synthetic sensor measurements
# ---------------------------------------------

def generate_measurements(
    n_samples,
    z_max,
    true_params,
    rng,
):
    """
    Generate measurements from the four-component beam model.

    Returns observed and expected distance arrays.
    """

    expected = rng.uniform(
        0.8, z_max - 0.3, size=n_samples
    )

    observed = np.empty(n_samples)

    weights = [
        true_params.z_hit,
        true_params.z_short,
        true_params.z_max,
        true_params.z_rand,
    ]

    components = rng.choice(
        4,
        size=n_samples,
        p=weights,
    )

    for i, component in enumerate(components):
        z_star = expected[i]

        if component == 0:
            # Hit: truncated Gaussian
            while True:
                z = rng.normal(
                    z_star, true_params.sigma_hit
                )
                if 0 <= z <= z_max:
                    break

        elif component == 1:
            # Short: truncated exponential
            lam = true_params.lambda_short
            u = rng.uniform()

            z = -np.log(
                1 - u * (1 - np.exp(-lam * z_star))
            ) / lam

        elif component == 2:
            # Max-range measurement
            z = z_max

        else:
            # Random measurement
            z = rng.uniform(0, z_max)

        observed[i] = z

    return observed, expected, components


# ---------------------------------------------
# Run parameter learning
# ---------------------------------------------

def main():

    rng = np.random.default_rng(42)

    z_max = 5.0
    n_samples = 5000

    # Ground-truth parameters used to generate data
    true_params = BeamModelParams(
        z_hit=0.65,
        z_short=0.20,
        z_max=0.10,
        z_rand=0.05,
        sigma_hit=0.12,
        lambda_short=1.5,
    )

    observed, expected, components = generate_measurements(
        n_samples, z_max, true_params, rng
    )

    # Starting guesses are deliberately different.
    initial_params = BeamModelParams(
        z_hit=0.55,
        z_short=0.15,
        z_max=0.15,
        z_rand=0.15,
        sigma_hit=0.25,
        lambda_short=0.8,
    )

    estimated, history = learn_intrinsic_parameters(
        observed,
        expected,
        z_max,
        initial_params=initial_params,
        max_iterations=200,
        tolerance=1e-6,
    )

    # ---------------------------------------------
    # Print results
    # ---------------------------------------------

    print("\nParameter estimation results")
    print("-" * 52)
    print(f"{'Parameter':<18}{'True':>14}{'Estimated':>14}")

    names = [
        "z_hit", "z_short", "z_max", "z_rand",
        "sigma_hit", "lambda_short",
    ]

    for name in names:
        print(
            f"{name:<18}"
            f"{getattr(true_params, name):>14.4f}"
            f"{getattr(estimated, name):>14.4f}"
        )

    # ---------------------------------------------
    # Plot parameter convergence
    # ---------------------------------------------

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    for i, (ax, name) in enumerate(zip(axes.flat, names)):
        values = history[:, i]

        ax.plot(values, label="Estimated")
        ax.axhline(
            getattr(true_params, name),
            color="red",
            linestyle="--",
            label="True",
        )

        ax.set_title(name)
        ax.set_xlabel("EM iteration")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("Learning Beam Model Parameters with EM")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
