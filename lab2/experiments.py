import numpy as np
import matplotlib.pyplot as plt
from sensor_models import beam_range_finder_model, p_hit, likelihood_field_range_finder_model
from ray_cast import expected_scan, measured_scan
from geometry import point_to_segment_distance

cats = [
    {"center": np.array([-1.0, -2.0]), "radius": 0.25},
    {"center": np.array([2.5, 0.0]), "radius": 0.25},
]

def test_beam_model():
    """Compare likelihoods of two controlled sensor scans."""

    params = [0.7, 0.1, 0.1, 0.1, 0.2, 1.0]
    max_range = 5.0

    z_expected = np.array([3.0, 4.0, 2.5, 3.5])

    # Measurements close to expected distances
    z_good = np.array([3.05, 3.9, 2.6, 3.45])

    # Measurements with larger errors
    z_bad = np.array([1.0, 1.5, 0.5, 2.0])

    q_good = beam_range_finder_model(
        z_expected, z_good, max_range, params
    )

    q_bad = beam_range_finder_model(z_expected, z_bad, max_range, params)

    print(f"Good scan likelihood: {q_good:.8f}")
    print(f"Bad scan likelihood:  {q_bad:.8f}")

    assert np.isfinite(q_good), "Good likelihood is NaN or infinity"
    assert np.isfinite(q_bad), "Bad likelihood is NaN or infinity"

    assert q_good >= 0, "Likelihood cannot be negative"
    assert q_bad >= 0, "Likelihood cannot be negative"

    assert q_good > q_bad, "Good scan should be more likely"
    print("Beam model test passed!")

def test_p_hit():
    z_expected = 3.0
    sigma_hit = 0.2
    max_range = 5.0

    z_expected_single = 3.0

    z_values = np.linspace(0, max_range, 500)

    densities = [
        p_hit(z, z_expected_single, sigma_hit, max_range)
        for z in z_values
    ]

    plt.figure(figsize=(8, 4))
    plt.plot(z_values, densities, label="p_hit")

    plt.axvline(
        z_expected_single,
        color="red",
        linestyle="--",
        label="Expected range"
    )

    plt.legend()
    plt.show()

def experiment_beam_likelihood(sim, params, rng):
    """Evaluate scan likelihood at different candidate positions."""

    true_pose = sim.robot.get_pose().copy()

    z_expected = expected_scan(sim)

    z_measured = z_expected + rng.normal(
        0, 0.1, size=len(z_expected)
    )
    z_measured = np.clip(
        z_measured, 0, sim.lidar_max_range
    )

    offsets = np.linspace(-0.5, 0.5, 101)
    likelihoods = []

    for offset in offsets:
        candidate_pose = true_pose.copy()
        candidate_pose[0] += offset

        z_candidate = expected_scan(sim, pose=candidate_pose)

        likelihood = beam_range_finder_model(
            z_candidate,
            z_measured,
            sim.lidar_max_range,
            params
        )

        likelihoods.append(likelihood)

    # Convert to NumPy array for verification
    likelihoods = np.asarray(likelihoods)

    # Check number of candidate positions
    assert len(offsets) == 101
    assert len(likelihoods) == len(offsets)

    # Check that likelihoods are valid
    assert np.all(np.isfinite(likelihoods))
    assert np.all(likelihoods >= 0)

    # Check that we haven't changed the actual robot pose
    assert np.allclose(sim.robot.get_pose(), true_pose)
    best_index = np.argmax(likelihoods)
    best_offset = offsets[best_index]

    print(f"Highest likelihood at offset: {best_offset:.3f} m")
    print(f"Maximum likelihood: {likelihoods[best_index]:.6g}")

    print("Candidate-pose experiment tests passed!")

    return offsets, likelihoods

def test_point_to_segment_distance():
    wall_start = np.array([3.0, -2.0])
    wall_end = np.array([3.0, 2.0])

    assert np.isclose(
        point_to_segment_distance([2, 0], wall_start, wall_end),
        1.0
    )

    assert np.isclose(
        point_to_segment_distance([3, 0], wall_start, wall_end),
        0.0
    )

    assert np.isclose(
        point_to_segment_distance([3, 3], wall_start, wall_end),
        1.0
    )

    print("Point-to-segment distance tests passed!")

def test_likelihood_field():
    world_map = {
        "lines": [
            {
                "p1": np.array([3.0, -2.0]),
                "p2": np.array([3.0, 2.0])
            }
        ],
        "circles": []
    }

    pose = np.array([0.0, 0.0, 0.0])
    sensor_angles = np.array([0.0])

    q_good = likelihood_field_range_finder_model(
        np.array([3.0]), pose, world_map,
        sensor_angles, 5.0
    )

    q_bad = likelihood_field_range_finder_model(
        np.array([1.0]), pose, world_map,
        sensor_angles, 5.0
    )

    print(f"Good endpoint likelihood: {q_good:.6f}")
    print(f"Bad endpoint likelihood:  {q_bad:.6f}")

    assert q_good > q_bad
    assert np.isfinite(q_good)
    assert np.isfinite(q_bad)

    print("Likelihood field test passed!")

def update_cats(cats, t):
    """Move cats along simple periodic trajectories."""

    cats[0]["center"] = np.array([
        -1.0 + 0.6 * np.sin(0.8 * t),
        -2.0 + 0.4 * np.cos(0.8 * t)
    ])

    cats[1]["center"] = np.array([
        2.5 + 0.5 * np.sin(0.5 * t),
        0.5 * np.cos(0.5 * t)
    ])

if __name__ == '__main__':
    test_beam_model()
    test_p_hit()

def collect_training_data(sim, cats, rng, num_scans=200):
    """Collect matching expected and measured range data."""

    all_expected = []
    all_measured = []

    for i in range(num_scans):
        # Move cats through their trajectories
        t = i * sim.dt
        update_cats(cats, t)

        # Static-map prediction
        z_expected = expected_scan(sim)

        # Sensor observation including cats
        z_measured = measured_scan(sim, cats, rng)

        all_expected.extend(z_expected)
        all_measured.extend(z_measured)

    return (
        np.array(all_measured),
        np.array(all_expected)
    )

from sensor_models import learn_intrinsic_parameters


def experiment_parameter_learning(sim, rng):

    cats = [
        {"center": np.array([-1.0, -2.0]), "radius": 0.25},
        {"center": np.array([2.5, 0.0]), "radius": 0.25},
    ]

    z_measured, z_expected = collect_training_data(
        sim, cats, rng, num_scans=200
    )

    initial_params = [
        0.7, 0.1, 0.1, 0.1, 0.2, 1.0
    ]

    learned_params, history = learn_intrinsic_parameters(
        z_measured,
        z_expected,
        sim.lidar_max_range,
        initial_params
    )

    names = [
        "z_hit", "z_short", "z_max", "z_rand",
        "sigma_hit", "lambda_short"
    ]

    for name, initial, learned in zip(
        names, initial_params, learned_params
    ):
        print(f"{name:15} {initial:.4f} -> {learned:.4f}")

    print(f"EM iterations: {len(history) - 1}")

    return history


def run_landmark_demo():
    from sensor_models import (
        landmark_model_known_correspondence,
        sample_landmark_model_known_correspondence
    )

    rng = np.random.default_rng(42)

    # [x, y, signature]
    landmarks = {
        0: np.array([2.0, 2.0, 1.0])
    }

    true_pose = np.array([0.0, 0.0, 0.0])

    # Simulated observation of landmark 0
    feature = np.array([
        np.sqrt(8),
        np.pi / 4,
        1.0
    ])

    sigma_r = 0.1
    sigma_phi = 0.05
    sigma_s = 0.1

    likelihood = landmark_model_known_correspondence(
        feature, 0, true_pose, landmarks,
        sigma_r, sigma_phi, sigma_s
    )

    samples = np.array([
        sample_landmark_model_known_correspondence(
            feature, 0, landmarks,
            sigma_r, sigma_phi, rng
        )
        for _ in range(500)
    ])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        samples[:, 0], samples[:, 1],
        s=8, alpha=0.4, label="Sampled robot poses"
    )
    ax.scatter(2, 2, marker="*", s=180, label="Landmark")
    ax.scatter(0, 0, marker="x", s=100, label="True robot pose")

    ax.set_aspect("equal")
    ax.set_title(f"Landmark model — likelihood: {likelihood:.3g}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    plt.show()