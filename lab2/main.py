import numpy as np
from SET_Prob import Simulator
from experiments import experiment_beam_likelihood, test_beam_model, test_point_to_segment_distance, test_likelihood_field, run_landmark_demo
from visualization import plot_beam_likelihood, run_sensor_simulation, plot_parameter_learning



def create_simulator():

    # Map from assignment
    walls = [
        # Outer boundary
        ((-5.0, -5.0), ( 5.0, -5.0)),
        (( 5.0, -5.0), ( 5.0,  5.0)),
        (( 5.0,  5.0), (-5.0,  5.0)),
        ((-5.0,  5.0), (-5.0, -5.0)),

        # Starting area
        ((-5.0, -1.0), (-3.0, -1.0)),
        ((-1.0, -1.0), (0.0, -1.0)),
        ((0.0, -1.0), (0.0, -5.0)),
        ((-5.0, -4.0), (-4.0, -4.0)),
        ((-4.0, -4.0), (-4.0, -5.0)),

        # Upper right area
        ((0.0, 5.0), (2.0, 1.0)),  # Diagonal
        ((3.0, 5.0), (3.0, 3.0)),  # Vertical
        ((4.0, 1.0), (5.0, 1.0)),  # Horizontas

        # Upper cross
        ((-3.5, 2.5), (-1.5, 2.5)),
        ((-2.5, 1.5), (-2.5, 3.5)),

        # Lower-right crossed walls
        ((1.5, -1.5), (3.5, -3.5)),
        ((1.5, -3.5), (3.5, -1.5)),
    ]

    # Initialize parameters and simulation
    world_map = {'lines': [], 'circles': []}

    for p1, p2 in walls:
        world_map['lines'].append({
            'p1': np.array(p1, dtype=float),
            'p2': np.array(p2, dtype=float),
            'color': np.array([0, 0, 0])
        })

    # world_map["circles"].append({
    #     "center": np.array([2.0, 0.0]),
    #     "radius": 0.5,
    #     "color": np.array([255, 0, 0])
    # })

    robot_params = {
        "initial_pose": np.array([-2.0, -3.0, np.pi / 2]),
        "a_max": 0.5,
        "alpha_max": 0.5,
        "dt": 0.1,
        "radius": 0.15
    }

    camera_params = {
        "width": 800,
        "height": 800,
        "meters_per_pixel": 0.015
    }

    # Init simulator
    sim = Simulator(
        robot_init_pose=robot_params,
        cam_params=camera_params,
        time_step=0.1,
        world_map=world_map
    )

    sim.lidar_max_range = 8.0

    return sim


def main():
    run_landmark_demo()
    test_likelihood_field()

    # Verify geometry
    test_point_to_segment_distance()

    # Test beam model
    test_beam_model()

    # Create simulation environment
    sim = create_simulator()
    params = [0.7, 0.1, 0.1, 0.1, 0.05, 1.0]
    rng = np.random.default_rng(42)

    offsets, likelihoods = experiment_beam_likelihood(
        sim, params, rng
    )

    plot_beam_likelihood(offsets, likelihoods)
    run_sensor_simulation(sim, params, rng)


if __name__ == '__main__':
    main()