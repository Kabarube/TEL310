import numpy as np

# Ray casting
def expected_scan(sim, num_beams=8, pose=None):
    """
    Calculate ideal range measurements using the known map.

    Parameters
    ----------
    sim : Simulator
        The professor-provided simulator.
    num_beams : int
        Number of equally spaced sensor beams.

    Returns
    -------
    np.ndarray
        Expected range measurement for each beam.
    """

    if pose is None:
        pose = sim.robot.get_pose()

    origin = pose[:2]
    theta = pose[2]

    angles = np.linspace(0, 2*np.pi, num_beams, endpoint=False)

    ranges = []

    # Loop through each beam and calculate the expected range
    for angle in angles:
        # Convert relative sensor angle to global direction
        ray_angle = theta + angle

        direction = np.array([
            np.cos(ray_angle),
            np.sin(ray_angle)
        ])

        # Calculate the predicted measurement beam k
        closest_distance = sim.lidar_max_range
        ray_end = origin + direction * sim.lidar_max_range

        # Check for intersections with all walls in the map
        for wall in sim.world_map['lines']:
            intersection = sim._get_line_line_intersection(
                origin,
                ray_end,
                wall['p1'],
                wall['p2']
            )

            if intersection is not None:
                distance = np.linalg.norm(intersection - origin)
                if distance < closest_distance:
                    closest_distance = distance

        # Check for intersections with all circles in the map
        for circle in sim.world_map['circles']:
            intersection = sim._get_line_circle_intersection(
                origin,
                direction,
                circle['center'],
                circle['radius'],
                sim.lidar_max_range
            )

            if intersection is not None:
                distance = np.linalg.norm(intersection - origin)

                if distance < closest_distance:
                    closest_distance = distance

        ranges.append(closest_distance)

    return np.array(ranges)


def measured_scan(sim, cats, rng, noise_std=0.05):
    """
    Simulate sensor measurements including dynamic obstacles.
    """

    z_measured = expected_scan(sim)

    pose = sim.robot.get_pose()
    origin = pose[:2]
    theta = pose[2]

    angles = np.linspace(0, 2*np.pi, len(z_measured), endpoint=False)

    for k, angle in enumerate(angles):
        global_angle = theta + angle

        direction = np.array([
            np.cos(global_angle),
            np.sin(global_angle)
        ])

        for cat in cats:
            intersection = sim._get_line_circle_intersection(
                origin,
                direction,
                cat["center"],
                cat["radius"],
                sim.lidar_max_range
            )

            if intersection is not None:
                distance = np.linalg.norm(intersection - origin)

                z_measured[k] = min(z_measured[k], distance)

    # Add ordinary sensor noise
    z_measured += rng.normal(0, noise_std, size=len(z_measured))

    return np.clip(z_measured, 0, sim.lidar_max_range)