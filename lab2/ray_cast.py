from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from SET_Prob import Simulator

# Ray casting
def expected_scan(
    sim: Simulator,
    num_beams: int = 8,
    pose: Sequence[float] | NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """
    Calculate ideal ranges to static obstacles using ray casting.

    Parameters
    ----------
    sim : Simulator
        Simulator providing the world map and maximum sensor range.
    num_beams : int, optional
        Number of beams evenly spaced over a full rotation. Default is 8.
    pose : array_like of shape (3,), optional
        Robot pose [x, y, theta], with theta in radians. Defaults to the
        simulator's current robot pose.

    Returns
    -------
    np.ndarray of shape (num_beams,)
        Distance to the nearest wall or circle along each beam, capped at
        sim.lidar_max_range. Beams start at the robot's heading and proceed
        counterclockwise. Beams with no obstacle return the maximum range.
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

        direction = np.array([np.cos(ray_angle), np.sin(ray_angle)])

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

def measured_scan(
    sim: Simulator,
    cats: Sequence[dict],
    rng: np.random.Generator,
    noise_std: float = 0.05,
) -> NDArray[np.float64]:
    """
    Simulate eight range measurements with dynamic obstacles and noise.

    Parameters
    ----------
    sim : Simulator
        Simulator providing the world map, current robot pose, and maximum
        sensor range.
    cats : iterable of dict
        Circular dynamic obstacles, each with a "center" containing its
        world coordinates and a "radius".
    rng : np.random.Generator
        Random number generator used to sample sensor noise.
    noise_std : float, optional
        Standard deviation of the zero-mean Gaussian range noise.
        Default is 0.05.

    Returns
    -------
    np.ndarray of shape (8,)
        Ranges to the nearest static or dynamic obstacle after adding noise,
        clipped to [0, sim.lidar_max_range]. Beams are evenly spaced over a
        full rotation, starting at the robot's heading.
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

def point_to_segment_distance(
    point: ArrayLike, start: ArrayLike, end: ArrayLike,
) -> float:
    """
    Calculate the shortest Euclidean distance from a point to a line segment.

    Parameters
    ----------
    point : array_like of shape (2,)
        Coordinates of the point.
    start : array_like of shape (2,)
        Coordinates of the segment's starting point.
    end : array_like of shape (2,)
        Coordinates of the segment's ending point.

    Returns
    -------
    float
        Distance to the closest point on the segment. If start and end
        coincide, returns the distance to that endpoint.
    """

    point = np.asarray(point, dtype=float)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    segment = end - start
    point_vector = point - start

    length_squared = np.dot(segment, segment)

    if length_squared == 0:
        return np.linalg.norm(point_vector)

    # Projection onto the infinite line
    t = np.dot(point_vector, segment) / length_squared

    # Restrict the projection to the actual segment
    t = np.clip(t, 0.0, 1.0)

    closest_point = start + t * segment

    return np.linalg.norm(point - closest_point)

def nearest_obstacle_distance(
    point: ArrayLike, world_map: dict[str, list[dict]],
) -> float:
    """
    Calculate the distance from a point to the nearest mapped wall.

    Parameters
    ----------
    point : array_like of shape (2,)
        World coordinates of the point.
    world_map : dict
        Map whose "lines" entry contains wall segments, each with endpoints
        "p1" and "p2".

    Returns
    -------
    float
        Shortest Euclidean distance to any wall segment, or np.inf if there
        are no walls.

    Notes
    -----
    Circular obstacles are not included in this calculation.
    """

    min_distance = np.inf

    for wall in world_map["lines"]:
        distance = point_to_segment_distance(point, wall["p1"], wall["p2"])

        min_distance = min(min_distance, distance)

    return min_distance

def beam_endpoint(
    pose: Sequence[float] | NDArray[np.float64],
    distance: float,
    sensor_angle: float,
) -> NDArray[np.float64]:
    """
    Calculate a sensor beam's endpoint in world coordinates.

    Parameters
    ----------
    pose : array_like of shape (3,)
        Robot pose [x, y, theta], with theta in radians.
    distance : float
        Range measured along the beam.
    sensor_angle : float
        Beam angle relative to the robot's heading, in radians.

    Returns
    -------
    np.ndarray of shape (2,)
        World coordinates [x, y] of the beam endpoint.
    """
    x, y, theta = pose

    angle = theta + sensor_angle

    return np.array([
        x + distance * np.cos(angle),
        y + distance * np.sin(angle)
    ])
