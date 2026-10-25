import numpy as np

def point_to_segment_distance(point, start, end):
    """Shortest Euclidean distance from a point to a line segment."""

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

def nearest_obstacle_distance(point, world_map):
    """Distance from a point to the closest static obstacle."""

    min_distance = np.inf

    for wall in world_map["lines"]:
        distance = point_to_segment_distance(point, wall["p1"], wall["p2"])

        min_distance = min(min_distance, distance)

    return min_distance

def beam_endpoint(pose, distance, sensor_angle):
    x, y, theta = pose

    angle = theta + sensor_angle

    return np.array([
        x + distance * np.cos(angle),
        y + distance * np.sin(angle)
    ])