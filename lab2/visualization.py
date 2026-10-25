import matplotlib.pyplot as plt
import numpy as np
import cv2

from ray_cast import expected_scan, measured_scan
from sensor_models import beam_range_finder_model, likelihood_field_range_finder_model
from experiments import update_cats


# Draw the sensor scan on the simulator image
def draw_scan(image, sim, ranges, color=(255, 0, 0)):
    """Draw the eight range measurements on the simulator image."""

    image = image.copy()
    pose = sim.robot.get_pose()
    x, y, theta = pose

    angles = np.linspace(0, 2*np.pi, len(ranges), endpoint=False)

    # Convert robot's world position into image pixels
    start = sim.camera.project_point([x, y])

    for angle, distance in zip(angles, ranges):
        global_angle = theta + angle

        # Calculate beam endpoint in world coordinates
        end_x = x + distance * np.cos(global_angle)
        end_y = y + distance * np.sin(global_angle)

        # Convert endpoint to pixel coordinates
        end = sim.camera.project_point([end_x, end_y])

        # Draw beam and intersection point
        cv2.line(image, start, end, color, 2)
        cv2.circle(image, end, 4, color, -1)

    return image

def plot_beam_likelihood(offsets, likelihoods):

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(offsets, likelihoods)
    ax.axvline(
        0,
        color="red",
        linestyle="--",
        label="True position"
    )

    ax.set_xlabel("Candidate x-offset [m]")
    ax.set_ylabel("Scan likelihood")
    ax.set_title("Beam Range Finder Model")
    ax.legend()
    ax.grid(True)

    plt.show()


def run_sensor_simulation(sim, params, rng):

    v, omega = 0.0, 0.0
    sensor_angles = np.linspace(0, 2*np.pi, 8, endpoint=False)

    cats = [
        {"center": np.array([-1.0, -2.0]), "radius": 0.25},
        {"center": np.array([2.5, 0.0]), "radius": 0.25},
    ]

    t = 0.0
    while True:

        image = sim.step(v, omega)

        z_expected = expected_scan(sim)

        update_cats(cats, t)
        t += sim.dt

        z_measured = measured_scan(sim, cats, rng)
                
        q_beam = beam_range_finder_model(
            z_expected,
            z_measured,
            sim.lidar_max_range,
            params
        )

        q_field = likelihood_field_range_finder_model(
            z_measured,
            sim.robot.get_pose(),
            sim.world_map,
            sensor_angles,
            sim.lidar_max_range,
            z_hit=params[0],
            z_rand=params[3],
            sigma_hit=params[4]
        )

        for cat in cats:
            center_px = sim.camera.project_point(cat["center"])
            radius_px = int(cat["radius"] / sim.camera.meters_per_pixel)

            cv2.circle(
                image,
                center_px,
                radius_px,
                (255, 165, 0),
                -1
            )


        # 6. Draw expected and measured scans
        image = draw_scan(
            image, sim, z_expected, color=(0, 160, 0)
        )

        image = draw_scan(
            image, sim, z_measured, color=(255, 0, 0)
        )

        # 7. Display the likelihoods
        cv2.putText(
            image, f"Beam likelihood: {q_beam:.3g}",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 0), 2
        )

        cv2.putText(
            image, f"Field likelihood: {q_field:.3g}",
            (15, 60), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 0), 2
        )


        cv2.imshow(
            "Lab 2 - Sensor Models",
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        )

        # 8. Keyboard controls
        key = cv2.waitKey(30) & 0xFF

        if key == ord("w"):
            v += 0.1
        elif key == ord("s"):
            v -= 0.1
        elif key == ord("a"):
            omega += 0.1
        elif key == ord("d"):
            omega -= 0.1
        elif key == ord("h"):
            v, omega = 0.0, 0.0
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()



def plot_parameter_learning(history):
    names = [
        "z_hit", "z_short", "z_max",
        "z_rand", "sigma_hit", "lambda_short"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))

    for i, ax in enumerate(axes.flat):
        ax.plot(history[:, i], marker=".")
        ax.set_title(names[i])
        ax.set_xlabel("EM iteration")
        ax.grid(True)

    plt.tight_layout()
    plt.show()