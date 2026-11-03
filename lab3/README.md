# TEL310 — Labwork 3

Python simulations of mobile robot localization using Extended and Unscented
Kalman Filters, landmark observations, and known or unknown landmark correspondence.

## Run

Requires Python, NumPy, and OpenCV. From `lab3`, select a localization algorithm
using the command-line argument:

```bash
python main.py --algorithm ekf-known    # EKF with known landmark correspondence
python main.py --algorithm ekf-unknown  # EKF with unknown landmark correspondence
python main.py --algorithm ukf          # Unscented Kalman Filter
python main.py --help                   # Show available options
```

Use `--sensor-range` to set the detection range in meters and `--fov` to set the
field of view in degrees. Press `q` to quit the simulation.

## AI disclaimer

AI was used partly for implementation, to help understand the theory behind
the implementations, write docstrings and type hints.
