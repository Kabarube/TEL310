# TEL310 — Labwork 2

Python simulations of robot perception: probabilistic range sensor models,
intrinsic parameter learning, landmark observations, and ray casting in a
mapped environment with moving cats.

## Run

Requires Python, NumPy, SciPy, Matplotlib, and OpenCV. From `lab2`, select a
demo using the command-line argument:

```bash
python main.py sensor    # Interactive sensor simulation
python main.py learning  # Learn sensor parameters using EM
python main.py landmark  # Landmark observation and pose sampling
python main.py --help    # Show options and keyboard controls
```

## AI disclaimer

AI was used partly for implementation, to help understand the theory behind
the implementations, write docstrings and typehints.
