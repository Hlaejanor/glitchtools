# Re-run the previous code after kernel reset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import numpy as np
import matplotlib.animation as animation

# Parameters for simulation
num_lanesheets = 10  # number of contributing sources
wavelengths = np.linspace(
    0.2, 1.0, num_lanesheets
)  # simulated wavelengths (arbitrary units)
grid_size = 200  # pixels per side
frame_count = 100  # number of animation frames
lane_spacing_factor = 30  # base spacing multiplier


# Generate composite lanesheets
def generate_lanesheet(center, spacing, size):
    """Generate a 2D lanesheet with circular ridges."""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    rings = np.cos(2 * np.pi * R / spacing)
    rings = (rings + 1) / 2  # normalize to [0, 1]
    return rings


# Create static lanesheets
np.random.seed(0)
centers = np.random.uniform(-0.5, 0.5, size=(num_lanesheets, 2))
spacings = lane_spacing_factor / wavelengths
lanesheets = [
    generate_lanesheet(centers[i], spacings[i], grid_size)
    for i in range(num_lanesheets)
]

# Composite lanesheet: sum all contributions
composite = np.sum(lanesheets, axis=0)

# Simulate detector drift across composite
drift_speed = 0.01  # per frame
window_size = 50  # sliding window size
window = np.zeros((window_size, window_size))

# Animation function
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(window, cmap="viridis", vmin=0, vmax=num_lanesheets, origin="lower")
ax.set_title("Simulated Drift Across Composite Lanesheet")


def update(frame):
    start_x = int((frame * drift_speed * grid_size) % (grid_size - window_size))
    start_y = int((frame * drift_speed * grid_size * 1.1) % (grid_size - window_size))
    window = composite[start_y : start_y + window_size, start_x : start_x + window_size]
    im.set_data(window)
    return [im]


ani = FuncAnimation(fig, update, frames=frame_count, interval=100, blit=True)
# Explicitly use FFMpegWriter and save to disk
writer = animation.FFMpegWriter(fps=10, bitrate=1800)
ani.save("animation_output.mp4", writer=writer)
plt.close(fig)
