import numpy as np
import matplotlib.pyplot as plt

# Stronger, smoother mottling
np.random.seed(0)
mu = 30
size = 256

# Pure Poisson
pure = np.random.poisson(mu, size=(size, size))

# Create smooth modulation field
raw = np.random.normal(size=(size, size))
# simple smoothing via convolution
kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
mod = raw.copy()
for _ in range(4):  # apply smoothing multiple times
    mod = np.convolve(mod.flatten(), kernel.flatten(), mode="same").reshape(size, size)

# Normalize and amplify modulation
mod = (mod - mod.min()) / (mod.max() - mod.min())
mod = 1 + 0.8 * (mod - 0.5)  # stronger contrast

mottled = np.random.poisson(mu * np.abs(mod))

# Save images
plt.imshow(mottled, cmap="gray")
plt.axis("off")
plt.savefig("plots/mottled_stronger.png", bbox_inches="tight", dpi=300)
plt.close()

plt.imshow(pure, cmap="gray")
plt.axis("off")
plt.savefig("plots/pure.png", bbox_inches="tight", dpi=300)
plt.close()
