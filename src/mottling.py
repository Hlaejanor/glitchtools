import numpy as np
import matplotlib.pyplot as plt


def generate_fields(mu=0.2, size=256, seed=0):
    np.random.seed(seed)
    pure = np.random.poisson(mu, size=(size, size))
    modulation = 1 + 0.2 * np.random.normal(size=(size, size))
    mottled = np.random.poisson(mu * np.abs(modulation))
    return pure, mottled


def main():
    pure, mottled = generate_fields()
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(pure, cmap="gray")
    plt.title("Pure Poisson")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mottled, cmap="gray")
    plt.title("Mottled (Excess Variance)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("plots/mottling_demo.png", dpi=300)


if __name__ == "__main__":
    main()
