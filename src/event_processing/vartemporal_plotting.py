import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.sandbox.stats.runs import runstest_1samp

from common.fitsmetadata import (
    FitsMetadata,
    ChunkVariabilityMetadata,
    ProcessingParameters,
    ComparePipeline,
)
from pandas import DataFrame


class Arrow:
    tbin_0: int
    wbin_0: int
    tbin_1: int
    wbin_1: int
    tbin_widtch_index: int
    label = "test"

    def __init__(self, wbin_0, tbin_0, wbin_1, tbin_1, tbin_widtch_index):
        self.tbin_0 = wbin_0
        self.wbin_0 = tbin_0
        self.tbin_1 = wbin_1
        self.wbin_1 = tbin_1
        self.tbin_widtch_index = tbin_widtch_index


def ensure_folder_exists(filepath):
    folder = os.path.dirname(filepath)
    folder_exists = os.path.isdir(folder)

    if not folder_exists:
        print("Creating folder for")
        os.mkdir(folder)


# def acf(X, Y, Z, title, x_label, y_label, z_label, filename, show):
# plot_acf(data_for_one_wavelength["Counts"])


def plot_image(
    image, title, x_label, y_label, z_label, filename, arrows: list[Arrow], show
):
    # Create a 3D surface plot
    """
    fig_3d = plt.figure(figsize=(12, 7))
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    surf = ax_3d.plot_surface(, Y, Z, cmap=cm.viridis, edgecolor="none")
    ax_3d.set_title(title)
    ax_3d.set_xlabel(x_label)
    ax_3d.set_ylabel(y_label)
    ax_3d.set_zlabel(z_label)
    fig_3d.colorbar(surf, shrink=0.5, aspect=10)
    """
    # Create a heatmap
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 7))
    c = ax_heatmap.imshow(
        image[:, :],
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    y_min, y_max = image[:, 1].min(), image[:, 1].max()
    # Add annotations (vertical markers/arrows) on the heatmap at given x positions
    if arrows is not None:
        # Normalize annotations into list of (x, label) pairs (label optional)
        if len(arrows) > 0:
            for arrow in arrows:
                # Draw a vertical dashed line for clarity

                ax_heatmap.annotate(
                    "",
                    xytext=(arrow.tbin_0, arrow.wbin_0),
                    xy=(arrow.tbin_1, arrow.wbin_1),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="white",
                        lw=0.8,
                        shrinkA=0,
                        shrinkB=0,
                        alpha=0.8,
                    ),
                    ha="center",
                    va="center",
                    clip_on=False,
                )

                """
                ax_heatmap.annotate(
                    "",
                    xy=(arrow.tbin_0, arrow.wbin_0),
                    xytext=(arrow.tbin_0 + 10, arrow.wbin_0 + 10),
                    arrowprops=dict(
                        arrowstyle="-|>", color="red", lw=1.5, shrinkA=0, shrinkB=0
                    ),
                    ha="center",
                    va="center",
                    clip_on=False,
                )
              

                # Optional small label above the arrow if provided
                if arrow.label:
                    ax_heatmap.text(
                        arrow.tbin_0,
                        arrow.wbin_0,
                        str(arrow.label),
                        color="red",
                        ha="center",
                        va="bottom",
                        clip_on=False,
                    )
                  """
    ax_heatmap.set_xscale("linear")
    ax_heatmap.set_yscale("linear")
    ax_heatmap.set_title(title)
    ax_heatmap.set_xlabel(x_label)
    ax_heatmap.set_ylabel(y_label)
    fig_heatmap.colorbar(c, ax=ax_heatmap, label=z_label)

    plt.tight_layout()

    filepath = f"./plots/{filename}"
    ensure_folder_exists(filepath)

    plt.savefig(filepath, dpi=300)
    if show:
        plt.show()


def vartemporal_plot(
    X, Y, Z, title, x_label, y_label, z_label, filename, annotations, show
):
    # Create a 3D surface plot

    fig_3d = plt.figure(figsize=(12, 7))
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    surf = ax_3d.plot_surface(X, Y, Z, cmap=cm.viridis, edgecolor="none")
    ax_3d.set_title(title)
    ax_3d.set_xlabel(x_label)
    ax_3d.set_ylabel(y_label)
    ax_3d.set_zlabel(z_label)
    fig_3d.colorbar(surf, shrink=0.5, aspect=10)

    # Create a heatmap
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 7))
    c = ax_heatmap.imshow(
        Z,
        aspect="auto",
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        origin="lower",
        cmap="viridis",
    )
    # Add annotations (vertical markers/arrows) on the heatmap at given x positions
    if annotations is not None:
        # Normalize annotations into list of (x, label) pairs (label optional)
        ann_list = []
        for a in annotations:
            if a is None:
                continue
            if isinstance(a, (list, tuple)) and len(a) >= 1:
                x = a[0]
                label = a[1] if len(a) > 1 else None
            else:
                x = a
                label = None
            try:
                x = float(x)
            except Exception:
                continue
            ann_list.append((x, label))

        if len(ann_list) > 0:
            y_min, y_max = Y.min(), Y.max()
            y_range = y_max - y_min if (y_max - y_min) != 0 else 1.0

            for x_val, label in ann_list:
                # skip annotations outside the visible extent
                if x_val < X.min() or x_val > X.max():
                    continue

                # Draw a vertical dashed line for clarity
                ax_heatmap.axvline(
                    x=x_val, color="red", linestyle="--", linewidth=1.0, alpha=0.8
                )

                # Draw a short arrow above the heatmap pointing down
                arrow_top = y_max + 0.06 * y_range
                arrow_tip = y_max - 0.01 * y_range
                ax_heatmap.annotate(
                    "",
                    xy=(x_val, arrow_tip),
                    xytext=(x_val, arrow_top),
                    arrowprops=dict(
                        arrowstyle="-|>", color="red", lw=1.5, shrinkA=0, shrinkB=0
                    ),
                    ha="center",
                    va="center",
                    clip_on=False,
                )

                # Optional small label above the arrow if provided
                if label:
                    ax_heatmap.text(
                        x_val,
                        arrow_top + 0.02 * y_range,
                        str(label),
                        color="red",
                        ha="center",
                        va="bottom",
                        clip_on=False,
                    )
    ax_heatmap.set_title("Excess Variability Heatmap")
    ax_heatmap.set_xlabel(x_label)
    ax_heatmap.set_ylabel(y_label)
    fig_heatmap.colorbar(c, ax=ax_heatmap, label=z_label)

    plt.tight_layout()

    filepath = f"./plots/{filename}"
    ensure_folder_exists(filepath)
    plt.savefig(filepath, dpi=300)
    if show:
        plt.show()


def harmonic_band_plot(
    pipe: ComparePipeline,
    meta: FitsMetadata,
    binned_data: DataFrame,
    time_bin_widht_index: int,
    wbin_widths: [],
    time_bin_widhts: [],
    bands: [],
):
    time_bin_column = f"Time Bin {time_bin_widht_index}"
    tbins = np.sort(binned_data[time_bin_column].dropna().unique().astype(int))
    wbins = binned_data["Wavelength Bin"].unique()

    time_bin_width = time_bin_widhts[time_bin_widht_index]

    time_bin_times = time_bin_width * tbins

    # 2. Count photons in each (Time Bin, Wavelength Bin)
    counts_per_wbin_tbin_width_combo = (
        binned_data[binned_data["Hit"] == 1]
        .groupby([time_bin_column, "Wavelength Bin"])
        .size()
        .reset_index(name="Counts")
    )

    Z_grid = counts_per_wbin_tbin_width_combo.pivot(
        index="Wavelength Bin", columns=time_bin_column, values="Counts"
    ).values

    Y_vals = wbin_widths[wbins]
    X_vals = time_bin_times

    X_grid, Y_grid = np.meshgrid(X_vals, Y_vals)
    title = f"Harmonic bands in {meta.star}"
    if meta.synthetic:
        title += " (synthetic)"

    vartemporal_plot(
        X_grid,
        Y_grid,
        Z_grid,
        title,
        "Time",
        "Wavelength",
        "SDMC",
        f"{pipe.id}/vartemporal_{meta.id}_{np.round(time_bin_width)}s.png",
        annotations=bands,
        show=False,
    )


if __name__ == "__main__":
    # Simulated data (placeholder)
    # Replace this with actual data loading step
    np.random.seed(42)
    time_bins = np.linspace(0, 1000, 100)  # 100 time bins
    wavelengths = np.linspace(0.1, 1.0, 50)  # 50 wavelength bins

    # Simulate some structured excess variability data with possible moiré-like oscillation
    X, Y = np.meshgrid(time_bins, wavelengths)
    Z = np.sin(2 * np.pi * X / 200 + 5 * np.pi * Y) * np.cos(
        2 * np.pi * Y / 0.1
    ) + np.random.normal(0, 0.1, X.shape)
    vartemporal_plot(
        X,
        Y,
        Z,
        title="Demo Vartemporal",
        x_label="Time",
        y_label="Wavelength",
        z_label="SDMC",
        filename="demo_vartemporal.png",
        annotations=None,
        show=False,
    )
