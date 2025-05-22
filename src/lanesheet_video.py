import argparse
import os
import pandas as pd
from pandas import DataFrame
from lanesheet_for_video.lanesheet_for_video import LanesheetForVideo
from common.helper import get_duration
from common.fitsmetadata import Spectrum, ProcessingParameters, GenerationParameters
from common.fitsread import (
    pi_channel_to_wavelength_and_width,
    read_crop_and_project_to_ccd,
)
from common.generate_data import (
    generate_synthetic_telescope_data,
    generate_synth_if_need_be,
)
from common.metadatahandler import load_gen_param, load_processing_param
from common.fitsmetadata import FitsMetadata
from common.fitsread import load_fits_metadata, fits_read

import numpy as np
import sys
import matplotlib.patches as patches
import imageio

import matplotlib.pyplot as plt
import matplotlib as mpl

# General typography setting
mpl.rcParams["font.family"] = "Serif"  # Or 'Baskerville', 'Palatino Linotype', etc.
mpl.rcParams["font.size"] = 14  # Slightly bigger font size
mpl.rcParams["axes.titlesize"] = 16
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.labelsize"] = 12
mpl.rcParams["ytick.labelsize"] = 12

# Optional: make lines slightly softer
mpl.rcParams["lines.linewidth"] = 2
mpl.rcParams["axes.linewidth"] = 1

# Optional: turn off "spines" for a cleaner typeset look
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False

# Optional: light grid
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.alpha"] = 0.3
mpl.rcParams["grid.linestyle"] = "--"


def plot_time_histogram(
    events, fps=24, duration=10, lambda_center=None, lambda_width=None
):
    """
    Plots a histogram of photon event times over video time (0 to duration).

    Parameters:
        events (DataFrame): Must contain column 'video_time'
        fps (int): Frames per second used in binning
        duration (float): Total duration in seconds for the visualization
    """
    assert "video_time" in events.columns, "Data must include 'video_time' column"
    bins = int(duration * fps)
    bin_edges = np.linspace(0, duration, bins + 1)
    if lambda_center is not None:
        lamb_max = lambda_center + lambda_width / 2
        lamb_min = lamb_max - lambda_width / 2

        wavelength_mask = np.logical_and(
            events["Wavelength (nm)"] > lamb_min,
            events["Wavelength (nm)"] < lamb_max,
        )
        events = events[wavelength_mask]

    plt.figure(figsize=(12, 4))
    plt.hist(events["video_time"], bins=bin_edges, color="skyblue", edgecolor="black")
    plt.xlabel("Video Time (s)")
    plt.ylabel("Photon Count")
    plt.title("Photon Event Histogram Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def prepare_data(
    meta: FitsMetadata,
    pparams: ProcessingParameters,
    genparams: GenerationParameters,
    take_first_seconds: int = None,
    duration: float = 10,
    fps: int = 24,
):
    assert genparams is not None, "Genparams must be a Gen param object"
    assert pparams is not None, "Processing Params must be set"
    # 1. Load preprocessed events
    success, meta, pp, events = read_crop_and_project_to_ccd(
        fits_id=meta.id, processing_param_id=pparams.id
    )

    assert len(events) > 0, "No events left after wavelength filtering"

    # 3. Determine available duration
    max_possible_duration = get_duration(meta, pp)
    end = events["relative_time"].max()

    # 4. Clamp duration if needed
    if take_first_seconds is not None and take_first_seconds < max_possible_duration:
        print(f"Reducing duration to {take_first_seconds}s")
        events = events[events["relative_time"] < take_first_seconds].copy()
        end = events["relative_time"].max()

    start = 0.0
    tpf = (end - start) / (duration * fps)

    # 5. Normalize to video time range
    events["video_time"] = (events["relative_time"] - start) / (end - start) * duration

    # 6. Bin into frame time bins
    bin_edges = np.arange(0, duration + 1e-6, 1 / fps)
    events["Time Bin"] = pd.cut(
        events["video_time"], bins=bin_edges, labels=False, include_lowest=True
    )
    events.sort_values("Time Bin", inplace=True)

    print(f"Prepared {len(events)} events over {duration}s video duration")
    return events


def make_video(
    meta,
    pparams: ProcessingParameters,
    genparams: GenerationParameters,
    events: DataFrame,
    duration: float = 10,
    fps: int = 24,
):
    assert genparams is not None, "Genparams must be a Gen param object"

    # 1. Prepare constants
    theta_deg = 180 * genparams.theta / np.pi
    velvec = np.array([np.cos(genparams.theta), np.sin(genparams.theta)])
    lambda_center = (pparams.max_wavelength + pparams.min_wavelength) / 2
    g = genparams.get_g(lambda_center)
    dy = (np.sqrt(3) / 2) * g

    tpf = duration / (duration * fps)  # time per frame in video-time
    bin_edges = np.arange(0, duration + 1e-6, 1 / fps)

    offset = genparams.phase * velvec
    perp_dir = np.array([-velvec[1], velvec[0]])
    offset += genparams.perp * perp_dir

    n_frames = int(duration * fps)
    histogram_bins = np.zeros(n_frames)
    sctr_x = []
    sctr_y = []

    output_path = f"videos/lanesheet_theta_{theta_deg:.2f}_{meta.id}.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig_width_inch, fig_height_inch = fig.get_size_inches()
    dpi = fig.dpi
    fig_width_px = fig_width_inch * dpi

    view_size = int(10 / g) * 2
    zoom = fig_width_px / view_size
    n_rows = int(view_size / (np.sqrt(3) / 2) + 1)
    n_cols = view_size
    frames = []

    # Precompute a dictionary: frame index → DataFrame slice
    events_by_frame = events.groupby("Time Bin")

    # Precompute a dictionary: frame index → DataFrame slice
    frame_dict = {frame: group for frame, group in events_by_frame}

    circles = np.empty((2 * n_rows * 2 * n_cols, 2), dtype=float)
    i = 0
    for row in range(-n_rows, n_rows):
        x_shift = (0.5 * g) if (row % 2) else 0.0
        for col in range(-n_cols, n_cols):
            circles[i][0] = col * g + x_shift
            circles[i][1] = row * dy

        i += 1

    print(f"-- Begin accumulating frames")
    for i in range(n_frames):
        print(f"Frame {i + 1} / {n_frames}")
        t = i * (1 / fps)
        camera_pos = offset + velvec * t
        x0 = camera_pos[0] % (2 * g * zoom)
        y0 = camera_pos[1] % (2 * g * zoom)

        ax1.clear()
        ax2.clear()
        # Anchor grid to camera center

        for circ in circles:
            ax1.add_patch(
                patches.Circle(
                    (circ[0] - x0, circ[1] - y0),
                    genparams.r_e * zoom,
                    edgecolor="black",
                    facecolor="none",
                    alpha=0.5,
                )
            )

        ax1.set_xlim(-view_size * zoom / 2, view_size * zoom / 2)
        ax1.set_ylim(-view_size * zoom / 2, view_size * zoom / 2)
        ax1.set_title(f"Fixed Camera View {i} (θ={theta_deg:.1f}°)")
        ax1.set_aspect("equal")
        ax1.plot(0, 0, "ro", markersize=5, label="Camera")
        if i in frame_dict:
            events_in_bin = np.array(frame_dict.get(i).iloc[:, 0])
            if len(events_in_bin) > 0:
                print(f"Plotting {len(events_in_bin)}")
                # Derive position using the same rule as lightlanes

                positions = offset + np.outer(events_in_bin, velvec)
                sctr_x.extend(positions[:, 0])
                sctr_y.extend(positions[:, 1])

                histogram_bins[i] = len(events_in_bin)

        ax2.set_xlim(0, duration)
        ax2.set_ylim(0, max(histogram_bins) * 1.1)
        ax2.bar(bin_edges[:i], histogram_bins[:i])
        ax2.set_title(f"Photon Events up to t={t:.2f}")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Photon Count")
        ax1.scatter(
            (sctr_x - camera_pos[0]) * zoom,
            (sctr_y - camera_pos[1]) * zoom,
            s=10,
            color="orange",
        )

        fig.canvas.draw()
        width, height = (
            fig.canvas.get_renderer().width,
            fig.canvas.get_renderer().height,
        )
        frame_image = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        frame_image = frame_image.reshape((int(height), int(width), 4))
        frame_image = frame_image[:, :, [1, 2, 3, 0]]  # ARGB to RGBA
        frames.append(frame_image)

    plt.close()
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    print(f"  Lightlane video generator  ")
    help = """
    Command line to generate videos using Lightlane metadata
    -id 412321
    -duration 40
    -fps = 24
    """

    meta_id_default = "29decde9"
    pp_id_default = "highenergy"  # Choose processing params that filter out all but very high energy events

    duration = 10
    fps = 10
    lamb = 0.12

    parser = argparse.ArgumentParser(description="Generate Lightlane video.")

    parser.add_argument(
        "-maxt", type=int, default=100, help="Maximum time from (default=24)"
    )
    parser.add_argument(
        "-a", type=str, required=False, default=meta_id_default, help="Meta ID to load"
    )
    parser.add_argument(
        "-duration", type=float, required=False, help="Video duration (seconds)"
    )
    parser.add_argument("-lamb", type=float, required=False, help="Wavelength lambda")
    parser.add_argument("-lwidth", type=float, required=False, help="Wavelength width")
    parser.add_argument("-pi", type=float, required=False, help="pi-channel")
    parser.add_argument(
        "-fps", type=int, default=24, help="Frames per second (default=24)"
    )
    parser.add_argument(
        "-pp", type=str, default=pp_id_default, help="Processing params"
    )

    parser.add_argument("-help", type=bool, default=False, help="Help (default=24)")
    args = parser.parse_args()

    help = args.help
    if help:
        print(help)
    if args.a:
        meta_id = args.a
    if args.duration:
        duration = args.duration
    if args.lamb:
        lamb = args.lamb
    if args.lwidth:
        lwidth = args.lwidth
    if args.fps:
        fps = args.fps

    pp = load_processing_param(args.pp)

    meta = load_fits_metadata(meta_id)
    genparam = load_gen_param(meta.gen_id)
    print(
        f"Generating video for id {args.a} {meta.star}, lambda: {lamb}, duration: {duration}s fps:{args.fps}"
    )
    fps = 24
    duration = 10
    take = 100

    frames = prepare_data(
        meta=meta,
        pparams=pp,
        genparams=genparam,
        take_first_seconds=take,
        duration=duration,
        fps=24,
    )

    plot_time_histogram(events=frames, fps=24, duration=duration)

    if True:
        make_video(
            events=frames,
            meta=meta,
            pparams=pp,
            genparams=genparam,
            fps=fps,
            duration=duration,
        )

    # meta: FitsMetadata = load_fits_metadata(f"meta_files/meta_{meta_id}.json")

    # make_video(meta, lamb=lamb, lwidth=lwidth, pi_channel=pi_channel, fps=fps)
