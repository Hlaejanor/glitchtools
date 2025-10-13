import numpy as np
from scipy.ndimage import gaussian_filter1d
from astropy.stats import bayesian_blocks
from scipy.signal import find_peaks
from astropy.timeseries import LombScargle
from scipy.stats import chi2
from skimage import feature, transform
from common.fitsmetadata import (
    FitsMetadata,
    ChunkVariabilityMetadata,
    ProcessingParameters,
    ComparePipeline,
)

from event_processing.binning import add_time_binning
from common.fitsread import (
    fits_save_events_generated,
    read_event_data_crop_and_project_to_ccd,
    fits_save_chunk_analysis,
    fits_read,
)
from pandas import DataFrame
import pandas as pd
from event_processing.vartemporal_plotting import Arrow


def tally_votes_find_rige_peaks(ridges, threshold=5.0, plot=True):
    t_min, t_max = ridges["Time"].min(), ridges["Time"].max()
    dt = 100
    time_bins = np.arange(t_min, t_max + dt, dt)
    votes, edges = np.histogram(ridges["Time"], bins=time_bins)
    votes_smooth = gaussian_filter1d(votes, sigma=1)

    peaks, props = find_peaks(votes_smooth, height=np.percentile(votes_smooth, 90))
    peak_times = (edges[peaks] + edges[peaks + 1]) / 2
    peak_strengths = props["peak_heights"]
    if plot:
        plot_vote_map(edges, votes_smooth, peak_times, peak_strengths, True)

    return peak_times, peak_strengths


def plot_vote_map(edges, votes_smooth, peak_times, peak_strengths, add_multiples=True):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 4))
    plt.plot((edges[:-1] + edges[1:]) / 2, votes_smooth, label="Vote Map (smoothed)")
    plt.scatter(
        peak_times, peak_strengths, color="red", zorder=3, label="Detected Peaks"
    )
    multiples = np.arange(1, 12) * peak_times[1]
    multiples_s = np.zeros(len(multiples))
    plt.scatter(multiples, multiples_s, color="green", zorder=3, label="Multipled")
    plt.xlabel("Time [s]")
    plt.ylabel("Votes (persistence)")
    plt.legend()
    plt.show()


def compute_beat_cube(pixelmap: np.ndarray, max_lag: int = 12, q: float = 0.75):
    """
    pixelmap: 2D array (n_wbins, n_tbins) of counts
    max_lag:  largest time offset (in bins) to test
    Returns:
        beats: uint8 array (n_wbins, n_tbins, max_lag+1)
               beats[w, t, i] == 1 iff pixelmap[w, t] and pixelmap[w, t+i] are both above threshold
    """
    pm = np.asarray(pixelmap)
    beats = np.zeros((pixelmap.shape[0], max_lag + 1), dtype=np.uint8)
    wbins = pixelmap.shape[0]
    tbins = pixelmap.shape[1]
    for wbin in range(wbins):
        thresh = np.mean(pm[wbin])
        for t in range(tbins):
            for i in range(1, max_lag + 1):
                # To ensure that we give each frequency an equal lag
                if t + max_lag >= tbins:
                    continue
                a = pm[wbin, t]
                b = pm[wbin, t + i]
                if a < thresh and b < thresh:
                    # Pause beat
                    beats[wbin, i] += 1
                elif a > thresh and b > thresh:
                    # Signal beat
                    beats[wbin, i] += 1

    return beats


def diag_run_detector_45(
    pixelmap: np.ndarray,
    tbin_width_index,  # seconds per time bin (x-axis)
    tbin_widths: list[float],  # list of all time bin widths
    tbin_start: int = 0,  # optional time origin (bin)
    sigma: float = 3.0,  # Canny smoothing
    tol_deg: float = 3.0,  # +/- angle tolerance around 45°
    min_len: int = 12,  # minimum segment length in pixels
    max_gap: int = 1,  # max pixel gap allowed within a segment
    direction: str = "both",  # 'both' | 'slash' | 'backslash'
    handle: str = None,
):
    """
    Detect ~45° ridges in a 2D array (rows = wavelength bins, cols = time bins).
    - 'slash' (/)  means slope ≈ -1 (up-right)
    - 'backslash' (\) means slope ≈ +1 (down-right)
    """
    # 1) Edges
    edges = feature.canny(pixelmap, sigma=sigma)
    y_idxs, x_idxs = np.nonzero(edges)
    print(f"{handle} : Canny edges: {len(x_idxs)} pixels")

    # 2) Restrict Hough angles to ±45° (normal-angle space for skimage)
    targets = []
    if direction in ("both", "backslash"):  # slope ≈ +1 (down-right)
        targets.append(np.deg2rad(45))  # theta ≈ +45°
    if direction in ("both", "slash"):  # slope ≈ -1 (up-right)
        targets.append(np.deg2rad(-45))  # theta ≈ -45°

    thetas = np.concatenate(
        [
            np.linspace(t - np.deg2rad(tol_deg), t + np.deg2rad(tol_deg), 21)
            for t in targets
        ]
    )
    thetas = np.unique(thetas)

    # 3) Probabilistic Hough restricted to those angles
    segments = transform.probabilistic_hough_line(
        edges, threshold=10, line_length=min_len, line_gap=max_gap, theta=thetas
    )
    print(f"{handle} : Hough segments (±{tol_deg}° around 45°): {len(segments)}")

    # 4) Build per-segment summary + per-pixel ridge points
    seg_rows = []
    ridge_frames = []
    arrows = []
    for rid, ((x0, y0), (x1, y1)) in enumerate(segments):
        dx = x1 - x0
        dy = y1 - y0
        if dx == 0:
            continue
        slope = dy / dx
        angle_deg = np.degrees(np.arctan2(dy, dx))  # line orientation wrt x-axis

        # Keep only ~45° by slope check as a safety (± ~tan tolerance)
        # Expected slope ≈ +1 (backslash) or -1 (slash)
        want = []
        if direction in ("both", "backslash"):
            want.append(+1.0)
        if direction in ("both", "slash"):
            want.append(-1.0)
        if not any(abs(slope - sgn) <= np.tan(np.deg2rad(tol_deg)) for sgn in want):
            continue

        # Summaries
        seg_len = int(np.hypot(dx, dy))
        seg_rows.append(
            {
                "ridge_id": rid,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "slope": slope,
                "angle_deg": angle_deg,
                "length_pix": seg_len,
                "direction": "backslash" if slope > 0 else "slash",
            }
        )

        # Sample integer points along the segment (one per x pixel)
        xs = np.arange(min(x0, x1), max(x0, x1) + 1)
        ys = np.round(y0 + slope * (xs - x0)).astype(int)
        # Clamp to array bounds
        h, w = pixelmap.shape
        mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs, ys = xs[mask], ys[mask]
        tbin_width = tbin_widths[tbin_width_index]
        times = (tbin_start + xs) * tbin_width
        ridge_frames.append(
            pd.DataFrame(
                {
                    "ridge_id": rid,
                    "Time Bin Index": tbin_start + xs,
                    "Time": times,
                    "Wavelength Bin": ys,
                }
            )
        )
        arrows.append(Arrow(x0, y0, x1, y1, tbin_width_index))

    segments_df = pd.DataFrame(seg_rows)
    ridges_df = (
        pd.concat(ridge_frames, ignore_index=True)
        if ridge_frames
        else pd.DataFrame(
            columns=[
                "ridge_id",
                "Time Bin Index",
                "Time Bin Width",
                "Time",
                "Wavelength Bin",
            ]
        )
    )
    return ridges_df, segments_df, arrows


def convert_to_2D_numpy_image(
    binned_data: DataFrame,
    time_bin_widht_index: int,
    phase_bin_index: int,
    time_bin_widths: list[float],
    deviation_from_mean=False,
    handle: str = None,
) -> list[tuple[int, float]]:
    time_binning_column = f"Time Bin {time_bin_widht_index} {phase_bin_index}"
    binned_data_grouped = (
        binned_data.groupby(["Wavelength Bin", time_binning_column])
        .size()
        .sort_index()
        .reset_index(name="Count")
    )

    binned_data_grouped["Count"] = binned_data_grouped["Count"].astype("Int16")

    if deviation_from_mean:
        mean_count = binned_data_grouped["Count"].mean()
        binned_data_grouped["Count"] = np.abs(binned_data_grouped["Count"] - mean_count)

    all_wbins = np.arange(
        binned_data["Wavelength Bin"].min(), binned_data["Wavelength Bin"].max() + 1
    )
    all_tbins = np.arange(
        binned_data[time_binning_column].min(),
        binned_data[time_binning_column].max() + 1,
    )

    pivot = (
        binned_data_grouped.pivot_table(
            index="Wavelength Bin",
            columns=time_binning_column,
            values="Count",
            aggfunc="sum",
            fill_value=0,
        )
        .sort_index(axis=0)  # sort wavelengths (y)
        .sort_index(axis=1)  # sort time bins (x)
    )

    pivot = pivot.reindex(index=all_wbins, columns=all_tbins, fill_value=0)
    # Coerce every cell to numeric and convert to a float array for skimage
    pivot = pivot.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Pandas ≥ 2.0:
    pixelmap = pivot.to_numpy(dtype=np.float32)

    wbin_axis = pivot.index.to_numpy()  # y-axis labels (rows)
    tbin_axis = pivot.columns.to_numpy()  # x-axis labels (cols)
    pixelmap = pivot.to_numpy(dtype=np.float32)
    print(
        f"{handle} Created pixelmap using {len(wbin_axis)}) wbins and {len(tbin_axis)} tbins)"
    )
    return pixelmap, wbin_axis, tbin_axis


def analyze_bands(
    binned_data: DataFrame, time_bin_widht_index: int, time_bin_widths: list[float]
) -> list[tuple[int, float]]:
    max_time = np.max(binned_data["relative_time"])
    print(f"Max time was {max_time}")
    time_binning_column = f"Time Bin {time_bin_widht_index}"
    # 2. Count photons in each (Time Bin, Wavelength Bin)
    # Count number of rows where Hit == 1 for each time bin

    counts_per_tbin = (
        binned_data.loc[binned_data["Hit"] == 1, time_binning_column]
        .value_counts(sort=False)
        .sort_index()
        .reset_index(name="Counts")
        .rename(columns={"index": time_binning_column})
    )
    print(counts_per_tbin.head(100))
    # Sort the list so that the wbins and time-bins are sorted
    counts_per_tbin = counts_per_tbin.sort_values([time_binning_column])

    tbins = counts_per_tbin[time_binning_column].unique()
    # SELECT DISTINCT Time Bin
    tbin_width = time_bin_widths[time_bin_widht_index]
    tbin_times = tbin_width * tbins

    # Produce a list of all time bin indexes

    edges = find_band_times(tbin_times, counts_per_tbin)
    idx_arr = np.zeros_like(edges) + time_bin_widht_index
    print(f"Detected band edges at times {edges}")
    power = lomb_scargle_periodogram(tbins, edges)
    print(f"Lomb-Scargle power spectrum: {power}")

    test_rayleigh(edges, base_period=edges[1])
    return np.column_stack((idx_arr, edges))


def find_band_times(bin_times, counts_per_tbin):
    """
    Identify change points in time series data using Bayesian Blocks.

    Parameters:
    t (array-like): Array of time values (e.g., bin centers).
    y (array-like): Array of corresponding measurements (e.g., counts per bin).

    Returns:
    band_times (array-like): Array of times where significant changes occur.
    """

    # t: bin centers (s), y: counts per bin (all wavelengths)
    x = counts_per_tbin["Counts"].values
    t = bin_times
    # edges = bayesian_blocks(tbins, y, fitness="measures")  # change-points
    # Convert counts into an event list (one event per detected photon) and run Bayesian Blocks

    # p0 controls the false-alarm probability (larger p0 -> fewer, broader blocks).
    # Increase p0 if you are getting one block per bin; try values like 0.1-0.5 to detect only broad ~10% excursions.
    edges = bayesian_blocks(t=t, x=x, fitness="events", p0=0.01)
    print(f"Bayesian Blocks edges: {len(edges)}")
    # Convert edges to midpoints of “upward” steps
    # e.g., detect bins where mean after > mean before by >k sigma
    band_times = 0.5 * (edges[1:] + edges[:-1])
    return band_times


def lomb_scargle_periodogram(t, band_times):
    """
    Compute Lomb-Scargle periodogram for impulse series at band_times.
    """

    # Build impulse series
    dt = np.median(np.diff(t))
    impulse = np.zeros_like(t)
    idx = np.searchsorted(t, band_times)
    idx = np.clip(idx, 0, len(t) - 1)
    impulse[idx] = 1.0  # or small Gaussians

    freq = np.linspace(1 / 10000, 1 / 200, 5000)  # 200–10000 s
    power = LombScargle(t, impulse).power(freq)
    print(f"Power {power}")
    return power


def rayleigh_p(times, P):
    """Rayleigh test for periodicity at period P in event times.       "
    # times: event times (s), P: trial period (s)
    """
    phases = (2 * np.pi * (times % P)) / P
    C = np.sum(np.cos(phases))
    S = np.sum(np.sin(phases))
    R = np.sqrt(C * C + S * S) / len(phases)
    Z = len(phases) * R**2
    # Large-sample Rayleigh p-value approx:
    p = np.exp(-Z) * (1 + (2 * Z - Z**2) / (4 * len(phases)))  # optional correction
    return p, R


def test_rayleigh(band_times, base_period=2000.0):
    print(f"Test rayleigh: Number of bands {len(band_times)}")
    """Test Rayleigh periodicity detection on band_times."""
    P0 = base_period
    print(f"Using integer fractional reduction of basetime {base_period}")
    for fraction in [
        1,
        1 / 2,
        1 / 3,
        1 / 4,
        1 / 5,
        1 / 6,
        1 / 7,
        1 / 8,
        1 / 9,
        1 / 10,
        1 / 10,
        1 / 12,
    ]:  # include “forbidden” 1/5,1/7
        p, R = rayleigh_p(band_times, P0 * fraction)
        print(
            f"Period {P0*fraction:7.1f} s: p={p:.3e}, mean resultant length R={R:.3f}"
        )

    print(f"Using integer multple incresed basetime {base_period}")
    for i in range(0, 12):
        p, R = rayleigh_p(band_times, P0 * i)
        print(
            f"Period {P0*fraction:7.1f} s: p={p:.3e}, mean resultant length R={R:.3f}"
        )
