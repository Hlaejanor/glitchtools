import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from common.lanesheetMetadata import LanesheetMetadata
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from pandas import DataFrame


def save_to_csv(
    true_ls: LanesheetMetadata, signature, csv_path="beat/beat_metadata.csv"
):
    # Save to CSV

    if signature is None:
        return

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "theta",
                    "g",
                    "alpha",
                    "lucretius",
                    "r_e",
                    "phase",
                    "perp",
                    "lambda_center",
                    "peak_order_string",
                    "peak1",
                    "peak1pos",
                    "peak2",
                    "peak2pos",
                    "peak3",
                    "peak3pos",
                    "peak4",
                    "peak4pos",
                    "spacing_1_2",
                    "spacing_1_3",
                    "spacing_2_3",
                    "power_ratio_1_2",
                    "power_ratio_1_3",
                    "power_ratio_2_3",
                ]
            )  # header

        row = [
            true_ls.theta,
            true_ls.get_g(),
            true_ls.alpha,
            true_ls.lucretius,
            true_ls.r_e,
            true_ls.phase,
            true_ls.perp,
            true_ls.lambda_center,
            signature["peak_order_string"],
            signature["peak1"],
            signature["peak1pos"],
            signature["peak2"],
            signature["peak2pos"],
            signature["peak3"],
            signature["peak3pos"],
            signature["peak4"],
            signature["peak4pos"],
            signature["spacing_1_2"],
            signature["spacing_1_3"],
            signature["spacing_2_3"],
            signature["power_ratio_1_2"],
            signature["power_ratio_1_3"],
            signature["power_ratio_2_3"],
        ]
        writer.writerow(row)


def detect_lightlane_periodicities(
    timestamps,
    isotropic=False,
    plot=True,
    smooth=True,
    sigma=2.0,
    meta_id: str = None,
    true_vals: LanesheetMetadata = None,
):
    """
    Estimate time-domain periodicities in photon arrival times, potentially induced by lightlane crossings.

    Parameters:
    - timestamps: array of photon arrival times
    - dt_sampling: resolution for time series binning
    - t_max: optional, if not provided, inferred from max timestamp
    - isotropic: if True, labels the plot accordingly
    - plot: if True, plots the resulting power spectrum
    - smooth: if True, applies Gaussian smoothing to power spectrum
    - sigma: std deviation for Gaussian kernel (used if smooth=True)

    Returns:
    - g_freq: array of sampled periodicities (in seconds)
    - power: (optionally smoothed) power at each periodicity
    - peaks_idx: indices in g_freq of detected peaks
    """
    dt_sampling = 1.0  # Single second
    t_max = np.max(timestamps)

    n_bins = int(np.ceil(t_max / dt_sampling))
    time_series = np.zeros(n_bins)

    # Bin the events into time steps
    indices = (timestamps / dt_sampling).astype(int)
    indices = indices[indices < n_bins]
    np.add.at(time_series, indices, 1)

    # Remove DC component
    time_series -= np.mean(time_series)

    # Autocorrelation (emphasizes periodicity)
    corr = np.correlate(time_series, time_series, mode="full")
    corr /= np.max(corr)
    corr = corr[corr.size // 2 :]  # keep only positive lags

    # Fourier Transform of the autocorrelation
    power_spectrum = np.abs(np.fft.rfft(corr))
    freqs = np.fft.rfftfreq(len(corr), dt_sampling)
    freqs = freqs[1:]  # Skip zero-frequency
    power_spectrum = power_spectrum[1:]

    # Convert frequencies to equivalent g values (distance between lanes)
    with np.errstate(divide="ignore"):
        g_freq = 1.0 / freqs
    power = power_spectrum

    # Optional: remove inf/NaN at freq=0
    valid = ~np.isnan(g_freq) & ~np.isinf(g_freq)
    g_freq = g_freq[valid]
    power = power[valid]

    if smooth:
        power = gaussian_filter1d(power, sigma=sigma)

    # Detect peaks
    peaks_idx, _ = find_peaks(power)

    if plot:
        filename = "plots/beatplots/fourier_peaks.png"
        title = "Beat plot for Isotropic" if isotropic else "Beat plot Lightlane"
        filename = f"plots/beatplots/fourier_peaks_{meta_id}.png"
        plt.figure(figsize=(10, 5))
        title = f"Beat plot  θ: unknown {meta_id}"
        if true_vals is not None:
            title = f"Beat plot  θ: {true_vals.theta:.2f} g:{true_vals.get_g():.2f} {meta_id}"

            for i in range(1, 5):
                plt.axvline(
                    x=i * true_vals.get_projected_g(),
                    color="gray",
                    linestyle="--",
                    alpha=0.3,
                )

        plt.plot(
            g_freq,
            power,
            label="Smoothed Power Spectrum" if smooth else "Raw Power Spectrum",
        )
        plt.scatter(
            g_freq[peaks_idx], power[peaks_idx], color="red", label="Detected Peaks"
        )
        plt.xscale("log")
        plt.xlabel("g (effective periodicity)")
        plt.ylabel("Power")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        if filename is not None:
            plt.savefig(filename)
        plt.show()

    return g_freq, power, peaks_idx


def projected_periodicities(g, theta):
    # Triangular lattice directions
    phi = np.array([0, np.pi / 3, 2 * np.pi / 3])
    cos_terms = np.cos(theta - phi)
    return g / cos_terms


def generate_freq_signature(
    meta_id: str, points_ll: DataFrame, true_vals: LanesheetMetadata = None, plot=False
):
    # If the lanesheet is very sparse, it might be that no data is collected at all.
    if len(points_ll) == 0:
        return
    timestamps_ll = points_ll[:, 0]
    # Detect periodicities
    g_freq, power, peaks_idx = detect_lightlane_periodicities(
        timestamps=timestamps_ll,
        isotropic=False,
        smooth=True,
        sigma=3,
        plot=plot,
        meta_id=meta_id,
        true_vals=true_vals,
    )

    # Convert peak indices to g values
    peak_g_values = g_freq[peaks_idx]
    best_idx = np.argmax(power)
    g_best = g_freq[best_idx]
    print(f"  -- Found the best fit periodicity to be {g_best}")

    # Convert peak indices to g values
    peak_g_values = g_freq[peaks_idx]
    peak_powers = power[peaks_idx]

    # Sort by power, descending
    sorted_indices = np.argsort(peak_powers)[::-1]
    peak_order_string = "".join(str(i) for i in sorted_indices[:5])
    first = peak_g_values[sorted_indices[0]] if len(sorted_indices) > 0 else 1.0
    second = peak_g_values[sorted_indices[1]] if len(sorted_indices) > 1 else 1.0
    third = peak_g_values[sorted_indices[2]] if len(sorted_indices) > 2 else 1.0
    fourth = peak_g_values[sorted_indices[3]] if len(sorted_indices) > 3 else 1.0

    first_pow = peak_powers[sorted_indices[0]] if len(sorted_indices) > 0 else 1.0
    second_pow = peak_powers[sorted_indices[1]] if len(sorted_indices) > 1 else 1.0
    third_pow = peak_powers[sorted_indices[2]] if len(sorted_indices) > 2 else 1.0
    # fourth_pow = peak_powers[sorted_indices[3]] if len(sorted_indices) > 3 else 1.

    with np.errstate(divide="ignore"):
        results = {
            "peak_order_string": peak_order_string,
            "peak1": first,
            "peak1pos": sorted_indices[0] if len(sorted_indices) > 0 else None,
            "peak2": second,
            "peak2pos": sorted_indices[1] if len(sorted_indices) > 1 else None,
            "peak3": third,
            "peak3pos": sorted_indices[2] if len(sorted_indices) > 2 else None,
            "peak4": fourth,
            "peak4pos": sorted_indices[3] if len(sorted_indices) > 3 else None,
            "spacing_1_2": first / second,
            "spacing_1_3": first / third,
            "spacing_2_3": second / third,
            "power_ratio_1_2": first_pow / second_pow,
            "power_ratio_1_3": first_pow / third_pow,
            "power_ratio_2_3": second_pow / third_pow,
        }

    print(f"Best: {first:.3f}, second {second:.3f} third {third:.3f}")
    return results


def reconstruct_offset_from_phase_perp(est_vals):
    # Use predicted values of phase and per to restore offset

    v = np.array([np.cos(est_vals["theta"]), np.sin(est_vals["theta"])])
    v_perp = np.array([-v[1], v[0]])

    offset = est_vals["g"] * (est_vals["phase"] * v + est_vals["perp"] * v_perp)
    return offset


def canonicalize_theta(theta):
    """Map theta to [0, π/6] by folding over π/3 symmetry."""
    # Wrap to [0, π/3]
    theta = theta % (np.pi / 3)
    # Reflect into [0, π/6]
    return np.minimum(theta, np.pi / 3 - theta)
