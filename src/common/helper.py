import numpy as np
import pandas as pd
from common.lanesheetMetadata import LanesheetMetadata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from common.fitsmetadata import (
    ComparePipeline,
    FitsMetadata,
    GenerationParameters,
    ProcessingParameters,
)
import os
from pandas import DataFrame

import csv
import os


def ensure_path_exists(path: str):
    try:
        if os.path.isdir(path):
            return True
        else:
            os.mkdir(path)
            return True

    except Exception as e:
        print(f"Failed to create {path}")
        raise Exception(f"Error creating folder {path}", repr(e))


def ensure_pipeline_folders_exists(meta: ComparePipeline):
    path = f"plots/{meta.id}"
    path2 = "files"
    success = True
    try:
        if os.path.isdir(path):
            print(f"Creating folder")
            success = True
        else:
            print(f"Creating folder")
            os.mkdir(path)
            success = True
        if os.path.isdir(path2):
            print(f"Creating folder {path}")
            success = True
        else:
            print(f"Creating folder {path2}")
            os.mkdir(path2)
            success = True
        return success

    except Exception as e:
        print(f"Failed to create {path}")
        raise Exception(f"Error creating folder {path}", repr(e))
        return False


def get_duration(meta: FitsMetadata, pp: ProcessingParameters):
    if pp.take_time_seconds:
        assert (
            pp.take_time_seconds < meta.t_max * 1.03
        ), f"Cannot take first {pp.take_time_seconds} seconds, because it is more than total length of {meta.t_max} seconds"
        duration = pp.take_time_seconds
    else:
        duration = meta.t_max - meta.t_min

    return duration


def get_wavelength_bins(pp: ProcessingParameters):
    wave_edges = np.linspace(
        pp.min_wavelength,
        pp.max_wavelength,
        pp.wavelength_bins + 1,
    )
    print("Bin edges:", wave_edges)
    wave_centers = 0.5 * (wave_edges[:-1] + wave_edges[1:])
    wave_widths = np.diff(wave_edges)

    return wave_edges, wave_centers, wave_widths


def get_uneven_time_bin_widths(pp: ProcessingParameters):
    # Using the ProcessingParameters, figure out what to do
    if pp.time_bin_widths_count is None or pp.take_time_seconds == 1:
        print("Using only one time-bin width")
        bin_widths = [pp.time_bin_seconds]
    else:
        print("Generating unevenly spaced ")
        t_bin_width_range = pp.time_bins_to - pp.time_bins_from
        t_bin_width_increment = t_bin_width_range / pp.time_bin_widths_count
        bin_widths = np.zeros(pp.time_bin_widths_count)
        bin_widths[0] = pp.time_bins_from
        for i in range(
            1,
            pp.time_bin_widths_count,
        ):
            bin_widths[i] = bin_widths[i - 1] + (
                t_bin_width_increment * np.random.uniform(0.7, 1.3)
            )

        print(
            f"Using {pp.time_bin_widths_count} widths, ranging from {pp.time_bins_from} to {pp.time_bins_to} seconds"
        )

    return bin_widths


def compare_variability_profiles(
    A_df,
    B_df,
    A_meta: FitsMetadata,
    B_meta: FitsMetadata,
    A_gen: GenerationParameters,
    B_gen: GenerationParameters,
):
    from scipy.stats import pearsonr

    # 1. Ensure matching wavelength bins
    wavelengths = A_df["Wavelength Bin"].values
    excess_real = A_df["Excess Variability"].values
    print(f"excess_real has {len(excess_real)} values")
    excess_synth = B_df["Excess Variability"].values
    print(f"excess_synth has {len(excess_synth)} values")

    # 2. Normalize to allow sign-independent comparisons
    real_abs = np.abs(excess_real)
    synth_abs = np.abs(excess_synth)

    # 3. Peak positions
    peak_idx_real = np.argmax(real_abs)
    peak_idx_synth = np.argmax(synth_abs)
    peak_wavelength_real = wavelengths[peak_idx_real]
    peak_wavelength_synth = wavelengths[peak_idx_synth]

    peak_diff = np.abs(peak_wavelength_real - peak_wavelength_synth)
    peak_amp_real = real_abs[peak_idx_real]
    peak_amp_synth = synth_abs[peak_idx_synth]
    peak_amp_diff = np.abs(peak_amp_real - peak_amp_synth)

    perc_deviance = np.abs(1 - (len(real_abs) / len(synth_abs)))
    assert (
        perc_deviance < 0.1
    ), f"The Real ({len(real_abs)} and Synth-set {len(synth_abs)} should have apporixmatley same number of values. Now > {perc_deviance*100}% difference"
    # 4. Shape similarity (absolute curve)
    mse = np.mean((real_abs - synth_abs) ** 2)
    mae = np.mean(np.abs(real_abs - synth_abs))
    corr, _ = pearsonr(real_abs, synth_abs)

    # 5. Bias in synthetic vs real (positive = synthetic overshoots)
    signed_bias = np.mean(excess_synth - excess_real)

    # 6. Low wavelength bins
    low_wavelength_mask = wavelengths < 0.35  # Emphasize left region

    low_wavelength_mse = np.average(
        (real_abs[low_wavelength_mask] - synth_abs[low_wavelength_mask]) ** 2
    )
    # Compose similarity summary
    summary = {
        "A_id": A_meta.id,
        "B_id": B_meta.id,
        "r_e": B_gen.r_e,
        "alpha": B_gen.alpha,
        "theta": B_gen.theta,
        "lucretius": B_gen.lucretius,
        "theta_change_per_sec": B_gen.theta_change_per_sec,
        "B_gen_id": B_gen.id if B_gen else None,
        "A_gen_id": A_gen.id if A_gen else None,
        "Peak λ real": peak_wavelength_real,
        "Peak λ synth": peak_wavelength_synth,
        "Peak λ diff": peak_diff,
        "Peak amp real": peak_amp_real,
        "Peak amp synth": peak_amp_synth,
        "Peak amp diff": peak_amp_diff,
        "MSE (abs)": mse,
        "Low wavelength MSE": low_wavelength_mse,
        "Combined MSE": low_wavelength_mse + mse,
        "MAE (abs)": mae,
        "Pearson corr (abs)": corr,
        "Signed bias (mean)": signed_bias,
    }
    for i in range(0, 7):
        summary[f"wbin_{i}"] = real_abs[i] - synth_abs[i]
    return summary


def estimate_lanecount_fast(ls: LanesheetMetadata):
    """
    Estimate the expected number of lightlanes in a cross section.

    Parameters
    ----------
    ls : LanesheetMetadata
        Metadata object containing lane parameters (r_e, alpha, lucretius, lambda_center).

    Returns
    -------
    float
        Estimated count of lightlanes based on geometric and physical parameters.
    """
    # Size of the lightlane cross section
    area_lane = np.pi * ls.r_e**2
    # Grid size : Increases with alpha and falls off with stronger negative lucretius parameter
    g = ls.alpha * np.exp(ls.lucretius * ls.lambda_center)
    # Area between lightlane center is a hexagonal grid. Each hexagon has an area equal to
    g_hex_cell = (np.sqrt(3) / 2) * g**2
    # Expected lightlane count is equal to the share of the lightlane cross section divided by the hex cell
    return area_lane / g_hex_cell


def rho_empirical(lambda_nm, A=385, lambda_0=0.38, sigma=0.77, C=5.0):
    """
    Empirical mimic of smoothed Chandra CXB spectrum using log-normal shape.
    Peaks at lambda_0, falls off smoothly toward both ends.

    Parameters
    ----------
    lambda_nm : float or array
        Wavelength in nanometers
    A : float
        Peak amplitude
    lambda_0 : float
        Peak wavelength (in nm)
    sigma : float
        Width of the log-normal curve
    C : float
        Asymptotic floor level

    Returns
    -------
    float or array
        Approximate counts per bin
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = (np.log(lambda_nm / lambda_0) / sigma) ** 2
        flux = A * np.exp(-log_term) + C

    return flux


def rho_empirical_2(
    lambda_nm, A=385, lambda_0=0.38, sigma=0.75, C=2.0, tail_amp=25.0, tail_exp=1.5
):
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = (np.log(lambda_nm / lambda_0) / sigma) ** 2
        lognorm = A * np.exp(-log_term)
        powerlaw_tail = tail_amp / (lambda_nm**tail_exp)
        flux = lognorm + powerlaw_tail + C

        # Tail correction for λ > 1.0
        tail_boost = np.ones_like(lambda_nm)
        mask = lambda_nm >= 1.0
        tail_boost[mask] = 1.0 + 1.2 * (lambda_nm[mask] - 1.0) ** 1.3  # Tunable growth

        return flux * tail_boost


def compare_dataframes(
    df1: pd.DataFrame, df2: pd.DataFrame, label1="DF1", label2="DF2", show_plot=True
):
    """
    Compare two DataFrames column by column.
    """

    print(f"Comparing datasets: {label1} vs {label2}")
    print(f"- {label1}: {len(df1)} rows")
    print(f"- {label2}: {len(df2)} rows")

    columns1 = set(df1.columns)
    columns2 = set(df2.columns)
    common_columns = columns1.intersection(columns2)
    only_in_df1 = columns1 - columns2
    only_in_df2 = columns2 - columns1

    if only_in_df1:
        print(f"Columns only in {label1}: {only_in_df1}")
    if only_in_df2:
        print(f"Columns only in {label2}: {only_in_df2}")

    print(f"Common columns: {common_columns}")

    results = {}

    for col in common_columns:
        series1 = df1[col].dropna()
        series2 = df2[col].dropna()

        mean1, mean2 = series1.mean(), series2.mean()
        std1, std2 = series1.std(), series2.std()
        min1, min2 = series1.min(), series2.min()
        max1, max2 = series1.max(), series2.max()

        results[col] = {
            f"{label1} mean": mean1,
            f"{label2} mean": mean2,
            f"{label1} std": std1,
            f"{label2} std": std2,
            f"{label1} min": min1,
            f"{label2} min": min2,
            f"{label1} max": max1,
            f"{label2} max": max2,
        }

        print(f"\nColumn: {col}")
        print(
            f"  {label1} mean: {mean1:.4f}, std: {std1:.4f}, min: {min1:.2f}, max: {max1:.2f}"
        )
        print(
            f"  {label2} mean: {mean2:.4f}, std: {std2:.4f}, min: {min2:.2f}, max: {max2:.2f}"
        )

        plt.figure(figsize=(8, 4))
        plt.hist(series1, bins=50, alpha=0.5, label=label1)
        plt.hist(series2, bins=50, alpha=0.5, label=label2)
        plt.title(f"Distribution of {col}")
        plt.legend()
        plt.grid(True)
        plt.xlabel(col)
        plt.ylabel("Counts")
        plt.savefig(f"plots/replicate_dataset/generator_plots_{col}.png")
        if show_plot:
            plt.show()
        plt.close()

    return results


def make_lambda_range(min_energy_kev: float, max_energy_kev: float):
    maxev = max_energy_kev * 1000.0
    minev = min_energy_kev * 1000.0

    return np.array([1239.84193 / minev, 1239.84193 / maxev])


def filter_by_wavelength_bin(
    events: pd.DataFrame, lambda_center: float, lwidth: float
) -> pd.DataFrame:
    """
    Filters the events DataFrame to include only rows within the given wavelength bin.

    Parameters:
    - events: pd.DataFrame with a "Wavelength Center" column
    - lambda_center: central wavelength of the bin
    - lwidth: width of the wavelength bin

    Returns:
    - Filtered DataFrame
    """
    halfwidth = lwidth / 2
    lower_bound = lambda_center - halfwidth
    upper_bound = lambda_center + halfwidth

    return events[
        (events["Wavelength (nm)"] >= lower_bound)
        & (events["Wavelength (nm)"] < upper_bound)
    ]


def randomly_sample_from(dataset, count: int, seed=None):
    print(f"Reducing dataset ...")
    """
    Randomly sample from real_df to match the number of rows in sim_df.

    Parameters
    ----------
    real_df : pd.DataFrame
        Full Chandra variability DataFrame
    sim_df : pd.DataFrame
        Simulated variability DataFrame to match length
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    sampled_real_df : pd.DataFrame
        Subset of real_df with same length as sim_df
    """

    if len(dataset) < count:
        raise Exception(
            f"Cannot downsample from too small dataset. Was {len(dataset)}, needs {count} "
        )

    print(
        f"Downsampling {count} from {len(dataset)}, taking only {(count / len(dataset)*100)}%"
    )

    return dataset.sample(n=count, random_state=seed, replace=True).reset_index(
        drop=True
    )


def read_from_csv(filename) -> DataFrame:
    """
    Append a result dict to a CSV file.
    Creates the file with headers if it doesn't exist.
    """
    data = pd.read_csv(filename)
    return data


def write_as_latex_table(df: DataFrame, columns_keep, column_labels, filename):
    file_exists = os.path.isfile(filename)

    table_str = df.to_latex(
        columns=columns_keep,
        header=column_labels,
        float_format="%.2e",
        escape=True,
    )
    with open(filename, mode="w", newline="") as f:
        f.write(table_str)


def write_result_to_csv(result: dict, filename):
    """
    Append a result dict to a CSV file.
    Creates the file with headers if it doesn't exist.
    """
    # Flatten tuple/list entries (like ci95_p_hat) into two separate columns
    flat_result = result.copy()
    if isinstance(flat_result.get("ci95_p_hat"), (tuple, list)):
        flat_result["ci95_p_hat_low"], flat_result["ci95_p_hat_high"] = flat_result.pop(
            "ci95_p_hat"
        )

    file_exists = os.path.isfile(filename)

    with open(filename, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=flat_result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(flat_result)


def split_into_time_bins(dataset, bins=100, seed=None):
    t_min = dataset["relative_time"].min()
    t_max = dataset["relative_time"].max()
    bin_edges = np.linspace(t_min, t_max, bins + 1)

    dataset["time_bin"] = pd.cut(dataset["relative_time"], bins=bin_edges, labels=False)

    # Group by the "time_bin" column and get row indices for each bin
    bin_indices = [group.index.to_numpy() for _, group in dataset.groupby("time_bin")]

    return bin_indices


def sample_evenly_from_wavelength_bins(dataset, count: int, bins=100, seed=None):
    print(f"Reducing dataset ...")

    if len(dataset) < count:
        raise Exception(
            f"Cannot downsample from too small dataset. Was {len(dataset)}, needs {count} "
        )

    # Step 1: Bin by wavelength
    dataset = dataset.copy()
    wave_min = dataset["Wavelength (nm)"].min()
    wave_max = dataset["Wavelength (nm)"].max()
    bin_edges = np.linspace(wave_min, wave_max, bins + 1)

    dataset["lambda_bin"] = pd.cut(
        dataset["Wavelength (nm)"], bins=bin_edges, labels=False
    )

    # Step 2: Compute samples per bin
    samples_per_bin = count // bins
    leftover = count % bins  # for uneven division, add remainder later

    sampled_frames = []

    rng = np.random.default_rng(seed)

    for b in range(bins):
        bin_df = dataset[dataset["lambda_bin"] == b]
        if len(bin_df) == 0:
            continue  # skip empty bins
        n = samples_per_bin + (1 if b < leftover else 0)
        replace = len(bin_df) < n
        sampled = bin_df.sample(n=n, random_state=rng.integers(0, 1e9), replace=replace)
        sampled_frames.append(sampled)

    sampled_df = pd.concat(sampled_frames).reset_index(drop=True)
    return sampled_df
