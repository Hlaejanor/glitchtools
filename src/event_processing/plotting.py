import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm
from common.helper import (
    get_duration,
    ensure_pipeline_folders_exists,
    get_uneven_time_bin_widths,
    get_wavelength_bins,
)
from common.fitsmetadata import (
    ChunkVariabilityMetadata,
    FitsMetadata,
    ProcessingParameters,
)
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from common.powerdensityspectrum import (
    PowerDensitySpectrum,
    Spectrum,
    exp_count_per_sec,
)
import matplotlib as mpl
import seaborn as sns
from typing import Optional, Tuple
from matplotlib.colors import LogNorm


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


def plot_multiple_series(
    x_vals: np.ndarray,
    y_series: np.ndarray,
    l0_vals: np.ndarray,
    filename: str,
    show=True,
):
    """
    kind: 'counts_per_second' (default) or 'total_counts'
    show_bin_widths: if True, plots steps rather than dots
    logy: if True, y-axis will be log-scaled
    """

    # Extract wavelength and count

    plt.clf()

    for i in range(0, len(y_series)):
        plt.plot(
            x_vals,
            y_series[i],
            label=f"l_0: {l0_vals[i]:.2e}",
        )

    # pds = PowerDensitySpectrum(meta.spectrum, "wavelength")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Expected lanecount")

    title = f"Expected lanecount varying l0"

    plt.title(title)
    plt.grid(True)
    plt.legend()

    plt.savefig(filename, dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_spectrum_vs_data(
    meta_A: FitsMetadata,
    source_A: DataFrame,
    filename: str,
    meta_B: FitsMetadata = None,
    source_B: DataFrame = None,
    show=False,
):
    """
    kind: 'counts_per_second' (default) or 'total_counts'
    show_bin_widths: if True, plots steps rather than dots
    logy: if True, y-axis will be log-scaled
    """

    # Extract wavelength and count

    # Group by wavelength bin and count
    grouped_data_A = (
        source_A.groupby(["Wavelength Center", "Wavelength Width"])
        .size()
        .reset_index(name="Count")
    )
    if meta_A.apparent_spectrum is None:
        print(f"Meta A {meta_A.id} has no apparent spectrum, cannot plot")
        return
    rho_empirical_A = exp_count_per_sec(
        grouped_data_A["Wavelength Center"],
        meta_A.apparent_spectrum.A,
        meta_A.apparent_spectrum.lambda_0,
        meta_A.apparent_spectrum.sigma,
        meta_A.apparent_spectrum.C,
    )
    plt.clf()
    plt.plot(
        grouped_data_A["Wavelength Center"],
        rho_empirical_A,
        label=f"Parametric {meta_A.star}",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )
    duration_A = meta_A.t_max - meta_A.t_min
    grouped_data_A["Flux per sec"] = grouped_data_A["Count"] / (
        duration_A * grouped_data_A["Wavelength Width"]
    )

    # Plot both empirical and synthetic
    plt.plot(
        grouped_data_A["Wavelength Center"],
        grouped_data_A["Flux per sec"],
        label=f"Observed {meta_A.star}",
        linestyle="-",
        linewidth=1,
        alpha=0.7,
    )
    if meta_B is not None:
        print(f"A length : {len(source_A)} vs B Length {len(source_B)}")
        grouped_data_B = (
            source_B.groupby(["Wavelength Center", "Wavelength Width"])
            .size()
            .reset_index(name="Count")
        )

        print(f"A wbins: {len(grouped_data_A)} vs B Wbins {len(grouped_data_B)}")

        # wavelength_B = grouped_data_B["Wavelength Center"]

        duration_B = meta_B.t_max - meta_B.t_min
        grouped_data_B["Flux per sec"] = grouped_data_B["Count"] / (
            duration_B * grouped_data_B["Wavelength Width"]
        )

        rho_empirical_B = exp_count_per_sec(
            grouped_data_B["Wavelength Center"],
            meta_B.apparent_spectrum.A,
            meta_B.apparent_spectrum.lambda_0,
            meta_B.apparent_spectrum.sigma,
            meta_B.apparent_spectrum.C,
        )
        # Plot both empirical and syntheti
        plt.plot(
            grouped_data_B["Wavelength Center"],
            rho_empirical_B,
            label=f"Spectrum {meta_B.star}",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )

        plt.plot(
            grouped_data_B["Wavelength Center"],
            grouped_data_B["Flux per sec"],
            label=f"Observed {meta_B.star}",
            linestyle="-",
            linewidth=1,
            alpha=0.7,
        )

    # pds = PowerDensitySpectrum(meta.spectrum, "wavelength")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Hits per sec")

    title = "Photon flux per sec"

    plt.title(title)
    plt.grid(True)
    plt.legend()

    plt.savefig(filename, dpi=300)
    if show:
        plt.show()
    plt.close()
    plt.close()


def plot_wbin_time_heatmap(
    data: DataFrame,
    dt: float,
    filename: str,
    wbins: list,
    wbin_col: str = "Wavelength (nm)",
    show: bool = True,
    tcol: str = "time",
    # e.g. "Hit" if you want to filter True hits
    log_scale: bool = False,  # log color scale via log10(count+1)
    cmap: str = "viridis",
    handle: str = None,
):
    """
    Heatmap: PI channel (y) vs time (x), color = counts in (dt, pi_bin_width) bins.

    Returns a dict with 'time_edges' and 'pi_edges' used for binning.
    """

    if data.empty:
        raise ValueError("No data to plot after filtering.")

    # --- Build time & PI bin edges ---
    tmin = float(data[tcol].min())
    tmax = float(data[tcol].max())
    # Start/stop on clean bin boundaries
    t0 = np.floor(tmin / dt) * dt
    t1 = np.ceil(tmax / dt) * dt
    time_edges = np.arange(t0, t1 + dt, dt)

    plt.ylabel("Wavelength")

    # --- 2D histogram: note the order (time, PI); we transpose for imshow ---
    H, xedges, yedges = np.histogram2d(
        data[tcol].to_numpy(),
        data[wbin_col].to_numpy(),
        bins=[time_edges, wbins],
    )
    # H shape: (len(time_edges)-1, len(pi_edges)-1) → transpose so rows = PI
    M = H.T

    if log_scale:
        M_plot = np.log10(M + 1.0)
        cbar_label = "log10(Counts + 1) per (dt, pi-bin)"
    else:
        M_plot = M
        cbar_label = "Counts per (dt, pi-bin)"

    extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])

    # --- Plot ---

    fig, ax = plt.subplots(figsize=(12, 6), dpi=1000)
    im = ax.imshow(
        M_plot,
        origin="lower",
        aspect="auto",  # <-- right place
        extent=extent,
        cmap=cmap,
    )

    cbar = fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Wavelength (nm)")
    ax.set_title(f"{handle} Wavelength × Time heatmap  (dt = {dt:g}s)")
    fig.tight_layout()
    fig.savefig(f"./plots/{filename}", dpi=1000)
    if show:
        plt.show()
    plt.close(fig)

    return


def plot_ccd_energy_map(
    data: DataFrame, dt: float, filename: str, show=False, handle: str = None
):
    """
    Bins the CCD hits over (CCD X, CCD Y), sums the energies (keV) in each pixel,
    and shows a 2D heatmap.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns:
          - "Hit" (boolean or int) indicating whether this photon was detected
          - "CCD X", "CCD Y" (pixel coordinates)
          - "Wavelength (nm)" for each photon
        Optionally ensure all relevant columns are present and well-defined.
    filename : str
        The file path to save the resulting plot
    show : bool
        Whether to display the plot interactively
    """
    print(f"Plotting energy map to {filename}")
    # 1. Filter for actual hits (Hit==1 or True)

    # 2. Convert wavelength (nm) to photon energy (keV).
    #    E(eV) = 1239.84193 / λ(nm) -> E(keV) = [1239.84193 / λ(nm)] / 1000
    def nm_to_keV(nm):
        # Avoid divide-by-zero if nm can be 0
        return (1239.84193 / nm) / 1000.0 if nm > 0 else 0.0

    data["Energy (keV)"] = data["Wavelength (nm)"].apply(nm_to_keV)

    # 3. Determine the pixel grid extents.
    #    Convert X/Y to int if not already.
    data["CCD X"] = data["CCD X"].astype(int)
    data["CCD Y"] = data["CCD Y"].astype(int)

    min_x, max_x = data["CCD X"].min(), data["CCD X"].max()
    min_y, max_y = data["CCD Y"].min(), data["CCD Y"].max()

    width = max_x - min_x + 1
    height = max_y - min_y + 1

    # 4. Create a 2D array ("energy_map") to accumulate total energy in keV per pixel.
    energy_map = np.zeros((height, width), dtype=np.float64)

    # 5. Fill the energy_map
    for _, row in data.iterrows():
        x = row["CCD X"] - min_x
        y = row["CCD Y"] - min_y
        energy_map[y, x] += row["Energy (keV)"]

    # 6. Plot the heatmap using imshow.
    plt.figure(figsize=(8, 8))

    # Use 'origin="lower"' so that (min_y, min_x) is at bottom-left in the plot.
    # The extent makes coordinates in the plot match actual CCD X/Y values.
    plt.imshow(
        energy_map,
        origin="lower",
        cmap="turbo",  # or 'jet', 'inferno', etc.
        extent=(min_x, max_x + 1, min_y, max_y + 1),
    )
    plt.colorbar(label="Sum of Photon Energy (keV)")

    plt.xlabel("CCD X")
    plt.ylabel("CCD Y")
    plt.title(f"Energy Intensity Map (Sum of photon energies per pixel) t:{dt:.0f}s")
    plt.tight_layout()
    plt.savefig(f"./plots/{filename}", dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_ccd_bin(data, filename, show=False, handle: str = None):
    # Count hits per pixel
    hit_counts = (
        data[data["Hit"]].groupby(["CCD X", "CCD Y"]).size().reset_index(name="Counts")
    )

    # Normalize hit counts to map to colors
    norm = plt.Normalize(
        vmin=hit_counts["Counts"].min(), vmax=hit_counts["Counts"].max()
    )
    colors = plt.cm.Reds(norm(hit_counts["Counts"]))

    # Plot the hits and misses on the CCD

    plt.figure(figsize=(8, 8))

    plt.scatter(
        hit_counts["CCD X"], hit_counts["CCD Y"], color=colors, s=10, label="Hits"
    )
    plt.gca().set_facecolor("black")
    plt.title("CCD count plot")
    plt.xlabel("CCD X")
    plt.ylabel("CCD Y")
    plt.colorbar(label="Counts")
    plt.legend()
    plt.axis("equal")
    plt.savefig(f"./plots/{filename}", dpi=300)
    if show:
        plt.show()
    plt.close()


def fractional_metrics(counts, dt):
    """
    counts: 1D array for a single chunk of fixed duration T_chunk
    dt: bin width (seconds)
    Returns dimensionless fractional metrics comparable across dt.
    """
    n = len(counts)
    mu = counts.mean()
    S2 = counts.var(ddof=1)

    # --- Fractional excess (counts) ---
    excess = max(S2 - mu, 0.0)
    F_var = np.sqrt(excess) / mu if mu > 0 else 0.0  # fractional RMS
    F_excess = (S2 - mu) / mu if mu > 0 else 0.0  # Fano-excess

    # --- Neighbor-differences (counts) ---
    diffs = np.diff(counts)
    if len(diffs) >= 2:
        S2_diff = diffs.var(ddof=1)
        excess_diff = max(S2_diff - 2 * mu, 0.0)
        F_adj = np.sqrt(excess_diff) / mu if mu > 0 else 0.0
    else:
        F_adj = 0.0

    # --- Optional: do rates as a cross-check ---
    rates = counts / dt
    mu_r = rates.mean()
    S2_r = rates.var(ddof=1)
    shot_r = mu_r / dt  # Poisson shot-noise variance in rate space
    excess_r = max(S2_r - shot_r, 0.0)
    F_var_rate = np.sqrt(excess_r) / mu_r if mu_r > 0 else 0.0

    return {
        "F_var": F_var,
        "F_excess": F_excess,
        "F_adj": F_adj,
        "F_var_rate": F_var_rate,
        "mu": mu,
        "n_bins": n,
        "dt": dt,
    }


def plot_periodicity_winners(
    A_winners, B_winners, wavelength_centers, filename, show=True
):
    # The second column contains the period computed in seconds
    if A_winners is not None:
        A_winners = np.array(A_winners)
        x_A = A_winners[:, 2]
        A_winning_idx = A_winners[:, 0].astype(int)
        y_A = wavelength_centers[A_winning_idx]

        plt.scatter(
            x_A,
            y_A,
            label="A",
            alpha=0.4,
        )

    if B_winners is not None:
        B_winners = np.array(B_winners)
        x_B = B_winners[:, 2]
        B_winning_idx = B_winners[:, 0].astype(int)
        y_B = wavelength_centers[B_winning_idx]

        plt.scatter(
            x_B,
            y_B,
            label="B",
            alpha=0.01,
        )
    # Adjust the axes
    plt.xlabel("Periodicity in seconds", fontsize=14)
    plt.ylabel("Wavelength (nm)", fontsize=14)

    # Add title and grid
    plt.title(
        "Periodicities",
        fontsize=16,
    )
    plt.legend(fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Customize tick marks for better legibility
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Save and show
    plt.savefig(f"./plots/{filename}", dpi=300)
    if show:
        plt.show()
    plt.close()


def odd_even_contrast(tbins_counts, signed=False, use_pooled=False, eps=1e-12):
    """
    tbins_counts: 1D array-like of length 12 (counts per time bin, 0-indexed).
    Returns S_alt (>=0 if signed=False; signed if signed=True).

    - use_pooled=False: denom uses overall mean (stable & symmetric).
    - use_pooled=True : denom uses mean_odd/n_odd + mean_even/n_even (slightly more 'local').
    """
    x = np.asarray(tbins_counts, dtype=float)
    if x.size % 2 == 1:
        raise ValueError("tbins_counts must have an equal length")

    even = x[::2]  # 0,2,4,6,8,10
    odd = x[1::2]  # 1,3,5,7,9,11
    n_even = even.size  # 6
    n_odd = odd.size  # 6

    mean_even = even.mean()
    mean_odd = odd.mean()
    mean_all = x.mean()

    # Difference (odd - even); take abs unless signed requested
    num = mean_odd - mean_even
    if not signed:
        num = abs(num)

    # Denominator under Poisson null
    if use_pooled:
        # Slightly more conservative: uses separate means
        denom_var = (mean_odd / n_odd) + (mean_even / n_even)
    else:
        # Symmetric and stable: uses overall mean
        denom_var = mean_all * (1.0 / n_odd + 1.0 / n_even)

    denom = np.sqrt(max(denom_var, eps))
    return float(num / denom)


def get_chunk_variability(wbin, tbins_counts, wbin_shot_noise):
    # The poisson noise for this chunk is equal to the mean of the chunks time-bin counts
    chunk_local_wbin_shot_noise = np.mean(
        tbins_counts
    )  # expected Poisson variance = mean

    # --- Global variance ---
    S2 = np.var(tbins_counts, ddof=1)  # sample variance
    total_std = np.std(tbins_counts, ddof=1)

    # --- Signed excess variance  ---
    signed_excess_var = S2 - chunk_local_wbin_shot_noise
    signed_excess_std = np.sign(signed_excess_var) * np.sqrt(abs(signed_excess_var))
    signed_excess_std_smoothed = (
        signed_excess_std / np.sqrt(chunk_local_wbin_shot_noise)
        if chunk_local_wbin_shot_noise > 0
        else 0
    )

    # --- Fano excess ---
    # Compute the Fano excess using both local and global shot noise estimates
    fano_excess_local = (
        (S2 - chunk_local_wbin_shot_noise) / chunk_local_wbin_shot_noise
        if chunk_local_wbin_shot_noise > 0
        else 0
    )
    fano_excess_global = (
        (S2 - wbin_shot_noise) / wbin_shot_noise if wbin_shot_noise > 0 else 0
    )
    # fano_excess = 0 → Poisson
    # fano_excess > 0 → extra variance
    # fano_excess < 0 → suppressed variance

    # --- Neighbour differences  ---
    diffs = np.diff(tbins_counts)
    S2_diffs = np.var(diffs, ddof=1)
    shotnoise_diffs = 2 * chunk_local_wbin_shot_noise
    signed_excess_adj_var = S2_diffs - shotnoise_diffs

    ves = np.sign(signed_excess_adj_var) * np.sqrt(abs(signed_excess_adj_var))
    vesa = (
        ves / np.sqrt(chunk_local_wbin_shot_noise)
        if chunk_local_wbin_shot_noise > 0
        else 0
    )
    oddeven = odd_even_contrast(
        tbins_counts=tbins_counts, signed=False, use_pooled=True
    )
    observation = {
        "Wavelength Bin": wbin,
        "Total Variability": total_std,
        "Shot Noise": np.sqrt(chunk_local_wbin_shot_noise),
        "Excess Variability": signed_excess_std,
        "Excess Variability Smoothed": signed_excess_std_smoothed,
        "Fano Excess Local Variability": fano_excess_local,
        "Fano Excess Global Variability": fano_excess_global,
        "Variability Excess Adjacent": ves,
        "Odd even contrast": oddeven,
        "Variability Excess Smoothed Adjacent": vesa,
        "Zero Count Bins": np.sum(tbins_counts == 0),
        "One Count Bins": np.sum(tbins_counts == 1),
    }

    return observation, chunk_local_wbin_shot_noise


def take_top_wavelengths_per_timescale(
    variability_percentile: int,
    variability_observations: pd.DataFrame,
    variability_type: str,
    *,
    time_bin_col: str = "Time Bin Width",
    wavelength_bin_col: str = "Wavelength Bin",
) -> pd.DataFrame:
    print(f"take_top_wavelengths_per_timescale() N ={variability_percentile}")
    """
    For each time-scale bin (e.g., 'Time Bin Width'), select the top-K  wavelength bins
    ranked by the chosen variability metric.

    Steps:
      1) Validate columns and coerce metric to numeric.
      2) Within each (time_bin, wavelength_bin), keep the row with the MAX metric (idxmax).
      3) Within each time_bin, sort wavelengths by metric descending and take top-K.

    Returns:
      A DataFrame consisting of the selected rows across all time bins, with two helper columns:
        - '_RankWithinTimeBin': rank (1..K) by variability within the time bin
        - '_TimeBinValue': the time-bin value used for grouping (copy of time_bin_col)

    Notes:
      - If a time bin has fewer than K distinct wavelength bins, it returns all available.
      - This neutralizes chunk multiplicity per wavelength by collapsing to one 'best' row per wavelength per time bin.
    """

    valid_metrics = {
        "Excess Variability",
        "Variability Excess Adjacent",
        "Excess Variability Smoothed",
        "Fano Excess Local Variability",
        "Fano Excess Global Variability",
        "Odd even contrast",
        "Variability Excess Smoothed Adjacent",
    }
    assert (
        variability_type in valid_metrics
    ), f"Error: '{variability_type}' not in valid variability metrics: {sorted(valid_metrics)}"

    required_cols = {time_bin_col, wavelength_bin_col, variability_type}
    missing = required_cols.difference(variability_observations.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    df = variability_observations.copy()

    # Coerce metric numeric once
    df["_Excess_metric"] = pd.to_numeric(df[variability_type], errors="coerce")

    # 1) Reduce to one best row per (time_bin, wavelength_bin) by metric
    #    (idxmax handles ties by first occurrence)
    idx = (
        df.groupby([time_bin_col, wavelength_bin_col], sort=False)["_Excess_metric"]
        .idxmax()
        .dropna()
        .astype(int)
    )
    best_per_wavelength = df.loc[idx].copy()

    # 2) Within each time_bin, take top-K wavelengths by metric
    best_per_wavelength.sort_values(
        [time_bin_col, "_Excess_metric"], ascending=[True, False], inplace=True
    )

    # Rank within time bin for convenience
    best_per_wavelength["_RankWithinTimeBin"] = (
        best_per_wavelength.groupby(time_bin_col)["_Excess_metric"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    selected = best_per_wavelength[
        best_per_wavelength["_RankWithinTimeBin"] <= variability_percentile
    ].copy()

    selected["_TimeBinValue"] = selected[time_bin_col]

    # Clean up helper column if you prefer
    # selected = selected.drop(columns=["_Excess_metric"])

    # Ensure deterministic column order: helpers at the end
    helper_cols = ["_Excess_metric", "_RankWithinTimeBin", "_TimeBinValue"]
    ordered_cols = [c for c in selected.columns if c not in helper_cols] + helper_cols
    selected = selected[ordered_cols]

    return selected


def take_max_variability(
    variability_percentile: int,
    variability_observations: DataFrame,
    variability_type: str,
) -> DataFrame:
    print(f"take_max_variability() {variability_percentile}")
    """
    Return up to variability_percentile  with the largest 'Excess Variability'.
    """
    variability_percentile = int(variability_percentile)
    assert variability_type in [
        "Excess Variability",
        "Variability Excess Adjacent",
        "Excess Variability Smoothed",
        "Fano Excess Local Variability",
        "Fano Excess Global Variability",
        "Odd even contrast",
        "Variability Excess Smoothed Adjacent",
    ], f"Error : Could not find {variability_type} in list of valid variability metrics"

    if variability_type not in variability_observations.columns:
        raise KeyError(
            f"Column '{variability_type}' not found in variability_observations"
        )

    top = variability_observations.sort_values(
        by=variability_type, ascending=False
    ).head(variability_percentile)

    return top


def take_max_variability_per_wbin(
    variability_percentile: int,
    variability_observations: DataFrame,
    variability_type: str,
) -> DataFrame:
    print(f"take_max_variability_per_wbin() {variability_percentile}")
    """
    Return up to variability_percentile rows per (Time Bin Width index, Wavelength Bin)
    with the largest 'Excess Variability'.

    This implementation:
     - validates the requested count,
     - coerces 'Excess Variability' to numeric to avoid lexicographic sorting,
     - sorts each group in descending numeric order and takes the top rows,
     - returns an empty DataFrame with the same columns if no results.
    """

    assert variability_type in [
        "Excess Variability",
        "Variability Excess Adjacent",
        "Excess Variability Smoothed",
        "Fano Excess Local Variability",
        "Fano Excess Global Variability",
        "Odd even contrast",
        "Variability Excess Smoothed Adjacent",
    ], f"Error : Could not find {variability_type} in list of valid variability metrics"

    if variability_type not in variability_observations.columns:
        raise KeyError(
            f"Column '{variability_type}' not found in variability_observations"
        )

    parts = []
    # coerce to numeric once to avoid repeating conversion and to ensure numeric sorting
    exv_numeric = pd.to_numeric(
        variability_observations[variability_type], errors="coerce"
    )
    variability_observations = variability_observations.copy()
    variability_observations["_Excess_metric"] = exv_numeric

    # group without forcing a particular sort order (preserve original order otherwise)
    groups = variability_observations.groupby(["Wavelength Bin"], sort=False)

    min_chunks_per_wbin = groups.size().min()

    for _, group in groups:
        # downsample so that all groups have the same number of chunks
        downsampled = group.sample(n=min_chunks_per_wbin, random_state=42)
        # sort by the numeric excess variability (NaNs go to the end) and take top N
        top = downsampled.sort_values(by="_Excess_metric", ascending=False).head(
            variability_percentile
        )
        # print(top.head(10))
        parts.append(top)

    if not parts:
        return variability_observations.head(0).drop(columns=["_Excess_metric"]).copy()

    filtered_table = pd.concat(parts, ignore_index=True)
    # remove helper column before returning
    if "_Excess_metric" in filtered_table.columns:
        filtered_table = filtered_table.drop(columns=["_Excess_metric"])

    return filtered_table


def monte_carlo_chunk_variability(
    df: DataFrame,
    pp,  # ProcessingParameters
    time_bin_width: float,
    N: int = 1,
    time_binning_column: str = "time_bin",
    wavelength_bin_column: str = "Wavelength Bin",
    rng: Optional[np.random.Generator] = None,
) -> DataFrame:
    """
    Draw N chunk-level variability observations for a given time_bin_width.

    """
    # print(f"monte_carlo_chunk_variability() ")
    if rng is None:
        rng = np.random.default_rng()

    chunk_len = int(pp.time_bin_chunk_length)

    # 1) Count photons in each (tbin, wbin)
    grouped = (
        df.groupby([time_binning_column, wavelength_bin_column])
        .size()
        .reset_index(name="Counts")
    )

    # 2) Ensure full rectangular grid (fills missing with 0)
    tbins = np.sort(grouped[time_binning_column].unique())
    wbins = np.sort(grouped[wavelength_bin_column].unique())

    full_index = pd.MultiIndex.from_product(
        [tbins, wbins], names=[time_binning_column, wavelength_bin_column]
    )
    rect = (
        grouped.set_index([time_binning_column, wavelength_bin_column])
        .reindex(full_index, fill_value=0)
        .reset_index()
        .sort_values([wavelength_bin_column, time_binning_column])
        .reset_index(drop=True)
    )

    # After reindex, make sure tbins are contiguous 0..num_tbins-1 (optional but convenient)
    # Map original tbin labels to 0..K-1
    tbin_old = np.sort(rect[time_binning_column].unique())
    tbin_map = {old: i for i, old in enumerate(tbin_old)}
    rect[time_binning_column] = rect[time_binning_column].map(tbin_map)

    # Recompute tbins array and counts per wbin time series
    tbins = np.sort(rect[time_binning_column].unique())
    num_tbins = len(tbins)
    if num_tbins < chunk_len:
        raise ValueError(
            f"Not enough time bins ({num_tbins}) for chunk length {chunk_len}. Try reducing the maximum time bin length"
        )

    # Build dict: wbin -> list of counts over time (aligned by tbin 0..num_tbins-1)
    counts_ts_by_wbin = (
        rect.pivot(
            index=time_binning_column, columns=wavelength_bin_column, values="Counts"
        )
        .sort_index()
        .fillna(0)
        .astype(int)
    )
    # Now columns are wbin labels, rows are tbin 0..num_tbins-1

    # Lookups for wavelength center/width (if needed downstream)
    wbin_meta = df.drop_duplicates(wavelength_bin_column).set_index(
        wavelength_bin_column
    )
    # Safe access (use get with defaults if columns might be missing)
    wcenter_col = (
        "Wavelength Center" if "Wavelength Center" in wbin_meta.columns else None
    )
    wwidth_col = "Wavelength Width" if "Wavelength Width" in wbin_meta.columns else None

    variability_list = []

    for _ in range(N):
        # Pick a random existing wbin label (do not assume 0..max)
        wbin = rng.choice(counts_ts_by_wbin.columns.to_numpy())

        # Pick a valid start index for a contiguous chunk
        tbin_start = rng.integers(0, num_tbins - chunk_len + 1)
        tbin_to = tbin_start + chunk_len

        # Time series for this wavelength bin
        w_ts = counts_ts_by_wbin[wbin].to_numpy()

        # Stationary Poisson mean (shot-noise baseline) for this wbin
        wbin_lambda = float(w_ts.mean())

        # Choose real vs randomized (Poisson) chunk
        chunk_counts = w_ts[tbin_start:tbin_to]

        # Fetch metadata (if present)
        wbin_w_center = (
            float(wbin_meta.loc[wbin, wcenter_col]) if wcenter_col else np.nan
        )
        wbin_w_width = float(wbin_meta.loc[wbin, wwidth_col]) if wwidth_col else np.nan

        # Compute variability metric for the chunk
        # (Assumes your get_chunk_variability returns a dict-like "observation" and used_mean)
        observation, wbin_used_mean_count = get_chunk_variability(
            wbin, chunk_counts, wbin_lambda
        )

        # Attach context
        observation["Time Bin from index"] = int(tbin_start)
        observation["Time Bin to index"] = int(tbin_to)
        observation["Wavelength Bin"] = (
            int(wbin) if isinstance(wbin, (int, np.integer)) else wbin
        )
        observation["Wavelength Center"] = wbin_w_center
        observation["Wavelength Width"] = wbin_w_width
        observation["Time Bin Width"] = float(time_bin_width)
        observation["Mean Count"] = float(wbin_used_mean_count)

        variability_list.append(observation)

    return pd.DataFrame(variability_list)


def compute_time_variability_async(
    binned_datasets: DataFrame,
    meta: FitsMetadata,
    pp: ProcessingParameters,
    time_bin_widths,
) -> Tuple[DataFrame, ChunkVariabilityMetadata]:
    print(f"compute_time_variability_async() {meta.star}")
    for df in binned_datasets:
        assert (
            "Wavelength Bin" in df.columns
        ), "Compute Time Variability requires Wavelength Bin to be assigned"
        assert (
            "Wavelength Center" in df.columns
        ), "Compute Time Variability requires Wavelength Center to be assigned"
        assert (
            "Wavelength Width" in df.columns
        ), "Compute Time Variability requires Wavelength Width to be assigned"
        assert (
            "Wavelength Width" in df.columns
        ), "Compute Time Variability requires Wavelength Width in the data"
        assert (
            "time_bin" in df.columns
        ), "Every binned dataset must have a Time Bin column"

    # Prepare the list of chunk observations
    variability_list = []
    assert len(time_bin_widths) == len(
        binned_datasets
    ), "There needs to be one binned dataset for each time-bin widths"

    observations = 0
    N = 50  # Create 500 random chunk observations per dataset
    # While we want more observations
    while observations < pp.chunk_counts:
        print(
            f" Chunk Phase Sampler (CPV)| {meta.id} : Accumulated {observations} ({(observations/pp.chunk_counts)*100:.0f}%) using Monte Carlo method"
        )
        # Select a random time-bin-width among those used to cut the dataset
        tbw = np.random.randint(0, len(time_bin_widths))
        time_bin_width = time_bin_widths[tbw]
        # This corresponds to the dataset at the tbw-th position
        df = binned_datasets[tbw]
        # Extract N chunks from this dataset using the monte_carlo method
        variability = monte_carlo_chunk_variability(df, pp, time_bin_width, N)
        # (Kludge to avoid even more kludgy table initialization)
        variability_list.append(variability)

        observations += len(variability)
    chunk_meta = ChunkVariabilityMetadata(
        f"{meta.id}_{pp.id}", meta.id, pp.id, meta.get_hash(), pp.get_hash()
    )
    variability_dataset = pd.concat(variability_list)

    variability_dataset.sort_index(axis=0, inplace=True)
    return variability_dataset, chunk_meta


def plot_obi_van_hist(
    variability, pp: ProcessingParameters, filename, use_log_scale=True, show=True
):
    filename = filename.replace(" ", "_")
    print(f"plot_obi_van_hist() {filename}")
    # Assume df has columns 'Photon Wavelength' and 'Time Bin Width'
    plt.figure(figsize=(10, 6))

    sns.histplot(
        data=variability, x="Wavelength Center", y="Time Bin Width", bins=50, cbar=True
    )
    plt.xlabel("Photon Wavelength (nm)")
    plt.ylabel("Winning Time Bin (s)")

    # Add title and grid
    plt.title(
        "OBI_VAN Winning time bins plot",
        fontsize=16,
    )
    plt.legend(fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Customize tick marks for better legibility
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Save and show
    plt.savefig(f"./plots/{filename}", dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_obi_van(
    variability, pp: ProcessingParameters, filename, use_log_scale=True, show=True
):
    # Ensure valid log values
    filename = filename.replace(" ", "_")
    print(f"plot_obi_van() {filename}")
    x = variability["Wavelength Center"]
    if use_log_scale:
        y = np.log(variability["Time Bin Width"])
    else:
        y = variability["Time Bin Width"]
    """
    intercept_error = ""
    try:
        intercept_error = False
        # Fit a power law to determine the exponent n
        idx = np.isfinite(x) & np.isfinite(y)  # Mask finite values
        n, intercept = np.polyfit(x[idx], y[idx], 1)
    except Exception as e:
        intercept_error = repr(e)
        print(intercept_error)
        intercept = 1
        n = 1
    """
    # Plot the variability vs. wavelength
    plt.figure(figsize=(10, 7))
    plt.scatter(
        x,
        y,
        label="OBI-VAN",
        alpha=0.8,
    )
    # Plot the power law fit
    """
    plt.plot(
        variability["Wavelength Center"],
        np.exp(intercept) * variability["Wavelength Center"] ** n,
        color="red",
        linestyle="--",
        label=f"Fit: V(λ) ∝ λ^{n:.2f}",
    )
    print(f"Fit: V(λ) ∝ λ^{n:.2f}")
    """
    # Adjust the axes
    if use_log_scale:
        plt.xscale("log")
        plt.yscale("log")

    plt.xlabel("Photon Wavelength (nm)", fontsize=14)
    if use_log_scale:
        plt.ylabel("Var maximizing time bin [log scale] (seconds)", fontsize=14)
    else:
        plt.ylabel("Var maximizing time bin (seconds)", fontsize=14)

    # Add title and grid
    plt.title(
        "OBI_VAN Variability plot",
        fontsize=16,
    )
    plt.legend(fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Customize tick marks for better legibility
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Save and show
    plt.savefig(f"./plots/{filename}", dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_flux_excess_variability(
    variability, pp: ProcessingParameters, filename, show=True
):
    filename = filename.replace(" ", "_")
    print(f"plot_flux_excess_variability() {filename}")
    # Ensure valid log values
    x = np.log(variability["Wavelength Center"])

    assert (
        pp.variability_type in variability.columns
    ), f"Variability type '{pp.variability_type}' not found in variability DataFrame columns"
    y = np.log(variability[pp.variability_type])

    intercept_error = ""

    try:
        intercept_error = False
        # Fit a power law to determine the exponent n
        idx = np.isfinite(x) & np.isfinite(y)  # Mask finite values
        n, intercept = np.polyfit(x[idx], y[idx], 1)
    except Exception as e:
        intercept_error = repr(e)
        print(intercept_error)
        intercept = 1
        n = 1

    # Plot the variability vs. wavelength
    plt.figure(figsize=(10, 7))
    plt.scatter(
        variability["Wavelength Center"],
        variability[pp.variability_type],
        color="blue",
        label=pp.variability_type,
        alpha=0.8,
    )
    # Plot the power law fit
    plt.plot(
        variability["Wavelength Center"],
        np.exp(intercept) * variability["Wavelength Center"] ** n,
        color="red",
        linestyle="--",
        label=f"Fit: V(λ) ∝ λ^{n:.2f}",
    )
    print(f"Fit: V(λ) ∝ λ^{n:.2f}")

    # Adjust the axes
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Photon Wavelength (nm) [Log Scale]", fontsize=14)
    plt.ylabel("Excess Variability [Log Scale]", fontsize=14)

    # Add title and grid
    plt.title(
        f"Flux Variability t:{pp.time_bin_seconds:.0f}s",
        fontsize=16,
    )
    plt.legend(fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Customize tick marks for better legibility
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Save and show
    plt.savefig(f"./plots/{filename}", dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_flux_residual_variability_linear(
    variability, pp: ProcessingParameters, filename, show=False, handle: str = ""
):
    """
    Plot the Excess Variability against wavelength on a linear scale.
    """
    filename = filename.replace(" ", "_")
    print(f"plot_flux_residual_variability_linear() {filename}")
    # Extract linear x and y values
    x = variability["Wavelength Center"]

    assert (
        pp.variability_type in variability.columns
    ), f"Variability type '{pp.variability_type}' not found in variability DataFrame columns"
    y = variability[pp.variability_type]

    intercept_error = ""

    try:
        intercept_error = False
        # Fit a power law to determine the exponent n
        idx = np.isfinite(x) & np.isfinite(y)  # Mask finite values
        n, intercept = np.polyfit(x[idx], y[idx], 1)
    except Exception as e:
        intercept_error = repr(e)
        print(intercept_error)

    # Plot the variability vs. wavelength
    plt.figure(figsize=(10, 7))
    plt.scatter(
        x,
        y,
        color="blue",
        label=pp.variability_type,
        alpha=0.8,
    )
    # Plot the power-law fit in linear space

    # Adjust the axes (linear scale)
    plt.xlabel("Photon Wavelength (nm)", fontsize=14)
    plt.ylabel(pp.variability_type, fontsize=14)

    # Add title and grid
    plt.title(
        f"Flux Variability (Linear Scale) t{pp.time_bin_seconds:.0f}s {handle}",
        fontsize=16,
    )
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5)

    # Customize tick marks for better legibility
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Save and show
    plt.savefig(f"./plots/{filename}", dpi=300)
    if show:
        plt.show()
    plt.close()


# Planck's law function
def planck_law(wavelength, T, scale):
    h = 1  # Planck's constant (J·s)
    c = 1  # Speed of light (m/s)
    k_B = 1  # Boltzmann constant (J/K)
    wavelength_m = wavelength * 1e-9  # Convert nm to meters
    return (
        scale
        * (2 * h * c**2 / wavelength_m**5)
        / (np.exp(h * c / (wavelength_m * k_B * T)) - 1)
    )


def supression_fit(wavelength: float, A_0: float, alpha: float, delta_t: float):
    """ "
    Suppression term of the  model used in flickering model
    """

    # Suppression factor for variability

    tau_lambda = A_0 * np.exp(-alpha * wavelength)
    suppression = delta_t / (delta_t + tau_lambda)

    # Total variability
    return suppression


def supressed_power_law(
    wavelength: float,
    A_0: float,
    epsilon: float,
    tau_0: float,
    alpha: float,
    delta_t: float,
):
    """
    Flickering model for variability as a function of wavelength, ensuring:
    - Power-law scaling for variability (matching simple power-law fit)
    - Suppression at short wavelengths due to prolonged dark-zoning.

    Parameters:
    - wavelength: array-like, wavelengths in nm
    - A_0: Amplitude scaling factor
    - epsilon: Power-law exponent controlling variability scaling
    - tau_0: Base flickering period at the longest wavelength
    - alpha: Growth rate of flickering period as wavelength decreases
    - delta_t: Time bin size of the observation

    Returns:
    - Variability (sigma_obs) as a function of wavelength
    """
    # Power-law amplitude scaling (ensures direct comparability with power-law fit)
    amp_term = A_0 * wavelength ** (-epsilon)

    # Total variability
    return amp_term


def flickering_model(
    wavelength: float,
    A_0: float,
    epsilon: float,
    tau_0: float,
    alpha: float,
    delta_t: float,
):
    """
    Flickering model for variability as a function of wavelength, ensuring:
    - Power-law scaling for variability (matching simple power-law fit)
    - Suppression at short wavelengths due to prolonged dark-zoning.

    Parameters:
    - wavelength: array-like, wavelengths in nm
    - A_0: Amplitude scaling factor
    - epsilon: Power-law exponent controlling variability scaling
    - tau_0: Base flickering period at the longest wavelength
    - alpha: Growth rate of flickering period as wavelength decreases
    - delta_t: Time bin size of the observation

    Returns:
    - Variability (sigma_obs) as a function of wavelength
    """
    # Power-law amplitude scaling (ensures direct comparability with power-law fit)
    amp_term = A_0 * wavelength ** (-epsilon)

    # Flickering period (slower for shorter wavelengths)
    tau_lambda = tau_0 * np.exp(-alpha * wavelength)

    # Adjust suppression term for numerical stability
    suppression = (delta_t / (delta_t + tau_lambda)) ** 2

    # Total variability
    return amp_term * suppression


def flickering_model_old(
    wavelength: float,
    A_0: float,
    epsilon: float,
    tau_0: float,
    alpha: float,
    delta_t: int,
):
    """
    Flickering model for variability as a function of wavelength, updated for:
    - Faster flickering at long wavelengths.
    - Suppression at short wavelengths due to prolonged dark-zoning.

    Parameters:
    - wavelength: array-like, wavelengths in nm
    - A_0: Amplitude scaling factor
    - beta: Decay rate of variability amplitude
    - tau_0: Base flickering period at the longest wavelength
    - alpha: Growth rate of flickering period as wavelength decreases
    - delta_t: Time bin size of the observation

    Returns:
    - Variability (sigma_obs) as a function of wavelength
    """
    base_wavelength = 1
    # Exponential amplitude decay
    amp_term = A_0 * np.exp(-epsilon * wavelength / base_wavelength)

    # Flickering period (slower for shorter wavelengths)
    tau_lambda = tau_0 * np.exp(-alpha * wavelength)

    # Suppression factor for variability
    suppression = delta_t / (delta_t + tau_lambda)

    # Total variability
    return amp_term * suppression


def simple_power_law(x, A, n):
    """Power-law function y = A * x^n"""
    return A * x**n


def plot_broken_power_law(
    variability: DataFrame,
    pp: ProcessingParameters,
    filename: str,
    show: bool = True,
    use_smoothed_PDS: bool = True,
    use_three: bool = False,
    handle: str = "",
):
    filename = filename.replace(" ", "_")
    """
    Plot the Excess Variability against wavelength and fit three power laws for different wavelength ranges,
    fitting in linear space instead of log space.
    """
    # Extract wavelength and variability values
    x = variability["Wavelength Center"].reset_index(drop=True)
    assert (
        pp.variability_type in variability.columns
    ), f"Variability type '{pp.variability_type}' not found in variability DataFrame columns"

    title = f"{pp.variability_type} t:{pp.time_bin_seconds:.0f}s {handle}"
    y = variability[pp.variability_type].reset_index(drop=True)
    label = pp.variability_type
    labelShort = f"{pp.variability_type}"

    # Filter out zero or NaN values
    valid_mask = (y != 0) & (~y.isna())
    x = x[valid_mask]
    y = y[valid_mask]

    # Define the wavelength ranges for the three power laws
    if use_three:
        ranges = {
            "< 0.35 nm": x <= 0.34,
            "0.34 - 1.5 nm": (x >= 0.33) & (x <= 1.5),
            "> 1.5 nm": (x >= 1.5),
        }
    else:
        ranges = {
            "< 0.35 nm": x <= 0.34,
            "0.33 - 1.2 nm": (x >= 0.33) & (x <= 1.3),
        }

    # Initialize the plot
    plt.figure(figsize=(10, 7))
    plt.scatter(
        x,
        y,
        color="blue",
        label=label,
        alpha=0.8,
    )

    # Fit and plot each power law
    for label, condition in ranges.items():
        x_subset = x[condition]
        y_subset = y[condition]

        if len(x_subset) > 1:  # Ensure enough points to fit
            try:
                # Fit power-law model directly in linear space
                popt, pcov = curve_fit(
                    simple_power_law, x_subset, y_subset, p0=[1e-3, -2]
                )

                A_0_fit, epsilon_fit = popt

                # Generate fitted curve
                x_fit = np.linspace(min(x_subset), max(x_subset), 100)
                y_fit = simple_power_law(x_fit, A_0_fit, epsilon_fit)

                # Plot the power-law fit
                plt.plot(
                    x_fit,
                    y_fit,
                    linestyle="--",
                    label=f"{label}: V(λ) ∝ λ^{epsilon_fit:.2f}",
                )
                print(f"{label}: V(λ) ∝ λ^{epsilon_fit:.2f}")

            except RuntimeError:
                print(f"Fit failed for range {label}")

    # Set axis labels and title
    plt.xlabel("Wavelength (nm)", fontsize=14)
    plt.ylabel(labelShort, fontsize=14)
    plt.title(
        title,
        fontsize=16,
    )
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5)

    # Save and show the plot
    plt.savefig(f"./plots/{filename}", dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_top_variability_timescales(
    variability,
    pp,
    filename: str,
    variability_type: Optional[str] = None,
    show: bool = False,
    fit_curve: bool = False,
    handle: str = "",
    ylim=None,
    xlim=None,
    heatmap_mode: str = "density",  # "stat" (median metric) or "density"
    gridsize: int = 60,  # hexbin resolution (increase for finer grid)
    reduce_stat: str = "median",  # "median" | "mean" (only for heatmap_mode="stat")
    mincnt: int = 3,  # suppress bins with too few points
):
    filename = filename.replace(" ", "_")
    """
    Plot timescale sampling using a 2D hexbin heatmap instead of a dense scatter.

    X-axis: Wavelength Center
    Y-axis: Time Bin Width
    Color : (a) median or mean of `variability_type` per bin (heatmap_mode="stat"), or
            (b) log-count density (heatmap_mode="density")

    Notes:
    - Set `gridsize` larger for more detail; use `mincnt` to hide noisy bins.
    - If you still want to overlay a fitted curve, set `fit_curve=True` (kept as before).
    """
    if variability_type is None:
        variability_type = pp.variability_type

    required = {"Wavelength Center", "Time Bin Width", variability_type, "Mean Count"}
    missing = required.difference(set(variability.columns))
    assert not missing, f"Missing required columns: {missing}"

    x = variability["Wavelength Center"].to_numpy()
    y = variability["Time Bin Width"].to_numpy()
    c = variability[variability_type].to_numpy()
    minc = np.min(c)
    maxc = np.max(c)
    title = f"{handle} : timescale sampling"

    plt.figure(figsize=(9, 6))

    if heatmap_mode == "density":
        # Plain density map (counts per hex), log-scaled
        hb = plt.hexbin(
            x,
            y,
            gridsize=gridsize,
            cmap="viridis",
            bins=None,
            mincnt=mincnt,
            xscale="linear",
            yscale="linear",
            vmin=minc,
            vmax=maxc,
            C=None,
            reduce_C_function=None,
        )
        cbar = plt.colorbar(hb)
        cbar.set_label("Count per bin (log)")

    elif heatmap_mode == "stat":
        # Aggregate the variability metric per hex (median/mean)
        if reduce_stat == "median":
            reducer = np.nanmedian
        elif reduce_stat == "mean":
            reducer = np.nanmean
        else:
            raise ValueError("reduce_stat must be 'median' or 'mean'")

        hb = plt.hexbin(
            x,
            y,
            C=c,
            reduce_C_function=reducer,
            gridsize=gridsize,
            cmap="viridis",
            mincnt=mincnt,  # require >= mincnt samples per hex
        )
        cbar = plt.colorbar(hb)
        cbar.set_label(f"{variability_type} ({reduce_stat} per bin)")

    else:
        raise ValueError("heatmap_mode must be 'density' or 'stat'")

    # Optional overlays (kept minimal; your old fit code can be slotted here if needed)
    if fit_curve:
        pass  # (You can overlay your model curve on top of the hexbin if desired.)

    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Time bin width (s)")
    plt.title(title)
    plt.grid(False)  # grids look odd on hexbins; turn off by default
    plt.tight_layout()
    plt.savefig(f"./plots/{filename}", dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_wavelength_counts_histogram(
    event_data_A: DataFrame,
    event_data_B: DataFrame,
    column: str,
    title: str,
    pp: ProcessingParameters,
    filename: str,
    show: bool = False,
    A_handle: str = "A",
    B_handle: str = "B",
    perc: float = None,
    saveplot: bool = False,
):
    try:
        filename = filename.replace(" ", "_")
        print(f"plot_wavelength_counts_histogram() : Filename {filename}")

        assert column in event_data_A.columns, f"Missing column {column} in A"
        assert column in event_data_B.columns, f"Missing column {column} in B"
        assert "Wavelength Center" in event_data_B.columns, "Missing Wavelength Center"

        print(
            f"Event data A and B needs to have similar counts. A:{event_data_A.shape[0]}, B:{event_data_B.shape[0]}"
        )
        assert (
            abs(1 - event_data_A.shape[0] / event_data_B.shape[0]) < 0.01
        ), "Event data A and B needs to have similar counts, now too different "

        label_x = "Wavelength center"
        label_y = "Counts"

        # Plot histogram
        plt.figure(figsize=(8, 6))
        counts, bins, _ = plt.hist(
            [event_data_A[column], event_data_B[column]],
            bins=pp.wavelength_bins,
            color=["black", "grey"],
            alpha=0.7,
            label=[A_handle, B_handle],
        )

        counts_A = counts[0]
        counts_B = counts[1]

        # Declare these so they exist even when curve fitting fails
        a_A = None
        b_A = None
        a_B = None
        b_B = None
        e_a_ = None
        se_b_A = None
        se_a_B = None
        se_b_B = None
        p = None
        significance = None
        # Use bin centers for fitting
        bin_centers_A = 0.5 * (bins[1:] + bins[:-1])
        bin_centers_B = 0.5 * (bins[1:] + bins[:-1])
        print("-- Curve fitting")
        # Fit a power law to histogram A
        try:
            popt_A, pcov_A = curve_fit(
                simple_power_law,
                bin_centers_A[counts_A > 0],  # avoid log(0)
                counts_A[counts_A > 0],
                p0=[1e3, -2],
            )
            print("-- Curve fitting done")
            a_A, b_A = popt_A
            var_a_A, var_b_A = np.diag(pcov_A)  # extract variances
            se_a_A, se_b_A = np.sqrt(var_a_A), np.sqrt(var_b_A)

            y_fit_A = simple_power_law(bin_centers_A, a_A, b_A)

            # Plot the fitted curve
            plt.plot(
                bin_centers_A,
                y_fit_A,
                linestyle="--",
                color="red",
                label=f"Power law A: a={a_A:.2e}, b={b_A:.2f}",
            )

        except Exception as e:
            print(f"Exception when fitting curve, recovering.", e)

        try:
            popt_B, pcov_B = curve_fit(
                simple_power_law,
                bin_centers_B[counts_B > 0],  # avoid log(0)
                counts_B[counts_B > 0],
                p0=[1e3, -2],
            )

            a_B, b_B = popt_B
            var_a_B, var_b_B = np.diag(pcov_B)  # extract variances
            se_a_B, se_b_B = np.sqrt(var_a_A), np.sqrt(var_b_A)

            y_fit_B = simple_power_law(bin_centers_B, a_B, b_B)

            # The difference in curves
            delta_b = b_A - b_B

            # The standard error for the delta is the square root of the summed squares
            se_delta = np.sqrt(se_b_A**2 + se_b_B**2)
            z = delta_b / se_delta
            p = 2 * (1 - norm.cdf(abs(z)))  # two-sided test
            significance = f"Top {perc}% obs. Δb = {delta_b:.3f}, p = {p:.3g}"

            # Plot the fitted curve
            plt.plot(
                bin_centers_A,
                y_fit_B,
                linestyle="--",
                color="blue",
                label=f"Power law B: a={a_B:.2e}, b={b_B:.2f}",
            )
        except Exception as e:
            print(f"Problem fitting curve, aborting due to {repr(e)}")

        plt.title(significance, fontsize=10, y=0.95)
        plt.suptitle(title, y=1.05, fontsize=18)
        plt.text(0.5, 0.1, significance, horizontalalignment="center", fontsize=10)
        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.title(f"{title}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if saveplot:
            plt.savefig(f"./plots/{filename}", dpi=300)
        if show:
            plt.show()
        plt.close()

        return a_A, b_A, a_B, b_B, se_a_A, se_b_A, se_a_B, se_b_B, p

    except Exception as e:
        print(f"Plotting failed: {repr(e)}")
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def plot_chunk_variability_shape(
    variabilityA: DataFrame,
    variabilityB: DataFrame,
    pp: ProcessingParameters,
    filename: str,
    variability_type: str = None,
    show: bool = False,
    handleA: str = "",
    handleB: str = "",
):
    """
    Plot a histogram of the chunk-level variability metric and fit normal distribution
    """
    try:
        filename = filename.replace(" ", "_")
        print(f"plot_chunk_variability_shape() {filename}")
        if variability_type is None:
            variability_type = pp.variability_type
        assert (
            variability_type in variabilityA.columns
        ), f"Variability type '{variability_type}' not found in variability DataFrame columns"

        wavelengths = variabilityA["Wavelength Center"]
        obsA = variabilityA[variability_type]
        obsB = variabilityB[variability_type]

        label_x = f"{variability_type}"
        label_y = "Counts"
        title = f"{variability_type} shape"

        plt.figure(figsize=(8, 6))
        plt.hist(
            [obsA, obsB],
            bins=100,
            color=["black", "grey"],
            alpha=0.7,
            label=[f"A : {handleA}", f"B : {handleB}"],
        )

        plt.xlabel(label_x)
        plt.ylabel(label_y)
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./plots/{filename}", dpi=300)
        if show:
            plt.show()
        plt.close()
    except Exception as e:
        print(f"Plotting chunk variability shape failed: {e}")
        raise e


def plot_chunk_variability_excess(
    variability: DataFrame,
    pp: ProcessingParameters,
    filename: str,
    variability_type: str = None,
    show: bool = False,
    handle: str = "",
    ylim=None,
):
    """
    Plot flickering variability model and fit to excess variability data.
    """
    filename = filename.replace(" ", "_")
    print(f"plot_chunk_variability_excess:  {filename}")
    if variability_type is None:
        variability_type = pp.variability_type
    assert (
        variability_type in variability.columns
    ), f"Variability type '{variability_type}' not found in variability DataFrame columns"

    wavelengths = variability["Wavelength Center"]

    y_raw = variability[variability_type]

    label_y = f"{variability_type}"
    title = f"{label_y}  {handle}"

    plt.figure(figsize=(8, 6))
    plt.scatter(wavelengths, y_raw, color="black", label=variability_type)
    if ylim is not None:
        plt.ylim(ylim)

    if False:
        try:
            # Model & Initial Guess
            # use the simple 2-param power_law(x, A, n) defined below
            model = simple_power_law

            # prepare numeric arrays and handle NaNs
            wavelengths_arr = np.asarray(wavelengths, dtype=float)
            y_raw_arr = np.asarray(y_raw, dtype=float)

            # Shift to positive values for fitting (use nan-aware min)
            if np.any(np.isfinite(y_raw_arr)):
                mn = np.nanmin(y_raw_arr)
                B = -mn + 1e-6 if mn < 0 else 0.0
            else:
                B = 0.0
            y_shifted = y_raw_arr + B

            # sensible initial guess: amplitude ~ median, exponent negative expected
            A0_guess = float(
                np.nanmedian(y_shifted[np.isfinite(y_shifted)])
                if np.any(np.isfinite(y_shifted))
                else 1.0
            )
            eps_guess = -1.0
            initial_guess = [max(1e-8, A0_guess), eps_guess]

            # allow exponent to be negative; keep amplitude >= 0
            lower_bounds = [0.0, -np.inf]
            upper_bounds = [np.inf, np.inf]
            print("Last line before curve_fit!")
            # Fit model to data (convert NaNs to numeric; curve_fit needs arrays)
            popt, pcov = curve_fit(
                model,
                wavelengths_arr,
                np.nan_to_num(y_shifted),
                p0=initial_guess,
                bounds=(lower_bounds, upper_bounds),
                maxfev=10000,
            )

            A_0, epsilon = popt

            print("Fitted Parameters:")
            print(f"A_0: {A_0:.6g}, ε: {epsilon:.6g}")

            # Reconstruct model curve and unshift
            wavelengths_fit = np.linspace(
                np.nanmin(wavelengths_arr), np.nanmax(wavelengths_arr), 500
            )
            y_fit_shifted = model(wavelengths_fit, *popt)
            y_fit = y_fit_shifted - B

            # Plot fitted model
            plt.plot(
                wavelengths_fit,
                y_fit,
                "r--",
                label=f"Fit: $A$={A_0:.3g}, $n$={epsilon:.3g}",
            )

        except RuntimeError as e:
            print(f"Curve fitting failed: {e}")
            plt.title(f"{title} [fit failed]")

    # Plot layout
    plt.xlabel("Wavelength (nm)")
    plt.ylabel(label_y)
    plt.title(title)
    plt.grid(True)
    plt.legend()

    plt.savefig(f"./plots/{filename}", dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_expected_flux(
    variability: DataFrame,
    pp: ProcessingParameters,
    filename: str,
    show: bool = True,
    wavelength_range=None,
    handle: str = "",
):
    print(f"plot_expected_flux() : {filename}")
    xaxis = variability["Wavelength Center"]
    pp.variability_type in variability.columns, f"Metric type '{pp.variability_type}' not found in variability DataFrame columns"
    y = variability[pp.variability_type]

    if wavelength_range is not None:
        selected_idx = (variability["Wavelength Center"] >= wavelength_range[0]) & (
            variability["Wavelength Center"] < wavelength_range[1]
        )

        # Apply filtering
        xaxis = variability.loc[selected_idx, "Wavelength Center"]
        y = variability.loc[selected_idx, pp.variability_type]

    # Plot Excess Variability vs wavelength
    plt.figure(figsize=(10, 6))
    # print(tabulate(variability.head(200)))
    plt.plot(
        xaxis,
        y,
        label="Expected photon flux",
        color="blue",
        marker="o",
    )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel(f"Metric: {pp.variability_type}")
    plt.title(f"Smoothed Power Density function t:{pp.time_bin_seconds:.0f}s")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./plots/{filename}", dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_mean_count(
    variability: DataFrame,
    pp: ProcessingParameters,
    filename: str,
    show: bool = True,
    wavelength_range=None,
    handle: str = "",
):
    print(f"plot_mean_count() : {filename}")
    xaxis = variability["Wavelength Center"]
    assert (
        pp.variability_type in variability.columns
    ), f"Metric type '{pp.variability_type}' not found in variability DataFrame columns"
    yaxis = variability[pp.variability_type]

    if wavelength_range is not None:
        selected_idx = (variability["Wavelength Center"] >= wavelength_range[0]) & (
            variability["Wavelength Center"] < wavelength_range[1]
        )

        # Apply filtering
        xaxis = variability.loc[selected_idx, "Wavelength Center"]
        yaxis = variability.loc[selected_idx, pp.variability_type]
    # Plot Excess Variability vs wavelength
    plt.figure(figsize=(10, 6))
    plt.plot(
        xaxis,
        yaxis,
        label=f"Metric : {pp.variability_type}",
        color="blue",
        marker="o",
    )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel(f"Metric: {pp.variability_type}")
    plt.title(f"{pp.variability_type} for wavelength t:{pp.time_bin_seconds:.0f}s")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./plots/{filename}", dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_mimic(
    meta: FitsMetadata,
    source: DataFrame,
    filename: str,
    show: bool = True,
    handle: str = "",
):
    print(f"plot_mimic() : {filename}")
    # Extract wavelength and count
    wavelength = source["Wavelength Center"]

    fluxpersec = source["Mean Count"] / (
        source["Time Bin Width"] * source["Wavelength Width"]
    )

    # pds = PowerDensitySpectrum(meta.spectrum, "wavelength")

    rho_empirical = exp_count_per_sec(
        wavelength.values,
        meta.apparent_spectrum.A,
        meta.apparent_spectrum.lambda_0,
        meta.apparent_spectrum.sigma,
        meta.apparent_spectrum.C,
    )

    # Plot both empirical and synthetic
    plt.figure(figsize=(10, 6))
    plt.plot(wavelength, fluxpersec, label="Observed", color="orange", marker="o")
    plt.plot(
        wavelength,
        rho_empirical,
        label="Empirical Mimic",
        linestyle="--",
        color="steelblue",
    )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Photon flux per sec")
    plt.title("Mean bin count for wavelength")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./plots/{filename}", dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_shot_noise_variability(
    variability: DataFrame,
    pp: ProcessingParameters,
    filename: str,
    show: bool = True,
    wavelength_range=None,
    handle: str = "",
):
    # Plot Excess Variability vs wavelength
    print(f"plot_shot_noise_variability : {filename}")
    xaxis = variability["Wavelength Center"]
    if pp.variability_type not in variability.columns:
        raise ValueError(
            f"Variability type '{pp.variability_type}' not found in variability DataFrame columns"
        )
    yaxis = variability[pp.variability_type]

    if wavelength_range is not None:
        selected_idx = (variability["Wavelength Center"] >= wavelength_range[0]) & (
            variability["Wavelength Center"] < wavelength_range[1]
        )

        # Apply filtering
        xaxis = variability.loc[selected_idx, "Wavelength Center"]
        yaxis = variability.loc[selected_idx, pp.variability_type]

    plt.figure(figsize=(10, 6))
    plt.plot(
        xaxis,
        yaxis,
        label=pp.variability_type,
        color="green",
        marker="o",
    )
    plt.xlabel("Wavelength (nm)")
    plt.ylabel(pp.variability_type)
    plt.title(f"{pp.variability_type} vs Wavelength t:{pp.time_bin_seconds:.0f}s")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./plots/{filename}", dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_excess_variability_smoothed(
    variability: DataFrame,
    pp: ProcessingParameters,
    filename: str,
    show=False,
    handle: str = "",
):
    print(f"plot_excess_variability_smoothed : {filename}")
    if pp.variability_type not in variability.columns:
        raise ValueError(
            f"Metric type '{pp.variability_type}' not found in variability DataFrame columns"
        )
    # Plot Excess Variability vs wavelength
    plt.figure(figsize=(10, 6))
    plt.scatter(
        variability["Wavelength Center"],
        variability[pp.variability_type],
        label=f"{pp.variability_type}",
        color="green",
        marker="o",
    )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel(f"Metric: {pp.variability_type}")
    plt.title(f"{pp.variability_type} vs Wavelength t:{pp.time_bin_seconds:.0f}s")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./plots/{filename}", dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_excess_variability(
    variability: DataFrame,
    pp: ProcessingParameters,
    filename: str,
    show=False,
    handle: str = "",
):
    # Plot Excess Variability vs wavelength
    plt.figure(figsize=(10, 6))
    assert (
        pp.variability_type in variability.columns
    ), f"Variability type '{pp.variability_type}' not found in variability DataFrame columns"
    plt.scatter(
        variability["Wavelength Center"],
        variability[pp.variability_type],
        label=f"{pp.variability_type}",
        color="green",
        marker="o",
    )
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("" f"Metric: {pp.variability_type}")
    plt.title(f"{pp.variability_type} vs Wavelength t:{pp.time_bin_seconds:.0f}s")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./plots/{filename}", dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_total_variability(
    variability,
    pp: ProcessingParameters,
    filename,
    show=False,
    wavelength_range=None,
    handle: str = "",
):
    # Plot Excess Variability vs wavelength
    # Plot Excess Variability vs wavelength
    xaxis = variability["Wavelength Center"]

    assert (
        pp.variability_type in variability.columns
    ), f"Variability type '{pp.variability_type}' not found in variability DataFrame columns"
    yaxis = variability[pp.variability_type]

    if wavelength_range is not None:
        selected_idx = (variability["Wavelength Center"] >= wavelength_range[0]) & (
            variability["Wavelength Center"] < wavelength_range[1]
        )

        # Apply filtering
        xaxis = variability.loc[selected_idx, "Wavelength Center"]
        yaxis = variability[pp.variability_type]

    plt.figure(figsize=(10, 6))
    plt.scatter(
        xaxis,
        yaxis,
        label="Total Variability",
        color="green",
        marker="o",
    )
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Std counts across time")
    plt.title(f"Total Variability vs Wavelength t:{pp.time_bin_seconds:.0f}s {handle}")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./plots/{filename}", dpi=300)
    if show:
        plt.show()
    plt.close()
