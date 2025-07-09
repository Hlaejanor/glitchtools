import pandas as pd
from scipy.optimize import curve_fit
from common.helper import get_duration, ensure_plot_folder_exists
from common.fitsmetadata import FitsMetadata, ProcessingParameters
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from common.powerdensityspectrum import (
    PowerDensitySpectrum,
    Spectrum,
    exp_count_per_sec,
)
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


def plot_spectrum_vs_data(
    meta_A: FitsMetadata,
    binned_data_A: DataFrame,
    processing_params: ProcessingParameters,
    filename: str,
    meta_B: FitsMetadata = None,
    binned_data_B: DataFrame = None,
    show=False,
):
    """
    kind: 'counts_per_second' (default) or 'total_counts'
    show_bin_widths: if True, plots steps rather than dots
    logy: if True, y-axis will be log-scaled
    """

    durationA = get_duration(meta_A, processing_params)
    ensure_plot_folder_exists(meta_A)

    # Extract wavelength and count

    # Group by wavelength bin and count
    grouped_data_A = (
        binned_data_A.groupby(["Wavelength Center", "Wavelength Width"])
        .size()
        .reset_index(name="Count")
    )
    # wavelength_A = grouped_data_A["Wavelength Center"]
    grouped_data_A["Flux per sec"] = grouped_data_A["Count"] / (
        durationA * grouped_data_A["Wavelength Width"]
    )

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
        label=f"Parametric {meta_A.star} id: {meta_A.id}",
    )

    # Plot both empirical and synthetic
    plt.plot(
        grouped_data_A["Wavelength Center"],
        grouped_data_A["Flux per sec"],
        label=f"Observed {meta_A.star} id: {meta_A.id}",
    )
    if meta_B is not None:
        durationB = get_duration(meta_A, processing_params)

        grouped_data_B = (
            binned_data_B.groupby(["Wavelength Center", "Wavelength Width"])
            .size()
            .reset_index(name="Count")
        )
        wavelength_B = grouped_data_B["Wavelength Center"]
        grouped_data_B["Flux per sec"] = grouped_data_B["Count"] / (
            durationB * grouped_data_B["Wavelength Width"]
        )

        rho_empirical_B = exp_count_per_sec(
            grouped_data_B["Wavelength Center"],
            meta_A.apparent_spectrum.A,
            meta_A.apparent_spectrum.lambda_0,
            meta_A.apparent_spectrum.sigma,
            meta_A.apparent_spectrum.C,
        )

        plt.plot(
            grouped_data_B["Wavelength Center"],
            rho_empirical_B,
            label=f"Parametric {meta_B.star} id:{meta_B.id}",
        )

        # Plot both empirical and synthetic
        plt.plot(
            grouped_data_B["Wavelength Center"],
            grouped_data_B["Flux per sec"],
            label=f"Observed {meta_B.star} id:{meta_B.id}",
        )

    # pds = PowerDensitySpectrum(meta.spectrum, "wavelength")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Mean count per wavelength")
    if meta_B is not None:
        title = f"Mean bin count for wavelength {meta_A.id} and {meta_B.id}"
    else:
        title = f"Mean bin count for wavelength {meta_A.id}"
    plt.title(title)
    plt.grid(True)
    plt.legend()

    plt.savefig(f"./plots/{meta_A.id}/{filename}")
    if show:
        plt.show()
    plt.close()
    plt.close()


def plot_ccd_energy_map(data, dt, filename, show=True):
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
    hits = data[data["Hit"]].copy()

    # 2. Convert wavelength (nm) to photon energy (keV).
    #    E(eV) = 1239.84193 / λ(nm) -> E(keV) = [1239.84193 / λ(nm)] / 1000
    def nm_to_keV(nm):
        # Avoid divide-by-zero if nm can be 0
        return (1239.84193 / nm) / 1000.0 if nm > 0 else 0.0

    hits["Energy (keV)"] = hits["Wavelength (nm)"].apply(nm_to_keV)

    # 3. Determine the pixel grid extents.
    #    Convert X/Y to int if not already.
    hits["CCD X"] = hits["CCD X"].astype(int)
    hits["CCD Y"] = hits["CCD Y"].astype(int)

    min_x, max_x = hits["CCD X"].min(), hits["CCD X"].max()
    min_y, max_y = hits["CCD Y"].min(), hits["CCD Y"].max()

    width = max_x - min_x + 1
    height = max_y - min_y + 1

    # 4. Create a 2D array ("energy_map") to accumulate total energy in keV per pixel.
    energy_map = np.zeros((height, width), dtype=np.float64)

    # 5. Fill the energy_map
    for _, row in hits.iterrows():
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

    plt.savefig(f"./plots/{filename}", dpi=150)
    if show:
        plt.show()
    plt.close()


def plot_ccd_bin(data, filename, show=True):
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
    plt.savefig(f"./plots/{filename}")
    if show:
        plt.show()
    plt.close()


def compute_time_variability(source_data: DataFrame, duration: float) -> DataFrame:
    """
    Compute time-dependent flux variability per wavelength bin,
    corrected for shot noise. If mean_counts_per_pixel is provided, it normalizes

    Parameters
    ----------
    data : pd.DataFrame
        Must contain at least:
          - 'Time': arrival time or event time (float)
          - 'Wavelength (nm)': assigned wavelength bin for the photon
          - 'Wavelength Bin' :
          - 'Time Bin' :
          - 'Hit': indicator if this row is a valid photon
    time_bins : int or array-like
        If int, will create that many uniform bins from min to max of data.Time.
        If array-like, interprets as the actual bin edges.
    wavelength_bins : int or array-like
        Same logic as time_bins, but for 'Wavelength (nm)'.

    Returns
    -------
    pd.DataFrame with columns:
       [ 'Wavelength Bin',
         'Total Variability',
         'Shot Noise',
         'Excess Variability',
         'Wavelength Center' ]
    """

    assert (
        "Wavelength Bin" in source_data.columns
    ), "Compute Time Variability requires Wavelength Bin to be assigned"
    assert (
        "Wavelength Center" in source_data.columns
    ), "Compute Time Variability requires Wavelength Center to be assigned"
    assert (
        "Wavelength Width" in source_data.columns
    ), "Compute Time Variability requires Wavelength Width to be assigned"
    assert (
        "Time Bin" in source_data.columns
    ), "Compute Time Variability requires Time Bin to be assigned"
    assert (
        "Wavelength Width" in source_data.columns
    ), "Compute Time Variability requires Wavelength Width in the data"
    # 0. Drop
    # Create wavelength bins

    # 1. Ensure the 'Hit' column is numeric or boolean
    source_data["Hit"] = source_data["Hit"].astype(int)

    # 2. Count photons in each (time bin, wavelength bin)

    # SELECT 'Time Bin', 'Wavelength Bin', 'Wavelength Width', 'Wavelength Center', Count(*) as Counts
    # FROM source_data GROUP BY 'Time Bin', 'Wavelength Bin', 'Wavelength Width', 'Wavelength Center'
    # WHERE 'Hit' = 1

    # 1. Ensure the 'Hit' column is numeric
    source_data["Hit"] = source_data["Hit"].astype(int)

    # 2. Count photons in each (Time Bin, Wavelength Bin)
    grouped_counts = (
        source_data[source_data["Hit"] == 1]
        .groupby(
            ["Time Bin", "Wavelength Bin", "Wavelength Width", "Wavelength Center"]
        )
        .size()
        .reset_index(name="Counts")
    )

    grouped_counts = grouped_counts.sort_values(["Wavelength Bin", "Time Bin"])
    # SELECT DISTINCT Wavelength Bin
    wbins = grouped_counts["Wavelength Bin"].unique()
    # SELECT DISTINCT Time Bin
    all_time_bins = grouped_counts["Time Bin"].unique()

    # 3. Create a complete multi-index
    index_all = pd.MultiIndex.from_product(
        [all_time_bins, wbins], names=["Time Bin", "Wavelength Bin"]
    )

    # 4. Total count per wavelength bin
    total_counts = grouped_counts.groupby("Wavelength Bin")["Counts"].sum().to_dict()

    # 5. Wavelength Width per bin
    wavelength_widths = (
        grouped_counts.drop_duplicates("Wavelength Bin")
        .set_index("Wavelength Bin")["Wavelength Width"]
        .to_dict()
    )

    # 6. Wavelength Center per bin
    wavelength_centers = (
        grouped_counts.drop_duplicates("Wavelength Bin")
        .set_index("Wavelength Bin")["Wavelength Center"]
        .to_dict()
    )

    # 7. Reindex to fill all (time, wavelength) bin combos
    grouped_counts = (
        grouped_counts.set_index(["Time Bin", "Wavelength Bin"])
        .reindex(index_all, fill_value=0)
        .reset_index()
    )

    wavelength_to_time_bin_groups = (
        grouped_counts.groupby("Wavelength Bin")["Counts"].apply(list).to_dict()
    )

    # 8. Compute flux per second per wavelength bin (flux density)
    # SQL: SELECT (Total Counts * Width) / Duration AS flux_per_sec_per_nm FROM ...
    flux_per_sec_per_nm = {
        wbin: (total_counts[wbin] / duration) / wavelength_widths[wbin]
        for wbin in wbins
    }

    # 6. For each wavelength bin, compute:
    #      - The standard deviation of Counts across time => 'Total Variability'
    #      - Mean Counts in that bin => for shot noise calculation
    #      - Shot noise across time => sqrt of the average or sum of means?
    #        We'll do a typical approach: Var_total = Var_signal + Var_noise
    #        So we approximate Var_noise by the average of the mean (Poisson).
    #        Then Excess Variability = sqrt(Var_signal) i.e. sqrt(Var_total - Var_noise)
    # print(tabulate(mean_counts_per_pixel.tail(20)))

    variability_list = []
    for wbin in wbins:
        counts = wavelength_to_time_bin_groups[wbin]

        if len(counts) == 0:
            continue
        flux = flux_per_sec_per_nm[wbin]
        w_center = wavelength_centers[wbin]
        w_width = wavelength_widths[wbin]
        mean_cnt = np.mean(counts)
        total_var = np.std(counts, ddof=1)
        var_noise = mean_cnt  # Poisson shot noise

        excess_var = np.var(counts, ddof=1) - var_noise
        # Ensure non-negative excess variance
        if excess_var < 0:
            print(f"Less variability than expected!  {excess_var}")
            excess_std = -np.sqrt(np.abs(excess_var))
        else:
            excess_std = np.sqrt(excess_var)

        # Smoothed version (optional)
        excess_std_smoothed = excess_std / np.sqrt(mean_cnt) if mean_cnt > 0 else 0

        variability_list.append(
            {
                "Wavelength Bin": wbin,
                "Wavelength Center": w_center,
                "Wavelength Width": w_width,
                "Mean Count": mean_cnt,
                "Total Variability": total_var,
                "Shot Noise": np.sqrt(var_noise),
                "Excess Variability": excess_std,
                "Flux per sec": flux,
                "Excess Variability Smoothed": excess_std_smoothed,
                "Zero Count Bins": np.sum(counts == 0),
                "One Count Bins": np.sum(counts == 1),
            }
        )

    # Compute Excess Variability as a function of wavelength

    # Convert list to DataFrame
    variability_df = pd.DataFrame(variability_list)

    print("Time-dependent flux variability (per wavelength bin):")
    # print(tabulate(variability_df, headers="keys", tablefmt="grid", floatfmt=".3f"))
    # Diagnostic for subtraction
    for idx, row in variability_df.iterrows():
        total_var = row["Total Variability"]
        shot_noise = row["Shot Noise"]
        residual_var = row["Excess Variability"]

        print(
            f"Wavelength Bin: {idx}, Total Variability^2: {total_var**2:.3f}, "
            f"Shot Noise^2: {shot_noise**2:.3f}, Excess Variability^2: {residual_var**2:.3f} Perc:{100*(residual_var/total_var):.3f}%"
        )
    return variability_df


def plot_flux_excess_variability(variability, filename, dt, show=True):
    # Ensure valid log values
    x = np.log(variability["Wavelength Center"])
    y = np.log(variability["Excess Variability"])

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
        variability["Excess Variability"],
        color="blue",
        label="Non-Poissonian Excess Variability",
        alpha=0.8,
    )
    # Plot the power law fit
    plt.plot(
        variability["Wavelength Center"],
        np.exp(intercept) * variability["Wavelength Center"] ** n,
        color="red",
        linestyle="--",
        label=f"Fit: V(λ) ∝ λ^{n:.2f}"
    )
    print(f"Fit: V(λ) ∝ λ^{n:.2f}")

    # Adjust the axes
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Photon Wavelength (nm) [Log Scale]", fontsize=14)
    plt.ylabel("Excess Variability [Log Scale]", fontsize=14)

    # Add title and grid
    plt.title(
        f"Flux Variability t:{dt:.0f}s",
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


def plot_flux_residual_variability_linear(variability, filename, dt, show=True):
    """
    Plot the Excess Variability against wavelength on a linear scale.
    """
    # Extract linear x and y values
    x = variability["Wavelength Center"]
    y = variability["Excess Variability"]

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
        label="Non-Poissonian Excess Variability",
        alpha=0.8,
    )
    # Plot the power-law fit in linear space

    # Adjust the axes (linear scale)
    plt.xlabel("Photon Wavelength (nm)", fontsize=14)
    plt.ylabel("Excess Variability", fontsize=14)

    # Add title and grid
    plt.title(
        f"Flux Variability (Linear Scale) t{dt:.0f}s",
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


def supression_fit(wavelength, A_0, alpha, delta_t):
    """ "
    Suppression term of the  model used in flickering model
    """

    # Suppression factor for variability

    tau_lambda = A_0 * np.exp(-alpha * wavelength)
    suppression = delta_t / (delta_t + tau_lambda)

    # Total variability
    return suppression


def flickering_model(wavelength, A_0, epsilon, tau_0, alpha, delta_t):
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
    wavelength,
    A_0,
    epsilon,
    tau_0,
    alpha,
    delta_t,
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


def power_law(x, A, n):
    """Power-law function y = A * x^n"""
    return A * x**n


def plot_broken_power_law(
    variability, filename, dt, show=True, use_smoothed_PDS=True, use_three=False
):
    """
    Plot the Excess Variability against wavelength and fit three power laws for different wavelength ranges,
    fitting in linear space instead of log space.
    """
    # Extract wavelength and variability values
    x = variability["Wavelength Center"].reset_index(drop=True)

    if use_smoothed_PDS:
        title = f"Flux Variability - broken power-Law (smoothed) t:{dt:.0f}s"
        y = variability["Excess Variability Smoothed"].reset_index(drop=True)
        label = "Non-Poissonian Excess Variability (smoothed PDS)"
        labelShort = "Excess Variability (smoothed PDS)"
    else:
        title = f"Flux Variability - broken power-Law t:{dt:.0f}s"
        y = variability["Excess Variability"].reset_index(drop=True)
        label = "Non-Poissonian Excess Variability"
        labelShort = "Excess Variability"
    # Filter out zero or NaN values
    valid_mask = (y != 0) & (~y.isna())
    x = x[valid_mask]
    y = y[valid_mask]

    # Define the wavelength ranges for the three power laws
    if use_three:
        ranges = {
            "< 0.35 nm": x <= 0.34,
            "0.34 - 1.0 nm": (x >= 0.33) & (x <= 1.0),
            "> 1.0 nm": (x >= 1.0),
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
                popt, pcov = curve_fit(power_law, x_subset, y_subset, p0=[1e-3, -2])

                A_0_fit, epsilon_fit = popt

                # Generate fitted curve
                x_fit = np.linspace(min(x_subset), max(x_subset), 100)
                y_fit = power_law(x_fit, A_0_fit, epsilon_fit)

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


def plot_flickering(
    variability, filename, dt, show=True, use_old=True, use_normalized=True
):
    """
    Plot flickering variability model and fit to excess variability data.
    """
    wavelengths = variability["Wavelength Center"]
    y_raw = (
        variability["Excess Variability Smoothed"]
        if use_normalized
        else variability["Excess Variability"]
    )
    label_y = (
        "Excess Variability (smoothed PDS)" if use_normalized else "Excess Variability"
    )
    title = f"{label_y} t:({dt:.0f}s)"

    plt.figure(figsize=(8, 6))
    plt.scatter(wavelengths, y_raw, color="blue", label="Empirical data")

    try:
        # Model & Initial Guess
        model = flickering_model_old if use_old else flickering_model
        initial_guess = (
            [1.0, 0.1, 10.0, 50, 1.0] if use_old else [100.0, 3.0, 10.0, 48.35, 1.0]
        )

        # Shift to positive values for fitting
        B = -np.min(y_raw) + 1e-2 if np.min(y_raw) < 0 else 0.0
        y_shifted = y_raw + B

        # Fit model to data
        popt, pcov = curve_fit(
            model,
            wavelengths,
            np.nan_to_num(y_shifted),
            p0=initial_guess,
            bounds=([0, 0, -5, 0, 1], [np.inf] * 5),
        )

        # Extract and display parameters
        A_0, epsilon, tau_lambda, alpha, delta_t_fit = popt
        print("Fitted Parameters:")
        print(f"A_0: {A_0:.2f}, epsilon (−2 + ε): {epsilon - 2:.2f}")
        print(
            f"τ₀ (periodicity): {tau_lambda/dt:.2f}s, α: {alpha:.2f}, Δt: {delta_t_fit:.2f}"
        )

        # Reconstruct model curve and unshift
        wavelengths_fit = np.linspace(min(wavelengths), max(wavelengths), 500)
        y_fit_shifted = model(wavelengths_fit, *popt)
        y_fit = y_fit_shifted - B

        # Plot fitted model
        plt.plot(
            wavelengths_fit,
            y_fit,
            "r--",
            label=f"Fit: $A_0$={A_0:.1f}, $-2+ε$={epsilon - 2:.2f}",
        )

        # Optional suppression curve
        suppression_curve = supression_fit(
            wavelengths_fit, tau_lambda, alpha, delta_t_fit
        )
        plt.plot(wavelengths_fit, suppression_curve, "k-", label="Suppression only")

    except RuntimeError as e:
        print(f"Curve fitting failed: {e}")
        plt.title(f"{title} [fit failed]")

    # Plot layout
    plt.xlabel("Wavelength (nm)")
    plt.ylabel(label_y)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./plots/{filename}", dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_expected_flux(variability, filename, dt, show=True, wavelength_range=None):
    xaxis = variability["Wavelength Center"]
    yaxis = variability["Flux per sec"]

    if wavelength_range is not None:
        selected_idx = (variability["Wavelength Center"] >= wavelength_range[0]) & (
            variability["Wavelength Center"] < wavelength_range[1]
        )

        # Apply filtering
        xaxis = variability.loc[selected_idx, "Wavelength Center"]
        yaxis = variability.loc[selected_idx, "Flux per sec"]

    # Plot Excess Variability vs wavelength
    plt.figure(figsize=(10, 6))
    # print(tabulate(variability.head(200)))
    plt.plot(
        xaxis,
        yaxis,
        label="Expected photon flux",
        color="blue",
        marker="o",
    )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Expected Flux coutn pr bin")
    plt.title(f"Smoothed Power Density function t:{dt:.0f}s")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./plots/{filename}")
    if show:
        plt.show()
    plt.close()


def plot_mean_count(variability, filename, dt, show=True, wavelength_range=None):
    xaxis = variability["Wavelength Center"]
    yaxis = variability["Mean Count"]

    if wavelength_range is not None:
        selected_idx = (variability["Wavelength Center"] >= wavelength_range[0]) & (
            variability["Wavelength Center"] < wavelength_range[1]
        )

        # Apply filtering
        xaxis = variability.loc[selected_idx, "Wavelength Center"]
        yaxis = variability.loc[selected_idx, "Mean Count"]
    # Plot Excess Variability vs wavelength
    plt.figure(figsize=(10, 6))
    plt.plot(
        xaxis,
        yaxis,
        label="Mean count",
        color="blue",
        marker="o",
    )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Mean count")
    plt.title(f"Mean count for wavelength t:{dt:.0f}s")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./plots/{filename}")
    if show:
        plt.show()
    plt.close()


def plot_mimic(meta: FitsMetadata, variability, filename, show=True):
    # Extract wavelength and count
    wavelength = variability["Wavelength Center"]
    fluxpersec = variability["Flux per sec"]

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
    plt.ylabel("Mean count per wavelength")
    plt.title("Mean bin count for wavelength")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./plots/{filename}")
    if show:
        plt.show()
    plt.close()


def plot_shot_noise_variability(
    variability, filename, dt, show=True, wavelength_range=None
):
    # Plot Excess Variability vs wavelength

    xaxis = variability["Wavelength Center"]
    yaxis = variability["Shot Noise"]

    if wavelength_range is not None:
        selected_idx = (variability["Wavelength Center"] >= wavelength_range[0]) & (
            variability["Wavelength Center"] < wavelength_range[1]
        )

        # Apply filtering
        xaxis = variability.loc[selected_idx, "Wavelength Center"]
        yaxis = variability.loc[selected_idx, "Shot Noise"]

    plt.figure(figsize=(10, 6))
    plt.plot(
        xaxis,
        yaxis,
        label="Shot Noise",
        color="green",
        marker="o",
    )
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Shot noise Variability")
    plt.title(f"Shot noise Variability vs Wavelength t:{dt:.0f}s")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./plots/{filename}")
    if show:
        plt.show()
    plt.close()


def plot_excess_variability_smoothed(variability, filename, dt, show=True):
    # Plot Excess Variability vs wavelength
    plt.figure(figsize=(10, 6))
    plt.scatter(
        variability["Wavelength Center"],
        variability["Excess Variability Smoothed"],
        label="Excess Variability (smoothed)",
        color="green",
        marker="o",
    )
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Excess Variability √(σ_total - σ_poisson) / √(σ_λ)")
    plt.title(f"Excess Variability vs Wavelength t:{dt:.0f}s")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./plots/{filename}")
    if show:
        plt.show()
    plt.close()


def plot_excess_variability(variability, filename, dt, show=True):
    # Plot Excess Variability vs wavelength
    plt.figure(figsize=(10, 6))
    plt.scatter(
        variability["Wavelength Center"],
        variability["Excess Variability"],
        label="Excess Variability",
        color="green",
        marker="o",
    )
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Excess Variability √(σ_total - σ_poisson)")
    plt.title(f"Excess Variability vs Wavelength t:{dt:.0f}s")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./plots/{filename}")
    if show:
        plt.show()
    plt.close()


def plot_total_variability(variability, filename, dt, show=True, wavelength_range=None):
    # Plot Excess Variability vs wavelength
    # Plot Excess Variability vs wavelength
    xaxis = variability["Wavelength Center"]
    yaxis = variability["Total Variability"]

    if wavelength_range is not None:
        selected_idx = (variability["Wavelength Center"] >= wavelength_range[0]) & (
            variability["Wavelength Center"] < wavelength_range[1]
        )

        # Apply filtering
        xaxis = variability.loc[selected_idx, "Wavelength Center"]
        yaxis = variability.loc[selected_idx, "Total Variability"]

    plt.figure(figsize=(10, 6))
    plt.plot(
        xaxis,
        yaxis,
        label="Total Variability",
        color="green",
        marker="o",
    )
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Std counts across time")
    plt.title(f"Total Variability vs Wavelength t:{dt:.0f}s")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"./plots/{filename}")
    if show:
        plt.show()
    plt.close()


# Compute and plot JensGPT's Law
