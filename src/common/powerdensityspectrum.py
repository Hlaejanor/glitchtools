import pandas as pd
from common.fitsmetadata import FitsMetadata, Spectrum, ProcessingParameters
from pandas import DataFrame
from common.helper import get_duration
from common.fitsread import chandra_like_pi_mapping, pi_channel_to_wavelength_and_width
import numpy as np
from scipy.optimize import curve_fit


class PowerDensitySpectrum:
    def __init__(
        self,
        spectrum: Spectrum = Spectrum(A=175, sigma=1, lambda_0=1.0, C=1.0),
        mode="wavelength",
    ):
        self.id = id
        self.mode = mode

        self.spectrum = spectrum


def exp_count_per_sec(lambda_nm, A: float, lambda_0: float, sigma: float, C: float):
    log_term = (np.log(lambda_nm / lambda_0) / sigma) ** 2
    return A * np.exp(-log_term) + C


def compute_spectrum_params_old(
    meta: FitsMetadata, pp: ProcessingParameters, source_data: DataFrame
) -> tuple[Spectrum, float]:
    try:
        print(source_data.columns)

        assert (
            "Wavelength Center" in source_data.columns
        ), "Compute spectrum params require the source_data to be cut on wavelength bins"
        assert (
            "Wavelength Width" in source_data.columns
        ), "Compute spectrum params require the Wavelength Widht column to exist"

        duration = get_duration(meta, pp)

        # Group by wavelength bin and count
        grouped_data = (
            source_data.groupby(["Wavelength Center", "Wavelength Width"])
            .size()
            .reset_index(name="Count")
        )

        xaxis = grouped_data["Wavelength Center"]
        y_flux_density = grouped_data["Count"] / (
            duration * grouped_data["Wavelength Width"]
        )

        # Fit model to data
        A_guess = y_flux_density.max()
        lambda_0_guess = xaxis[y_flux_density.argmax()]
        sigma_guess = 0.5
        C_guess = max(0.0, y_flux_density.min())

        popt, _ = curve_fit(
            exp_count_per_sec,
            xaxis,
            y_flux_density,
            p0=[A_guess, lambda_0_guess, sigma_guess, C_guess],
            bounds=([0, 0.1, 0.01, 0], [np.inf, np.inf, np.inf, np.inf]),
        )

        A_fit, lambda_0_fit, sigma_fit, C_fit = popt
        fitted_y = exp_count_per_sec(xaxis, *popt)
        residuals = y_flux_density - fitted_y
        r_squared = 1 - np.sum(residuals**2) / np.sum(
            (y_flux_density - y_flux_density.mean()) ** 2
        )

        return Spectrum(A_fit, lambda_0_fit, sigma_fit, C_fit), r_squared

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise Exception("Spectrum fitting failed.") from e


def compute_spectrum_params(
    meta: FitsMetadata, source_data: DataFrame
) -> tuple[Spectrum, float]:
    # Preconditions
    assert (
        "Wavelength Center" in source_data.columns
    ), "source_data must be pre-binned by wavelength"
    assert "Wavelength Width" in source_data.columns, "missing 'Wavelength Width'"

    duration = get_duration(meta)

    # Group -> flux density
    grouped = (
        source_data.groupby(["Wavelength Center", "Wavelength Width"])
        .size()
        .reset_index(name="Count")
    )

    grouped = grouped.sort_values("Wavelength Center").reset_index(drop=True)

    lam = grouped["Wavelength Center"].to_numpy(dtype=float)  # nm
    w = grouped["Wavelength Width"].to_numpy(dtype=float)  # nm
    k = grouped["Count"].to_numpy(dtype=float)

    # Flux density and its Poisson stdev
    y = k / (duration * w)
    y_sigma = np.where(k > 0, np.sqrt(k) / (duration * w), 1.0 / (duration * w))

    # Initial guesses (robust-ish)
    # Weighted median for lambda0
    weights = y / (y_sigma + 1e-12)
    weights = np.clip(weights, 0, np.percentile(weights, 95))  # cap extremes
    cdf = np.cumsum(weights) / (weights.sum() + 1e-12)
    lam0_guess = np.interp(0.5, cdf, lam)

    y_min, y_max = float(np.min(y)), float(np.max(y))
    C_guess = np.percentile(y, 10)
    A_guess = max(y_max - C_guess, y_max * 0.25)
    sigma_guess = 0.15  # dimensionless log-width; tune per instrument

    # Bounds: lambda0 within data range; sigma in plausible log-range
    lam_lo, lam_hi = float(lam.min()), float(lam.max())
    bounds_lo = [0.0, max(lam_lo * 0.9, 1e-6), 0.03, 0.0]
    bounds_hi = [np.inf, lam_hi * 1.1, 0.7, np.inf]

    # Fit
    popt, pcov = curve_fit(
        exp_count_per_sec,
        lam,
        y,
        p0=[A_guess, lam0_guess, sigma_guess, C_guess],
        bounds=(bounds_lo, bounds_hi),
        sigma=y_sigma,
        absolute_sigma=True,
        maxfev=20000,
    )
    A_fit, lambda_0_fit, sigma_fit, C_fit = map(float, popt)

    fitted = exp_count_per_sec(lam, *popt)
    resid = y - fitted
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Diagnostics: flag “flat” solutions
    flatish = (sigma_fit > 0.65) or (A_fit < 0.1 * (C_fit + 1e-12))
    if flatish:
        print(
            "[warn] Spectrum fit is nearly flat. "
            f"sigma={sigma_fit:.3f}, A={A_fit:.3g}, C={C_fit:.3g}. "
            "Consider a power-law fit or tighter sigma bounds."
        )

    return Spectrum(A_fit, lambda_0_fit, sigma_fit, C_fit), r2


def compute_spectrum_params_2(
    meta: FitsMetadata, data: DataFrame
) -> PowerDensitySpectrum:
    """Compute per-bin flux spectrum from photon data."""

    if "Hit" in data.columns:
        data["Hit"] = data["Hit"].astype(int)
        data = data[data["Hit"] == 1]

    # Determine mode
    if meta.lowPI is not None and meta.highPI is not None:
        print("Mode: PI")
        mode = "PI"
        bin_column = "pi"
        bins = np.arange(meta.lowPI, meta.highPI + 1)  # Explicit integer bins
    else:
        print("Mode: Wavelength")
        mode = "wavelength"
        bin_column = "Wavelength (nm)"
        wmin = data[bin_column].min()
        wmax = data[bin_column].max()
        bins = np.linspace(wmin, wmax, meta.wavelength_bins + 1)

    exposure_time = meta.t_max

    if mode == "PI":
        # Assign PI bins (no need for pd.cut, just integer mapping)
        data["Wavelength Bin"] = data[bin_column] - meta.lowPI

        # Build a DataFrame with all bins even if some have 0 counts
        total_bins = meta.highPI - meta.lowPI + 1
        all_bins = pd.DataFrame({"Wavelength Bin": np.arange(total_bins)})

        # Group and merge
        grouped = (
            data.groupby("Wavelength Bin")
            .size()
            .reset_index(name="Wavelength Bin Count")
        )
        merged = all_bins.merge(grouped, on="Wavelength Bin", how="left").fillna(0)
        merged["Wavelength Bin Count"] = merged["Wavelength Bin Count"].astype(int)

        # Build output
        out_list = []
        for wbin in range(total_bins):
            pi_value = meta.lowPI + wbin

            bin_counts = merged.loc[
                merged["Wavelength Bin"] == wbin, "Wavelength Bin Count"
            ].values[0]
            counts_per_second = bin_counts / exposure_time
            bin_center, bin_width, pi_channel = pi_channel_to_wavelength_and_width(
                pi_value
            )
            out_list.append(
                {
                    "Bin index": wbin,
                    "PI Channel": pi_value,
                    "PI width": 1.0,
                    "Wavelength Center": bin_center,  # For uniformity
                    "Wavelength Width": bin_width,  # Always width 1 PI channel
                    "Total Counts": bin_counts,
                    "Counts per second": counts_per_second,
                }
            )

        output = pd.DataFrame(out_list)

    else:
        # Wavelength mode: use pd.cut
        data["Wavelength Bin"] = pd.cut(
            data[bin_column],
            bins=bins,
            labels=False,
            include_lowest=True,
        )

        grouped = (
            data.groupby("Wavelength Bin")
            .size()
            .reset_index(name="Wavelength Bin Count")
        )

        out_list = []
        for wbin in range(len(bins) - 1):
            bin_center = 0.5 * (bins[wbin] + bins[wbin + 1])
            bin_width = bins[wbin + 1] - bins[wbin]

            row = grouped[grouped["Wavelength Bin"] == wbin]
            bin_counts = row["Wavelength Bin Count"].values[0] if len(row) > 0 else 0
            counts_per_second = bin_counts / exposure_time
            pi_value = chandra_like_pi_mapping(bin_center)

            out_list.append(
                {
                    "Bin index": wbin,
                    "PI Channel": pi_value,
                    "PI width": None,
                    "Wavelength Center": bin_center,
                    "Wavelength Width": bin_width,
                    "Total Counts": bin_counts,
                    "Counts per second": counts_per_second,
                }
            )

        output = pd.DataFrame(out_list)

    # Instantiate PDS
    pds = PowerDensitySpectrum(meta.id, output, mode)
    return pds
