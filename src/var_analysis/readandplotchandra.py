import numpy as np
import pandas as pd
from common.fitsmetadata import FitsMetadata, Spectrum, ProcessingParameters
from common.metadatahandler import save_fits_metadata
from common.powerdensityspectrum import PowerDensitySpectrum, compute_spectrum_params
from var_analysis.plotting import (
    plot_broken_power_law,
    plot_excess_variability,
    plot_excess_variability_smoothed,
    plot_expected_flux,
    plot_flickering,
    plot_flux_excess_variability,
    plot_flux_residual_variability_linear,
    plot_mean_count,
    plot_shot_noise_variability,
    plot_total_variability,
    plot_mimic,
    plot_ccd_bin,
    plot_ccd_energy_map,
)
from tabulate import tabulate
from pandas import DataFrame
import os


def experiment_exists(df, r_e, theta, alpha, lucretius):
    if "r_e" not in df.columns:
        return False
    if "theta" not in df.columns:
        return False
    if "alpha" not in df.columns:
        return False
    if "lucretius" not in df.columns:
        return False
    match = df[
        (df["r_e"] == r_e)
        & (df["theta"] == theta)
        & (df["alpha"] == alpha)
        & (df["lucretius"] == lucretius)
    ]
    return not match.empty


def make_time_bins(duration, time_bin_seconds_seconds):
    time_edges = np.arange(0, duration, time_bin_seconds_seconds)

    return time_edges


def var_analysis_plot(
    meta: FitsMetadata,
    pp: ProcessingParameters,
    source_data,
    variability_data,
    skip_ccd=True,
):
    print("Plotting")

    if not skip_ccd:
        plot_ccd_energy_map(
            source_data,
            pp.time_bin_seconds,
            f"{meta.id}/plot_{meta.id}_energy_{meta.id}.png",
            False,
        )
        plot_ccd_bin(
            source_data, f"{meta.id}/plot_{meta.id}_count_{meta.id}.png", False
        )

    # Expects Wavelength Center
    # Expects Excess Variability

    assert (
        "Wavelength Center" in variability_data.columns
    ), "Variability_data requires Wavelength Center column"
    assert (
        "Excess Variability" in variability_data.columns
    ), "Variability_data requires Wavelength Center column"

    assert (
        "Wavelength Width" in variability_data.columns
    ), "Variability_data requires Wavelength Width column"
    subdir = f"plots/{meta.id}"
    folder_exists = os.path.isdir(subdir)

    if not folder_exists:
        print("Creating folder for")
        os.mkdir(subdir)
    plot_flux_excess_variability(
        variability_data,
        f"{meta.id}/plot_{meta.id}_RV_{pp.time_bin_seconds}s.png",
        pp.time_bin_seconds,
        False,
    )

    plot_flickering(
        variability_data,
        f"{meta.id}/plot_{meta.id}_flickerfit_RV_norm_old_{pp.time_bin_seconds}s.png",
        pp.time_bin_seconds,
        show=False,
        use_old=True,
        use_normalized=True,
    )

    plot_flickering(
        variability_data,
        f"{meta.id}/plot_{meta.id}_flickerfit_RV_abs_{pp.time_bin_seconds}s.png",
        pp.time_bin_seconds,
        show=False,
        use_old=False,
        use_normalized=False,
    )

    plot_flickering(
        variability_data,
        f"{meta.id}/plot_{meta.id}_flickerfit_RV_abs_old_{pp.time_bin_seconds}s.png",
        pp.time_bin_seconds,
        show=False,
        use_old=True,
        use_normalized=False,
    )

    plot_flickering(
        variability_data,
        f"{meta.id}/plot_{meta.id}_flickerfit_RV_norm_{pp.time_bin_seconds}s.png",
        pp.time_bin_seconds,
        show=False,
        use_old=False,
        use_normalized=True,
    )

    plot_flux_residual_variability_linear(
        variability_data,
        f"{meta.id}/plot_{meta.id}_excess_variability_linear_{pp.time_bin_seconds}s.png",
        pp.time_bin_seconds,
        False,
    )

    plot_broken_power_law(
        variability_data,
        f"{meta.id}/plot_{meta.id}_RV_2bpl_abs_{pp.time_bin_seconds}s.png",
        pp.time_bin_seconds,
        show=False,
        use_smoothed_PDS=False,
        use_three=False,
    )

    plot_broken_power_law(
        variability_data,
        f"{meta.id}/plot_{meta.id}_RV_2bpl_norm_{pp.time_bin_seconds}s.png",
        pp.time_bin_seconds,
        show=False,
        use_smoothed_PDS=True,
        use_three=False,
    )

    plot_broken_power_law(
        variability_data,
        f"{meta.id}/plot_{meta.id}_RV_3bpl_abs_{pp.time_bin_seconds}s.png",
        pp.time_bin_seconds,
        show=False,
        use_smoothed_PDS=False,
        use_three=True,
    )

    plot_broken_power_law(
        variability_data,
        f"{meta.id}/plot_{meta.id}_RV_3bpl_norm_{pp.time_bin_seconds}s.png",
        pp.time_bin_seconds,
        show=False,
        use_smoothed_PDS=True,
        use_three=True,
    )

    plot_total_variability(
        variability_data,
        f"{meta.id}/plot_{meta.id}_total_variability_{pp.time_bin_seconds}s.png",
        pp.time_bin_seconds,
        False,
        [0, 2.0],
    )

    plot_excess_variability_smoothed(
        variability_data,
        f"{meta.id}/plot_{meta.id}_excess_variability_smoothed_{pp.time_bin_seconds}s.png",
        pp.time_bin_seconds,
        False,
    )

    plot_excess_variability(
        variability_data,
        f"{meta.id}/plot_{meta.id}_excess_variability_{pp.time_bin_seconds}s.png",
        pp.time_bin_seconds,
        False,
    )

    plot_shot_noise_variability(
        variability_data,
        f"{meta.id}/plot_{meta.id}_shot_noise_variability_{pp.time_bin_seconds}s.png",
        pp.time_bin_seconds,
        False,
        [0, 2.0],
    )

    plot_mean_count(
        variability_data,
        f"{meta.id}/plot_{meta.id}_mpc_{pp.time_bin_seconds}s.png",
        pp.time_bin_seconds,
        False,
        [0, 1.0],
    )

    plot_expected_flux(
        variability_data,
        f"{meta.id}/plot_{meta.id}_expected_cnt_{pp.time_bin_seconds}s.png",
        pp.time_bin_seconds,
        False,
        [0, 2.0],
    )

    plot_mimic(
        meta,
        variability_data,
        f"{meta.id}/plot_{meta.id}_counts_{pp.time_bin_seconds}s.png",
        False,
    )


def binning_process(
    source_data: DataFrame, meta: FitsMetadata, pp: ProcessingParameters
) -> tuple[DataFrame, FitsMetadata]:
    try:
        """Loads a metadata file"""
        # meta = load_fits_metadata(filename)
        # source_data = pd.read_csv(f"temp/{meta.source_filename}")
        # background_data = pd.read_csv(meta.background_filename)

        if pp.take_time_seconds:
            duration = pp.take_time_seconds
        else:
            duration = meta.t_max

        max_time_in_data = source_data["relative_time"].max()

        if max_time_in_data - 1 > duration:
            raise Exception(
                f"Dataset should have been cropped in time. Saw time {max_time_in_data} but only {duration} is allowed"
            )

        min_lambda = meta.get_min_wavelength()
        max_lambda = meta.get_max_wavelength()
        print(
            f"Filtering the dataset based on high and low wavelength. From [{min_lambda:.2f}, {max_lambda:.2f}]nm"
        )
        if np.abs(min_lambda) < 0.001:
            print(f" Somehow the min lambda is too low!")
        if np.abs(max_lambda) < 0.001:
            print(f" Somehow the max lambda is too low!")
        valid_range = source_data["Wavelength (nm)"] >= min_lambda
        source_data = source_data[valid_range].reset_index(drop=True)
        valid_range = source_data["Wavelength (nm)"] <= max_lambda

        source_data = source_data[valid_range].reset_index(drop=True)
        if len(source_data) == 0:
            return source_data, meta
        # Suppose we want 1-second time bins:
        print(source_data.head(200))
        bin_edges = make_time_bins(duration, pp.time_bin_seconds)
        if len(bin_edges) <= 1:
            raise Exception(f"Problem occured, needs more than one bin edge")
        print("Bin edges:", bin_edges)
        # Example output: [0.   1.   2.   3.   4.  ] if max_time was ~3.14

        # First we bin the events based on the number of bin edges
        source_data["Time Bin"] = pd.cut(
            source_data["relative_time"],
            bins=bin_edges,
            labels=False,
            include_lowest=True,
        )

        lamb_min = source_data["Wavelength (nm)"].min()
        lamb_max = source_data["Wavelength (nm)"].max()

        if np.isnan(lamb_min):
            raise Exception("Lamb min is nan")

        if np.isnan(lamb_max):
            raise Exception("Lamb min is nan")

        wave_edges = np.linspace(
            lamb_min,
            lamb_max,
            pp.wavelength_bins + 1,
        )
        print("Bin edges:", wave_edges)
        wave_centers = 0.5 * (wave_edges[:-1] + wave_edges[1:])
        wave_widths = np.diff(wave_edges)

        source_data["Wavelength Bin"] = pd.cut(
            source_data["Wavelength (nm)"],
            bins=wave_edges,
            labels=False,
            include_lowest=True,
        )

        source_data.sort_values(["Wavelength Bin"], inplace=True)
        source_data["Wavelength Center"] = source_data["Wavelength Bin"].apply(
            lambda i: wave_centers[i] if pd.notnull(i) else np.nan
        )

        source_data["Wavelength Width"] = source_data["Wavelength Bin"].apply(
            lambda i: wave_widths[i] if pd.notnull(i) else np.nan
        )

        print("Estimate spectrum")
        generated_spectrum, r_squared = compute_spectrum_params(
            meta=meta, pp=pp, source_data=source_data
        )
        print(f"Fitted Spectrum with residual error {r_squared:.3f}")
        print(generated_spectrum.to_string())
        # print(f"Estimating spectrum params from {len(count_per_lambda_bin)} obs")

        meta.apparent_spectrum = generated_spectrum

        save_fits_metadata(metadata=meta)

    except Exception as e:
        print(f"Exception occured in binning process {repr(e)}")
        raise e
    return source_data, meta
