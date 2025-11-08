import numpy as np
import pandas as pd
from common.fitsmetadata import (
    FitsMetadata,
    Spectrum,
    ProcessingParameters,
    ComparePipeline,
)

from common.helper import get_wavelength_bins, get_uneven_time_bin_widths

from common.metadatahandler import save_fits_metadata
from common.powerdensityspectrum import PowerDensitySpectrum, compute_spectrum_params
from event_processing.vartemporal_plotting import vartemporal_plot
from event_processing.plotting import (
    plot_broken_power_law,
    plot_expected_flux,
    plot_chunk_variability_excess,
    plot_flux_excess_variability,
    plot_flux_residual_variability_linear,
    plot_mean_count,
    plot_shot_noise_variability,
    plot_total_variability,
    plot_mimic,
    plot_ccd_bin,
    plot_ccd_energy_map,
    plot_obi_van,
    plot_obi_van_hist,
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


def SDMC_plot(
    meta: FitsMetadata,
    SDMC_df: DataFrame,
    pp: ProcessingParameters,
    pipe: ComparePipeline,
    handle: str = "",
):
    Z_grid = SDMC_df.pivot(
        index="Wavelength Center", columns="time_bin", values="SDMC"
    ).values

    X_vals = SDMC_df["Time Bin 0"].unique() * pp.time_bin_seconds
    Y_vals = SDMC_df["Wavelength Center"].unique()
    X_grid, Y_grid = np.meshgrid(X_vals, Y_vals)
    title = f"{handle} Counts {meta.star}"
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
        f"{pipe.id}/vartemporal_{meta.id}_{pp.time_bin_seconds}s.png",
        show=False,
    )


def make_standard_plots(
    meta: FitsMetadata,
    pp: ProcessingParameters,
    time_binned_source: DataFrame,
    pipline_id: str = "default",
    handle: str = "",
):
    print("Plotting")

    # Expects Wavelength Center
    # Expects Excess Variability

    assert (
        "Wavelength Center" in time_binned_source.columns
    ), "Variability_data requires Wavelength Center column"
    assert (
        "Excess Variability" in time_binned_source.columns
    ), "Variability_data requires Wavelength Center column"

    assert (
        "Wavelength Width" in time_binned_source.columns
    ), "Variability_data requires Wavelength Width column"

    assert "time_bin" in time_binned_source.columns, "Time Bin required"

    assert "Time Bin Width" in time_binned_source.columns, "Time Bin Width required"

    subdir = f"plots/{pipline_id}"
    folder_exists = os.path.isdir(subdir)

    if not folder_exists:
        print("Creating folder for")
        os.mkdir(subdir)
    plot_obi_van_hist(
        variability=time_binned_source,
        pp=pp,
        filename=f"{pipline_id}/plot_obivan_hist_{handle}_{meta.id}.png",
        show=False,
    )

    plot_obi_van(
        variability=time_binned_source,
        pp=pp,
        filename=f"{pipline_id}/plot_obivan_{handle}_{meta.id}.png",
        use_log_scale=False,
        show=False,
    )

    plot_obi_van(
        variability=time_binned_source,
        pp=pp,
        filename=f"{pipline_id}/plot_obivan_log_{handle}_{meta.id}.png",
        use_log_scale=True,
        show=False,
    )

    plot_flux_excess_variability(
        variability=time_binned_source,
        pp=pp,
        filename=f"{pipline_id}/plot_flux_excess_{handle}_{meta.id}.png",
        show=False,
    )

    plot_chunk_variability_excess(
        variability=time_binned_source,
        pp=pp,
        filename=f"{pipline_id}/plot_flickerfit_RV_{handle}_{meta.id}.png",
        show=False,
        use_old=True,
        handle=handle,
    )

    plot_chunk_variability_excess(
        variability=time_binned_source,
        pp=pp,
        filename=f"{pipline_id}/plot_flickerfit_RV_abs_{handle}_{meta.id}.png",
        show=False,
        use_old=False,
        handle=handle,
    )

    plot_chunk_variability_excess(
        variability=time_binned_source,
        pp=pp,
        filename=f"{pipline_id}/plot_flickerfit_RV_abs_old_{handle}_{meta.id}s.png",
        show=False,
        use_old=True,
        handle=handle,
    )

    plot_chunk_variability_excess(
        variability=time_binned_source,
        pp=pp,
        filename=f"{pipline_id}/plot_flickerfit_RV_norm_{handle}_{meta.id}.png",
        show=False,
        use_old=False,
        handle=handle,
    )

    plot_flux_residual_variability_linear(
        variability=time_binned_source,
        pp=pp,
        filename=f"{pipline_id}/plot_excess_variability_linear_{handle}_{meta.id}.png",
        show=False,
        handle=handle,
    )

    plot_broken_power_law(
        variability=time_binned_source,
        pp=pp,
        filename=f"{pipline_id}/plot_RV_2bpl_abs_{handle}_{meta.id}.png",
        show=False,
        use_smoothed_PDS=False,
        use_three=False,
        handle=handle,
    )

    plot_broken_power_law(
        time_binned_source,
        pp,
        f"{pipline_id}/plot_RV_2bpl_norm_{handle}_{meta.id}.png",
        show=False,
        use_smoothed_PDS=True,
        use_three=False,
        handle=handle,
    )

    plot_broken_power_law(
        variability=time_binned_source,
        pp=pp,
        filename=f"{pipline_id}/plot_RV_3bpl_abs_{handle}_{meta.id}.png",
        show=False,
        use_smoothed_PDS=False,
        use_three=True,
        handle=handle,
    )

    plot_broken_power_law(
        variability=time_binned_source,
        pp=pp,
        filename=f"{pipline_id}/plot_RV_3bpl_norm_{handle}_{meta.id}.png",
        show=False,
        use_smoothed_PDS=True,
        use_three=True,
        handle=handle,
    )

    plot_total_variability(
        variability=time_binned_source,
        pp=pp,
        filename=f"{pipline_id}/plot_total_variability_{handle}_{meta.id}.png",
        show=False,
        wavelength_range=[0, 2.0],
        handle=handle,
    )

    plot_excess_variability_smoothed(
        variability=time_binned_source,
        pp=pp,
        filename=f"{pipline_id}/plot_excess_variability_smoothed_{handle}_{meta.id}.png",
        show=False,
        handle=handle,
    )

    plot_excess_variability(
        variability=time_binned_source,
        pp=pp,
        filename=f"{pipline_id}/plot_excess_variability_{handle}_{meta.id}.png",
        show=False,
        handle=handle,
    )

    plot_shot_noise_variability(
        variability=time_binned_source,
        pp=pp,
        filename=f"{pipline_id}/plot_shot_noise_variability_{handle}_{meta.id}.png",
        show=False,
        wavelength_range=[0, 2.0],
        handle=handle,
    )

    plot_mean_count(
        variability=time_binned_source,
        pp=pp,
        filename=f"{pipline_id}/plot_mpc_{handle}_{meta.id}.png",
        show=False,
        wavelength_range=[0, 1.0],
        handle=handle,
    )

    plot_expected_flux(
        variability=time_binned_source,
        pp=pp,
        filename=f"{pipline_id}/plot_expected_cnt_{handle}_{meta.id}.png",
        show=False,
        wavelength_range=[0, 2.0],
        handle=handle,
    )

    plot_mimic(
        meta=meta,
        source=time_binned_source,
        filename=f"{pipline_id}/plot_counts_{handle}_{meta.id}.png",
        show=False,
        handle=handle,
    )
