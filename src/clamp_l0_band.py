import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from common.helper import compare_dataframes, ensure_path_exists, get_duration
from event_processing.plotting import (
    plot_multiple_series,
    compute_time_variability_async,
)
from event_processing.binning import load_source_data, cut_dataset_simple
from event_processing.chandratimeshift import chandrashift
from event_processing.binning import add_wavelength_bin_columns
from common.powerdensityspectrum import (
    compute_spectrum_params,
    exp_count_per_sec,
    PowerDensitySpectrum,
)
from common.metadatahandler import (
    load_fits_metadata,
    save_fits_metadata,
    load_gen_param,
    save_gen_param,
    load_pipeline,
    save_pipeline,
    save_processing_metadata,
    load_processing_param,
    load_chunk_metadata,
    save_chunk_metadata,
)
from common.helper import (
    compare_variability_profiles,
    ensure_pipeline_folders_exists,
    ensure_path_exists,
    get_wavelength_bins,
)

from common.fitsread import (
    fits_save_events_with_pi_channel,
    fits_to_dataframe,
    pi_channel_to_wavelength_and_width,
    read_event_data_crop_and_project_to_ccd,
    fits_save_chunk_analysis,
    fits_read,
    fits_save_cache,
    get_cached_filename,
    fits_read_cache_if_exists,
    fits_save_event_file,
)

from common.fitsmetadata import (
    FitsMetadata,
    Spectrum,
    ProcessingParameters,
    ComparePipeline,
    GenerationParameters,
)
from common.helper import randomly_sample_from

from event_processing.binning import binning_process_distributed
from common.generate_data import generate_synthetic_telescope_data
import csv
import os
import sys
import itertools
import uuid
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import argparse
from pandas import DataFrame
from common.helper import estimate_lanecount
from common.lanesheetMetadata import LanesheetMetadata


def exp_lanecount_curve(genmeta: GenerationParameters, lambdas):
    exp_lanecount = np.ones(len(lambdas))
    i = 0
    for lamb, lambda_bin_width in lambdas:
        t = 0.0
        flux_per_sec = exp_count_per_sec(
            lamb,
            genmeta.spectrum.A,
            genmeta.spectrum.lambda_0,
            genmeta.spectrum.sigma,
            genmeta.spectrum.C,
        )
        assert lambda_bin_width > 0, f"Lambda bin must be > 0, was {lambda_bin_width}"
        print(f"Flux per {lamb}:  {flux_per_sec} per sec from empirical lambda ")

        ls = LanesheetMetadata(
            id=f"{genmeta.id}_{i}",
            truth=False,
            empirical=genmeta.empirical,
            lambda_center=lamb,
            lambda_width=lambda_bin_width,
            exp_flux_per_sec=flux_per_sec,
            alpha=genmeta.alpha,
            # theta=genmeta.theta,
            r_e=genmeta.r_e,
            perp=genmeta.perp,
            phase=genmeta.phase,
            lucretius=genmeta.lucretius,
            lucretius_tolerance=None,
            alpha_tolerance=None,
            perp_tolerance=None,
            phase_tolerance=None,
            theta_tolerance=None,
        )

        exp_lanecount[i], g = estimate_lanecount(
            ls
        )  # Use the new, empirical computation of lanecount which has parity with
        if i == 0:
            print(f"First value is {exp_lanecount[i]}")
        i += 1

    return exp_lanecount


def main(use_defaults=True, overload_defaults: bool = True):
    print("Clamp band")

    start = 1e32

    pipe = load_pipeline("anisogen_empirical")
    pp = load_processing_param(pipe.pp_id)
    genmeta = load_gen_param(pipe.gen_id)

    lambda_nm = np.flip(
        np.linspace(genmeta.max_wavelength, genmeta.min_wavelength, pp.wavelength_bins)
    )
    lambda_bin_width = np.ones(pp.wavelength_bins) * np.abs(
        (lambda_nm[0] - lambda_nm[1])
    )
    lambdas = np.stack((lambda_nm, lambda_bin_width), axis=1)
    series = []
    l0_vals = []
    for i in range(0, 20):
        l0 = (genmeta.lucretius / 100) * (90 + i)
        l0_vals.append(l0)
        genmeta.lucretius = l0
        exp_lanecount_spectrum = exp_lanecount_curve(genmeta, lambdas)
        series.append(exp_lanecount_spectrum)

    ensure_path_exists("plots/empirical")
    plot_multiple_series(lambda_nm, series, l0_vals, f"plots/empirical/l0_bounds.png")


if __name__ == "__main__":
    main(use_defaults=True, overload_defaults=False)
