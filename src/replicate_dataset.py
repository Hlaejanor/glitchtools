import pandas as pd
import numpy as np
from common.helper import compare_dataframes, get_duration
from event_processing.plotting import (
    plot_spectrum_vs_data,
    compute_time_variability_async,
)
from event_processing.binning import load_source_data
from event_processing.binning import add_wavelength_bin_columns
from common.powerdensityspectrum import compute_spectrum_params, PowerDensitySpectrum
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
from event_processing.binning import get_binned_datasets
from common.helper import compare_variability_profiles
from common.fitsread import (
    fits_save_events_with_pi_channel,
    read_event_data_crop_and_project_to_ccd,
    fits_save_chunk_analysis,
    fits_read,
    fits_save_cache,
    get_cached_filename,
    fits_read_cache_if_exists,
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
from common.generate_data import (
    generate_synthetic_telescope_data,
    generate_synth_if_need_be,
)
import csv
import os
import sys
import itertools
import uuid
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import argparse


def parse_arguments(use_defaults: bool):
    parser = argparse.ArgumentParser(
        description="Create mimic dataset: Generate synthetic dataset B to mimic real dataset A"
    )
    if use_defaults:
        parser.add_argument(
            "-s",
            "--source",
            required=not use_defaults,
            default="antares",
            type=str,
            help="Select a source dataset",
        )

        parser.add_argument(
            "-g",
            "--gen_id",
            required=not use_defaults,
            default="anisogen",
            type=str,
            help="Select a generation parameters",
        )

        parser.add_argument(
            "-pp",
            "--pp",
            required=not use_defaults,
            default="default",
            type=str,
            help="Select a processing parameters",
        )

    args = parser.parse_args()

    return args


def create_default_pipeline():
    pipeline = load_pipeline("default")

    default_meta = load_fits_metadata(pipeline.A_fits_id)
    # antares_meta = FitsMetadata(
    #    id="antares0",
    #    raw_event_file="antares_chandra/15734/primary/hrcf15734N003_evt2.fits",
    #    source_pos_x=512,
    #    source_pos_y=512,
    #    star="Antares",
    #    synthetic=False,
    #    max_energy=11.0,
    #    min_energy=1,
    #    source_count=None,
    #    apparent_spectrum=None,
    #    t_max=None,
    # )
    # save_fits_metadata(antares_meta)

    # assert antares_meta is not None, "Chandra meta needs to be called"

    pparam = ProcessingParameters(
        id="test_1",
        resolution=1024,
        processed_filename=None,
        source_radius=512,
        max_wavelength=1.0,
        min_wavelength=0.1,
        wavelength_bins=10,
        time_bin_seconds=100,
        start_time_seconds=0,
        end_time_seconds=10000,
        anullus_radius_inner=None,
        anullus_radius_outer=None,
        padding_strategy=False,
        downsample_strategy=None,
    )
    save_processing_metadata(pparam)

    gen = GenerationParameters(
        id="test_1",
        alpha=3.0,
        lucretius=-1,
        velocity=1.0,
        theta_change_per_sec=0.0,
        r_e=0.8,
        theta=0.51,
        t_max=pparam.end_time_seconds,
        perp=0.0,
        phase=0.0,
        max_wavelength=1.0,
        min_wavelength=0.12,
        spectrum=pipeline.apparent_spectrum,
        star="Antares gen",
    )

    save_gen_param(gen)
    if not pipeline:
        pipeline = ComparePipeline(
            id=pipeline.id,
            A_fits_id=default_meta.id,
            B_fits_id=None,
            pp_id=pparam.id,
            gen_id=gen.id,
        )
    else:
        pipeline.A_fits_id = default_meta.id
        pipeline.B_fits_id = None
        pipeline.pp_id = pparam.id
        pipeline.gen_id = gen.id

        save_pipeline(pipeline)


def main(use_defaults=False, generate: bool = True):
    print("Create mimic dataset")

    # Parse command line arguments
    args = parse_arguments(use_defaults)

    genparam = load_gen_param(args.gen_id)
    assert genparam is not None, "Generation parameters must be set"
    pp = load_processing_param(args.pp)
    assert pp is not None, "Processing parameters must be set"

    # Load metadata for target set A
    source_meta = load_fits_metadata(args.source)
    log_A, source, source_A = load_source_data(source_meta, pp)
    filename = f"plots/spectra/{source.id}.png"
    # Show that the computed spectrum matches the actual data
    plot_spectrum_vs_data(
        meta_A=source_meta,
        source_A=source,
        filename=filename,
        show=False,
    )

    estimated_spectrum, r_squared = compute_spectrum_params(
        meta=source_meta, pp=pp, source_data=source_A
    )
    source_meta.apparent_spectrum = estimated_spectrum
    genparam.spectrum = source.apparent_spectrum

    genparam.t_max = source.t_max
    genparam.t_min = source.t_min

    genparam.t_max = source.t_max
    genparam.t_min = source.t_min
    # Save the generation params
    save_gen_param(genparam)
    # Generate a synthetic event file using B_gen, which include the spectrum and duration
    generated_events_arr = generate_synthetic_telescope_data(
        genparam, pp.wavelength_bins
    )
    #
    generated_events = pd.DataFrame(generated_events_arr, columns=source_A.columns)
    # Save the the generated dataset as a fits file
    generated_meta, generated_events = fits_save_events_with_pi_channel(
        generated_events, genparam
    )
    # Update the pipeline with reference to the new fits file
    save_fits_metadata(source_meta)
    save_gen_param(genparam)

    filename = f"plots/spectra/{source_meta.id}_vs_{generated_meta.id}.png"
    # Show that the computed spectrum matches the actual data
    plot_spectrum_vs_data(
        meta_A=source,
        source_A=source_A,
        meta_B=generated_meta,
        source_B=generated_events,
        pp=pp,
        filename=filename,
        show=False,
    )

    print("Saved spectrum comparison")


if __name__ == "__main__":
    main(use_defaults=True)
