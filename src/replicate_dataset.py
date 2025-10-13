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
from event_processing.binning import add_time_binning
from common.helper import compare_variability_profiles
from common.fitsread import (
    fits_save_events_generated,
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
    default_pipeline = "anisogen_vs_rnd_anisogen"
    default_pipeline = "antares_vs_anisogen2"
    default_pipeline = "antares_vs_anisogen"
    default_pipeline = "isogen_vs_rnd_isogen"
    default_pipeline = "antares_vs_isogen"
    default_pipeline = "antares_vs_anisogen"
    default_pipeline = "antares_vs_anisogen2"
    default_pipeline = "anisogen_vs_rnd_anisogen"
    parser = argparse.ArgumentParser(
        description="Create mimic dataset: Generate synthetic data variants."
    )
    if use_defaults:
        parser.add_argument(
            "-p",
            "--pipeline",
            required=not use_defaults,
            default=default_pipeline if use_defaults else None,
            type=str,
            help="Set the pipeline you want to run",
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
        take_time_seconds=10000,
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
        theta_change_per_sec=0.0,
        r_e=0.8,
        theta=0.51,
        t_max=pparam.take_time_seconds,
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

    #  Ensure that the default pipeline is present
    # create_default_pipeline()

    # Parse command line arguments
    args = parse_arguments(use_defaults)
    assert (
        args.pipeline is not None
    ), "Pipeline meta missing. Try to run create_default_pipeline.sh"
    # Load the default or specified pipeline
    pipe = load_pipeline(args.pipeline)
    # Load the processing params
    pp = load_processing_param(pipe.pp_id)
    assert pp is not None, "Processing param was none"
    # Load the generation parameter from the pipeline
    genparam = load_gen_param(pipe.gen_id)
    assert genparam is not None, "Generation parameters must be set"

    # Load metadata for target set A
    meta_A = load_fits_metadata(pipe.A_fits_id)
    log_A, meta_A, source_A = load_source_data(meta_A, pp)
    filename = f"plots/{pipe.id}/plot_spectrum_{meta_A.id}.png"
    # Show that the computed spectrum matches the actual data
    plot_spectrum_vs_data(
        meta_A=meta_A,
        source_A=source_A,
        pp=pp,
        filename=filename,
        show=False,
    )

    """
    # Apply binning according to specs in ProcessingParameters
    logA, meta_A, binned_datasets = load_event_file_and_apply_binning(
        pipeline=pipe, source_data=source_A, meta=meta_A, pp=pp, handle="A"
    )
    """

    if generate:
        print(f"Generating synthetic dataset B")
        # Ensure that the generation parameters uses the same spectrum as the target dataset
        genparam.spectrum = meta_A.apparent_spectrum
        # Ensure that we don't generate more data than necessary
        genparam.t_max = meta_A.t_max
        # Save the generation params
        save_gen_param(genparam)
        # Generate a synthetic event file using B_gen, which include the spectrum and duration
        B_events = generate_synthetic_telescope_data(genparam, pp.wavelength_bins)
        # Save the the generated dataset as a fits file
        meta_B = fits_save_events_generated(B_events, genparam)
        # Update the pipeline with reference to the new fits file
        pipe.B_fits_id = meta_B.id
        # Save the pipeline
        save_pipeline(pipe)
    else:
        # If we skip generation, load the pipeline metadata
        meta_B = load_fits_metadata(pipe.B_fits_id)

    log_B, meta_B, source_B = load_source_data(meta_B, pp)
    filename = f"plots/{pipe.id}/plot_spectrum_A_{meta_A.id}_vs_B_{meta_B.id}.png"
    # Show that the computed spectrum matches the actual data
    plot_spectrum_vs_data(
        meta_A=meta_A,
        source_A=source_A,
        meta_B=meta_B,
        source_B=source_B,
        pp=pp,
        filename=filename,
        show=False,
    )

    print("Saved spectrum comparison")


if __name__ == "__main__":
    main(use_defaults=True, generate=True)
