import pandas as pd
import numpy as np
from common.helper import compare_dataframes, ensure_path_exists, get_duration
from event_processing.plotting import (
    plot_spectrum_vs_data,
    compute_time_variability_async,
)
from event_processing.binning import load_source_data, cut_dataset_simple
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
from common.helper import (
    compare_variability_profiles,
    ensure_pipeline_folders_exists,
    ensure_path_exists,
)
from common.fitsread import (
    fits_save_events_generated,
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


def poissonize_homogeneous(
    source_data: pd.DataFrame,
    meta: FitsMetadata,
    time_column: str = "relative_time",
    rng=None,
):
    """
    Generate homogeneous Poisson event times, but preserve spectrum shape and count
    on [t_min, t_max).
    If N is given, draw exactly n_fixed times i.i.d.
    """

    t_min = float(source_data[time_column].min())
    t_max = float(source_data[time_column].max())
    N = source_data.shape[0]
    rng = np.random.default_rng() if rng is None else rng

    times = rng.uniform(t_min, t_max, size=N)
    # Unconditional simulation via exponential gaps
    source_data[time_column] = times
    meta.t_min = t_min
    meta.t_max = t_max
    return source_data, meta


def parse_arguments(use_defaults: bool):
    parser = argparse.ArgumentParser(
        description="Create mimic dataset: Generate synthetic data variants."
    )
    if use_defaults:
        parser.add_argument(
            "-p",
            "--pipe",
            required=not use_defaults,
            default=None,
            type=str,
            help="Set the pipeline you want to run",
        )
        parser.add_argument(
            "-a",
            "--a_id",
            required=not use_defaults,
            default=None,
            type=str,
            help="Set the A dataset you want to test",
        )
        parser.add_argument(
            "-b",
            "--b_id",
            required=not use_defaults,
            default=None,
            type=str,
            help="Set the B dataset you want to use as a null",
        )
        parser.add_argument(
            "-pp",
            "--pp_id",
            required=not use_defaults,
            default="default",
            type=str,
            help="Set the Processing parameters, will default to 'default'",
        )
        parser.add_argument(
            "-g",
            "--gen_id",
            required=not use_defaults,
            default=None,
            type=str,
            help="Set the generation parameters",
        )
        parser.add_argument(
            "-tA",
            "--taskA",
            required=not use_defaults,
            default="generate",
            type=str,
            help="Generate the A dataset",
        )
        parser.add_argument(
            "-tB",
            "--taskB",
            required=not use_defaults,
            default="poissonize",
            type=str,
            help="Poissonize the A dataset to create B",
        )

    args = parser.parse_args()

    return args


def main(use_defaults=False, generate: bool = True):
    print("Create a poissonizied and store that as B in the pipeline")

    #  Ensure that the default pipeline is present
    # create_default_pipeline()

    # Parse command line arguments
    args = parse_arguments(use_defaults)
    if True:
        args.pipe = "anisogen_vs_rnd_anisogen"
        args.a_id = "anisogen"
        args.b_id = "anisogen_hp"
        args.gen_id = "anisogen"
        args.taskA = "generate"
        args.taskB = "poissonize"
        args.pp = "default"

    assert (
        args.pipe is not None
    ), "Pipeline name missing, cannot create pipeline if you won't say the name!"
    # Load the default or specified pipeline
    pipe = load_pipeline(args.pipe)
    if pipe is not None:
        print("WARNING : Pipeline exists, modifying this pipe")
        pipe.A_fits_id = args.a_id
        pipe.B_fits_id = args.b_id
        pipe.gen_id = args.gen_id
        pipe.pp_id = args.pp_id
    else:
        pipe = ComparePipeline(args.pipe, args.a_id, args.b_id, args.pp_id, args.gen_id)
    # Load the processing params

    pp = load_processing_param(pipe.pp_id)
    assert (
        pp is not None
    ), "Processing param was none, use default or make sure that meta file exists"

    # Load metadata for target set A
    meta_A = load_fits_metadata(pipe.A_fits_id)
    assert pp is not None, "A slot in pipeline cannot be None"

    if args.taskA == "generate":
        genparam = load_gen_param(pipe.gen_id)
        assert genparam is not None, "Generation parameters must be set"
        print("Generating synthetic dataset A")
        # Ensure that the generation parameters uses the same spectrum as the target dataset
        genparam.spectrum = meta_A.apparent_spectrum
        # Ensure that we don't generate more data than necessary
        genparam.t_max = meta_A.t_max
        # Save the generation params
        save_gen_param(genparam)
        # Generate a synthetic event file using B_gen, which include the spectrum and duration
        A_events = generate_synthetic_telescope_data(genparam, pp.wavelength_bins)
        # Save the the generated dataset as a fits file
        meta_A = fits_save_events_generated(A_events, genparam, use_this_id=args.a_id)
        # Update the pipeline with reference to the new fits file
        pipe.A_fits_id = meta_A.id
        save_fits_metadata(meta_A)

    else:
        meta_A = load_fits_metadata(pipe.A_fits_id)

    raw_events_A = fits_read(meta_A.raw_event_file)
    print(f"Loaded {len(raw_events_A)} events from file {meta_A.raw_event_file}")

    assert len(raw_events_A) > 0, "Raw event file for slot A cannot be empty"

    if args.taskB == "poissonize":
        print("Assuming B is homogeneous Poissonization of A")
        ensure_path_exists("fits/poissonized/")
        filename = f"fits/poissonized/{meta_A.id}_hp.fits"
        meta_B = FitsMetadata(
            f"{meta_A.id}_hp",
            filename,
            True,
            meta_A.source_pos_x,
            meta_A.source_pos_y,
            meta_A.max_energy,
            meta_A.min_energy,
            meta_A.source_count,
            meta_A.star,
            meta_A.t_min,
            meta_A.t_max,
            meta_A.gen_id,
            meta_A.ascore,
            meta_A.apparent_spectrum,
        )
        raw_events_B, meta_B = poissonize_homogeneous(
            raw_events_A.copy(), time_column="time", meta=meta_B
        )
        print("Done poissonizing")
        save_fits_metadata(meta=meta_B)
        fits_save_event_file(raw_events_B, meta_B)
        pipe.B_fits_id = meta_B.id
        meta_B = load_fits_metadata(pipe.B_fits_id)

    else:
        # If we skip generation, load the pipeline metadata
        meta_B = load_fits_metadata(pipe.B_fits_id)

    save_pipeline(pipe)

    print("Compute the spectrum to check if everything is fine")
    print("Computing the spectrum of both files")
    log_A, meta_A, source_A = load_source_data(meta_A, pp)
    log_B, meta_B, source_B = load_source_data(meta_B, pp)

    ensure_pipeline_folders_exists(pipe)
    # Show that the computed spectrum matches the actual data
    plot_spectrum_vs_data(
        meta_A=meta_A,
        source_A=source_A,
        meta_B=meta_B,
        source_B=source_B,
        pp=pp,
        filename=f"plots/{pipe.id}/plot_spectrum_A_{meta_A.id}_vs_B_{meta_B.id}.png",
        show=False,
    )

    print("Saved spectrum comparison")


if __name__ == "__main__":
    main(use_defaults=True, generate=True)
