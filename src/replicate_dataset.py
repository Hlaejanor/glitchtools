import pandas as pd
import numpy as np
from common.helper import compare_dataframes, get_duration
from var_analysis.plotting import plot_spectrum_vs_data, compute_time_variability
from common.powerdensityspectrum import compute_spectrum_params, PowerDensitySpectrum
from common.metadatahandler import (
    load_fits_metadata,
    save_fits_metadata,
    load_gen_param,
    save_gen_param,
    load_pipeline,
    save_pipeline,
    save_processing_metadata,
)
from common.helper import compare_variability_profiles
from common.fitsread import (
    read_crop_and_project_to_ccd,
    fits_save_from_generated,
    fits_read,
    fits_save,
)

from common.fitsmetadata import (
    FitsMetadata,
    Spectrum,
    ProcessingParameters,
    ComparePipeline,
    GenerationParameters,
)
from common.helper import randomly_sample_from
from var_analysis.readandplotchandra import (
    binning_process,
    experiment_exists,
)
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
    default_pipeline = "fullfat"
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
    )
    save_processing_metadata(pparam)

    gen = GenerationParameters(
        id="test_1",
        alpha=3.0,
        lucretius=-1,
        r_e=0.8,
        theta=0.51,
        t_max=take_time,
        perp=0.0,
        phase=0.0,
        max_wavelength=1.0,
        min_wavelength=0.12,
        spectrum=antares_meta.apparent_spectrum,
        star="Antares gen",
    )

    save_gen_param(gen)
    if not pipeline:
        pipeline = ComparePipeline(
            id=pipeId,
            A_fits_id=antares_meta.id,
            B_fits_id=None,
            pp_id=pparam.id,
            gen_id=gen.id,
        )
    else:
        pipeline.A_fits_id = antares_meta.id
        pipeline.B_fits_id = None
        pipeline.pp_id = pparam.id
        pipeline.gen_id = gen.id

        save_pipeline(pipeline)


def downsample_antares(N):
    print("Loading file for downsampling ")
    A_meta = load_fits_metadata("antares0")
    A_events = fits_read(A_meta.raw_event_file)

    A_reduced = randomly_sample_from(A_events, N)
    A_meta.source_count = N
    A_meta.id = f"antares_{int(N/1000)}k"

    A_meta = fits_save(A_reduced, A_meta)
    save_fits_metadata(A_meta)


def main(use_defaults=False, generate: bool = True):
    print("Create mimic dataset")

    #  Ensure that the default pipeline is present
    # create_default_pipeline()

    # Parse command line arguments
    args = parse_arguments(use_defaults)

    # Load the default or specified pipeline
    pipeline_meta = load_pipeline(args.pipeline)
    assert (
        pipeline_meta is not None
    ), "Pipeline meta missing. Try to run create_default_pipeline.sh"
    # Load metadata for target set A
    A_meta = load_fits_metadata(pipeline_meta.A_fits_id)

    # Load the generation parameter from the pipeline
    B_gen = load_gen_param(pipeline_meta.gen_id)

    # Load the photon events
    A_events = fits_read(A_meta.raw_event_file)
    A_meta.source_count = A_events.shape[0]  # Update the source count
    save_fits_metadata(A_meta)  # Commit to disk

    # Preprocess events
    success, A_meta, pp, A_processed_events = read_crop_and_project_to_ccd(
        pipeline_meta.A_fits_id, pipeline_meta.pp_id
    )
    # Bin the events
    A_binned, A_meta = binning_process(A_processed_events, A_meta, pp)

    # Plot charts for the binned data
    filename = f"plot_spectrum_{A_meta.id}.png"
    plot_spectrum_vs_data(
        meta_A=A_meta,
        binned_data_A=A_binned,
        processing_params=pp,
        filename=filename,
        show=True,
    )

    # Determine the duration for the processing pipeline
    A_duration = get_duration(A_meta, pp)

    # Compute the variability data
    A_time_variability_data = compute_time_variability(
        source_data=A_binned, duration=A_duration
    )

    # Ensure that the genreration parameters uses the same specturm as the target dataset
    B_gen.spectrum = A_meta.apparent_spectrum

    # Ensure that we don't generate more data than necessary
    B_gen.t_max = A_duration
    # Save the generation params
    save_gen_param(B_gen)

    if generate:
        # Generate a synthetic event file using B_gen, which include the spectrum and duration
        B_events = generate_synthetic_telescope_data(B_gen)
        # Save the the generated dataset as a fits file
        B_meta = fits_save_from_generated(B_events, B_gen)

        # Update the pipeline with reference to the new fits file
        pipeline_meta.B_fits_id = B_meta.id

        # Save the pipeline
        save_pipeline(pipeline_meta)
    else:
        # If we skip generation, load the pipeline metadata
        B_meta = load_fits_metadata(pipeline_meta.B_fits_id)

    # Load the fits events from disk
    B_events = fits_read(B_meta.raw_event_file)
    print(f"A (raw) : {len(A_binned)}, B (raw) {len(B_events)}")
    b_events = len(B_events)
    a_events = len(A_binned)
    if b_events / a_events < 0.8:
        print(f" Created a dataset that was {b_events/a_events}")
        # raise Exception(
        #    "Error, the generated databset should have the same amount of events"
        # )

    # Compare the dataframes
    compare_dataframes(A_events, B_events, "A", "B", True)

    # Crop the aprocess synth dataaset
    success, b_meta, b_pp, B_processed_events = read_crop_and_project_to_ccd(
        pipeline_meta.B_fits_id, pipeline_meta.pp_id
    )
    # Compare processeded dataframes
    compare_dataframes(A_processed_events, B_processed_events, "A", "B", True)
    # synth_reduced = randomly_sample_from(synth_data, downsample_target)

    print(f"Counts A {len(A_processed_events)}, B {len(B_processed_events)}")

    # Guard against possibility of processing parameters varying beteen A and B
    b_dict = b_pp.dict()
    a_dict = pp.dict()
    for key in a_dict.keys():
        if b_dict[key] != a_dict[key]:
            raise Exception("Processing parameters must be the same")

    print(
        f"A (processed) : {len(A_processed_events)}, B (processed) {len(B_processed_events)}"
    )
    # Bin B
    B_binned, B_meta = binning_process(B_processed_events, b_meta, b_pp)

    # Plot a comparison chart between the real and generated datasets
    filename = f"plot_spectrum_{A_meta.id}_{B_meta.id}.png"
    plot_spectrum_vs_data(
        meta_A=A_meta,
        binned_data_A=A_binned,
        meta_B=B_meta,
        binned_data_B=B_binned,
        processing_params=pp,
        filename=filename,
        show=True,
    )
    # Compare to real data
    B_duration = get_duration(B_meta, pp)

    # Compute time-variability
    B_time_variability_data = compute_time_variability(
        source_data=B_binned, duration=B_duration
    )

    # Compare the variability differences
    summary = compare_variability_profiles(
        -A_time_variability_data,
        B_time_variability_data,
        A_meta,
        b_meta,
        A_gen=None,
        B_gen=B_gen,
    )


if __name__ == "__main__":
    main(use_defaults=True, generate=True)
