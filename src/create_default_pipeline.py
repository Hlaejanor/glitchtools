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


def create_default_pipeline():
    take_time = 10000
    pipeline = load_pipeline("default")

    antares_meta = FitsMetadata(
        id="default",
        raw_event_file="antares_chandra/15734/primary/hrcf15734N003_evt2.fits",
        source_pos_x=512,
        source_pos_y=512,
        star="Antares",
        synthetic=False,
        max_energy=11.0,
        min_energy=1,
        source_count=None,
        apparent_spectrum=None,
        t_max=None,
    )

    save_fits_metadata(antares_meta)

    assert antares_meta is not None, "Chandra meta needs to be called"

    pparam = ProcessingParameters(
        id="default",
        resolution=1024,
        processed_filename=None,
        source_radius=512,
        max_wavelength=1.0,
        min_wavelength=0.1,
        wavelength_bins=100,
        time_bin_seconds=100,
        take_time_seconds=take_time,
        anullus_radius_inner=None,
        anullus_radius_outer=None,
    )
    save_processing_metadata(pparam)

    gen = GenerationParameters(
        id="default",
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
            id="default",
            A_fits_id=antares_meta.id,
            B_fits_id=None,
            pp_id=pparam.id,
            gen_id=gen.id,
        )
    else:
        pipeline.id = "default"
        pipeline.A_fits_id = antares_meta.id
        pipeline.B_fits_id = None
        pipeline.pp_id = pparam.id
        pipeline.gen_id = gen.id

    save_pipeline(pipeline)


def downsample_antares(N):
    print("Loading file for downsampling ")
    A_meta = load_fits_metadata("default")
    A_events = fits_read(A_meta.raw_event_file)

    A_reduced = randomly_sample_from(A_events, N)
    A_meta.source_count = N
    A_meta.id = f"antares_{int(N/1000)}k"

    A_meta = fits_save(A_reduced, A_meta)
    save_fits_metadata(A_meta)


def main(generate: bool = True):
    print("Create default pipeline")

    #  Ensure that the default pipeline is present
    create_default_pipeline()

    print(
        "Successfully created the default pipeline. You can inspect it in meta_files/pipelines/"
    )


if __name__ == "__main__":
    main(True)
