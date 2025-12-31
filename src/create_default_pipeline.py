import pandas as pd
import numpy as np
from common.helper import compare_dataframes, get_duration
from event_processing.plotting import (
    plot_spectrum_vs_data,
    compute_time_variability_async,
)
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
    read_event_data_crop_and_project_to_ccd,
    fits_save_events_with_pi_channel,
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

from common.generate_data import generate_synthetic_telescope_data
import csv
import os
import sys
import itertools
import uuid
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import argparse


def create_default_pipeline():
    print("creating deffault pipepline")
    take_time = None
    pipeline = load_pipeline("default")

    antares_meta = FitsMetadata(
        id="antares",
        raw_event_file="antares_chandra/15734/primary/hrcf15734N003_evt2.fits",
        source_pos_x=512,
        source_pos_y=512,
        star="Antares",
        synthetic=False,
        gen_id="default",
        max_energy=11.0,
        min_energy=1,
        source_count=None,
        apparent_spectrum=None,
        t_max=None,
        ascore=None,
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
        variability_percentile=None,
        time_bins_from=20,
        time_bins_to=300,
        phase_bins=30,
        time_bin_widths_count=30,
        time_bin_chunk_length=None,
        start_time_seconds=0,
        end_time_seconds=take_time,
        anullus_radius_inner=None,
        anullus_radius_outer=None,
        padding_strategy=None,
        downsample_strategy="flatten",
        downsample_target_count=None,
        variability_type="Fano Excess Local Variability",
    )
    save_processing_metadata(pparam)

    pparam = ProcessingParameters(
        id="randomizer",
        resolution=1024,
        processed_filename=None,
        source_radius=512,
        max_wavelength=1.0,
        min_wavelength=0.1,
        wavelength_bins=100,
        time_bin_seconds=100,
        variability_percentile=None,
        time_bins_from=20,
        time_bins_to=300,
        time_bin_widths_count=30,
        phase_bins=30,
        time_bin_chunk_length=None,
        start_time_seconds=0,
        end_time_seconds=take_time,
        anullus_radius_inner=None,
        anullus_radius_outer=None,
        padding_strategy=None,
        downsample_strategy="flatten",
        downsample_target_count=None,
        variability_type="Fano Excess Local Variability",
    )
    save_processing_metadata(pparam)

    isogen = GenerationParameters(
        id="isogen",
        alpha=0.5,
        lucretius=-2,
        theta_change_per_sec=0.0,
        r_e=0.8,
        theta=0.51,
        t_max=take_time,
        perp=0.0,
        phase=0.0,
        max_wavelength=1.0,
        min_wavelength=0.12,
        spectrum=antares_meta.apparent_spectrum,
        star="Antares iso gen",
    )
    save_gen_param(isogen)

    anisogen = GenerationParameters(
        id="anisogen",
        alpha=5.0,
        velocity=1.0,
        lucretius=-2,
        theta_change_per_sec=0.000001,
        r_e=0.8,
        theta=0.51,
        t_max=take_time,
        perp=0.0,
        phase=0.0,
        max_wavelength=1.0,
        min_wavelength=0.12,
        spectrum=antares_meta.apparent_spectrum,
        star="Synth",
    )
    save_gen_param(anisogen)

    # Select A using an Id, flatten it. Then copy it, and poissonize it to get B. Then save B.
    # Generate A using GenId, flatten it and then poissonize it. Then copy it and poissonize it again to get B.
    # Should be more similar than antares_vs_rnd_antares

    pipeline1 = ComparePipeline(
        id="antares_vs_isogen",
        source=antares_meta.id,
        A_fits_id="antares_f",
        B_fits_id="isogen_f",
        A_tasks=["chandrashift", "flatten", "chandrashift"],
        B_tasks=["generate", "flatten"],
        pp_id=pparam.id,
        gen_id=isogen.id,
    )
    save_pipeline(pipeline1)

    # The control that proves that the poissonization is working as intended
    pipeline2 = ComparePipeline(
        id="rnd_antares_vs_rnd_antares",
        source=antares_meta.id,
        A_fits_id="antares_hp0",
        B_fits_id="antares_hp1",
        A_tasks=["chandrashift", "flatten", "poissonize"],
        B_tasks=["copy", "poissonize"],
        pp_id=pparam.id,
        gen_id=isogen.id,
    )
    save_pipeline(pipeline2)

    # The second control that proves that the poissonization is working as intended
    pipeline3 = ComparePipeline(
        id="rnd_anisogen_vs_rnd_anisogen",
        A_fits_id="isogen_hp0",
        B_fits_id="isogen_hp1",
        A_tasks=["generate", "flatten", "poissonize"],
        B_tasks=["copy", "poissonize"],
        pp_id=pparam.id,
        gen_id=anisogen.id,
    )

    save_pipeline(pipeline3)

    pipeline4 = ComparePipeline(
        id="anisogen_vs_rnd_anisogen",
        source=None,
        A_fits_id="anisogen",
        B_fits_id="anisogen_hp",
        A_tasks=["generate", "flatten"],
        B_tasks=["copy", "poissonize"],
        pp_id=pparam.id,
        gen_id=anisogen.id,
    )

    save_pipeline(pipeline4)


def main(generate: bool = True):
    print("Create default pipeline")

    #  Ensure that the default pipeline is present
    create_default_pipeline()

    print(
        "Successfully created the default pipeline. You can inspect it in meta_files/pipelines/"
    )


if __name__ == "__main__":
    main(True)
