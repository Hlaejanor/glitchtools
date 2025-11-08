import pandas as pd
import numpy as np
import gzip
import shutil
from common.fitsread import generate_metadata_from_fits
from common.helper import compare_dataframes, get_duration
from event_processing.plotting import (
    plot_spectrum_vs_data,
    compute_time_variability_async,
)
from common.powerdensityspectrum import compute_spectrum_params, PowerDensitySpectrum
from common.metadatahandler import (
    load_fits_metadata,
    save_fits_metadata,
    load_processing_param,
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
from event_processing.var_analysis_plots import (
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


def parse_arguments():
    parser = argparse.ArgumentParser(description="File")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default="antares_chandra/15734/primary/hrcf15734N003_evt2.fits",
        help="Path to the default fits file",
    )

    parser.add_argument(
        "-n",
        "--name",
        required=True,
        type=str,
        help="Name or handle for this dataset",
    )

    args = parser.parse_args()
    return args


def decompress_gz_file(gz_path):
    if not gz_path.endswith(".gz"):
        raise ValueError("Expected a .gz file for decompression.")

    fits_path = gz_path[:-3]
    print(f"Decompressing {gz_path} -> {fits_path}")
    with gzip.open(gz_path, "rb") as f_in, open(fits_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return fits_path


def main(generate: bool = True):
    print("Create default metadata")

    #  Ensure that the default pipeline is present
    args = parse_arguments()
    try:
        if not os.path.isfile(args.file):
            raise Exception(f"File {args.file} does not exist")

        if args.file.endswith(".fits"):
            file = args.file
        elif args.file.endswith(".fits.gz"):
            file = decompress_gz_file(args.file)
            print(f"Decompressed file, now using {file}")
        else:
            raise ValueError("Unsupported file format. Must be .fits or .fits.gz")

        default_metadata = generate_metadata_from_fits(file, args.name, False)
    except Exception as e:
        print(f"Error when creating metadata. Full error {repr(e)}")
        sys.exit()

    pp = load_processing_param("default")
    assert pp is not None, "Processing parameters cannot be None"
    save_fits_metadata(meta=default_metadata)

    source_data = fits_read(default_metadata.raw_event_file)

    print("Estimate spectrum")

    (
        success,
        default_metadata,
        pp,
        A_processed_events,
    ) = read_event_data_crop_and_project_to_ccd(default_metadata.id, pp.id)

    if not success:
        raise Exception("Error when corpping and project")
    binned_data, default_metadata = binning_process(
        A_processed_events, default_metadata, pp=pp
    )

    print(f"Successfully created the metadata for event file {file}")


if __name__ == "__main__":
    main(True)
