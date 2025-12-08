import sys
import os
import numpy as np
from scipy.stats import binomtest
from common.generate_data import generate_synthetic_telescope_data
import argparse
import numpy as np
import datetime

from pandas import DataFrame
from event_processing.plotting import (
    compute_time_variability_async,
    take_max_variability_per_wbin,
    take_top_wavelengths_per_timescale,
    plot_top_variability_timescales,
    plot_wbin_time_heatmap,
    plot_ccd_bin,
)
from common.fitsread import (
    fits_save_events_with_pi_channel,
    read_event_data_crop_and_project_to_ccd,
    fits_save_chunk_analysis,
    fits_read,
    get_cached_filename,
    fits_read_cache_if_exists,
    fits_save_cache,
)
from event_processing.detectbands import (
    analyze_bands,
    find_band_times,
    lomb_scargle_periodogram,
    test_rayleigh,
)
from common.helper import (
    ensure_pipeline_folders_exists,
    randomly_sample_from,
    get_duration,
    write_result_to_csv,
    read_from_csv,
    write_as_latex_table,
)
from common.fitsmetadata import (
    FitsMetadata,
    ChunkVariabilityMetadata,
    ProcessingParameters,
    ComparePipeline,
)
from common.metadatahandler import (
    load_fits_metadata,
    load_chunk_metadata,
    load_best_metadata,
    load_processing_param,
    load_pipeline,
    load_gen_param,
    save_chunk_metadata,
    save_pipeline,
)
from event_processing.var_analysis_plots import (
    make_standard_plots,
    SDMC_plot,
    get_uneven_time_bin_widths,
    plot_chunk_variability_excess,
)
from event_processing.binning import (
    get_binned_datasets,
    load_or_compute_chunk_variability_observations,
    load_source_data,
)


version = 1.0


pp_id = "default"
# Define the variability types to include in the table
variability_types = np.array(
    [
        #   "Excess Variability Smoothed",
        "Variability Excess Smoothed Adjacent",
        #   "Fano Excess Global Variability",
    ]
)

# Define the test control.s
test_controls = [
    "antares_vs_rnd_antares",  # Antares default time bins without flattening
    "antares_vs_rnd_antares_f",  # Antares flattened and default time bins
    "antares_vs_rnd_antares_fl",  # Antares flattened and longer duration time bins
    "antares_vs_rnd_antares_lt",  # Antares longer duration time bins and thinned
    "antares_vs_rnd_antares_t",  # Antares default time bins and thinned
]
# Define pipeline to ignore for the preparation of the table
ignore_pipelines = [
    "antares_vs_rnd_antares_357",
    "antares_vs_rnd_antares_f347",
    "antares_vs_rnd_antares_c",
    "rnd_isogen_vs_rnd_antares",  # Ignored because the counts in the spectrum is 19.5 and 14.5 respectively, which creates false positives for the bionomial tail enrichment test
    "antares_vs_isogen",  # Ignored because the counts in the spectrum is 19 and 14.5 respectively, which creates false positives for the bionomial tail enrichment test
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare two metadata files and optionally set downsampling."
    )
    parser.add_argument(
        "-p",
        "--pipe",
        type=str,
        default="default",
        help="The name of the pipe",
    )

    args = parser.parse_args()
    return args


def run_pipeline(pp: ProcessingParameters, outfile: str | None = None):
    try:
        assert pp is not None, "Processing parameters cannot be None"
        csv_file = f"files/statistical_tests.csv"
        csv = read_from_csv(csv_file)

    except Exception as e:
        print(f"ERROR in summarizing runs using  {pp.id}, aborting")
        print(e)
        raise e
    return []


def main():
    log = []
    print("Generate tables for statistical test results")
    args = parse_arguments()

    if args.pp is None:
        raise Exception(f"Cannot process, need pp")

    pp = load_processing_param(args.pp)
    run_pipeline(pp, outfile=args.tex)


if __name__ == "__main__":
    main()
