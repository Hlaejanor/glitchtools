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
    plot_pi_time_heatmap,
    plot_ccd_bin,
)
from common.fitsread import (
    fits_save_events_generated,
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
    add_time_binning,
    load_or_compute_chunk_variability_observations,
    load_source_data,
)


version = 1.0


pp_id = "default"
variability_types = np.array(
    [
        "Excess Variability",
        "Excess Variability Smoothed",
        "Fano Excess Local Variability",
        "Fano Excess Global Variability",
        "Variability Excess Adjacent",
        "Variability Excess Smoothed Adjacent",
        "Odd even contrast",
    ]
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare two metadata files and optionally set downsampling."
    )
    parser.add_argument(
        "-pp",
        "--pp",
        type=str,
        default=pp_id,
        help="Select the processing pipeline",
    )

    args = parser.parse_args()
    return args


def get_variability_type(hash: str, df: DataFrame):
    cols = [
        "date",
        "pp_hash",
        "pipeline_id",
        "variability_type",
        "A",
        "B",
        "threshold_q",
        "null_n",
        "antares_m",
        "exceedances_x",
        "exceed_rate_p_hat",
        "null_rate_p0",
        "enrichment_p_hat_over_p0",
        "p_value_binomial",
        "ci95_p_hat_low",
        "ci95_p_hat_high",
        "comparing_to_self",
        "significant",
    ]

    hash_controls = [hash]
    positive_controls = ["anisogen2_vs_rnd_anisogen2", "antares_vs_rnd_antares"]

    test_controls = ["antares_vs_rnd_antares"]
    for c in cols:
        assert c in df.columns, f"Require {c} in columns"

    sig = df["significant"].astype(bool)
    pos_mask = df["pipeline_id"].isin(positive_controls)

    # Obtain the variability types that work in the positive control
    vartypes_pos = df.loc[(sig & pos_mask), "variability_type"].unique()
    negative_controls = ["rnd_anisogen2_vs_rnd_anisogen2", "rnd_antares_vs_rnd_antares"]
    same_as_pos_mask = df["variability_type"].isin(vartypes_pos)
    neg_mask = df["pipeline_id"].isin(negative_controls)
    hash_mask = df["pp_hash"].isin(hash_controls)

    use = ((sig & pos_mask) | ((~sig) & neg_mask)) & hash_mask & same_as_pos_mask

    return df.loc[use], vartypes_pos


def run_pipeline(pp: ProcessingParameters):
    try:
        csv_file = f"files/tail_exceedance_test_{pp.get_hash()}.csv"
        csv = read_from_csv(csv_file)

        table, accepted_vartypes = get_variability_type(pp.get_hash(), csv)
        print(
            f"The following var-types can be used as statistically significant",
            accepted_vartypes,
        )
        latex_table_filename = f"files/statistical_test.tex"
        print(table.columns)
        columns_keep = [
            "A",
            "B",
            "variability_type",
            "p_value_binomial",
            "threshold_q",
            "exceedances_x",
        ]
        columns_labels = [
            "A",
            "B",
            "Variability metric",
            "p-value (binomial)",
            "Threshold",
            "Exceedances",
        ]
        print(columns_keep)
        write_as_latex_table(table, columns_keep, columns_labels, latex_table_filename)

        print(f"Finished summarizing runs using  {pp.id} processing parameters")
    except Exception as e:
        print(f"ERROR in summarizing runs using  {pp.id}, aborting")
        print(e)
        raise e
    return []


def main():
    log = []
    print("Lightlanefinder")
    args = parse_arguments()

    if args.pp is None:
        raise Exception(f"Cannot process, need pp")

    pp = load_processing_param(args.pp)
    run_pipeline(pp)


if __name__ == "__main__":
    main()
