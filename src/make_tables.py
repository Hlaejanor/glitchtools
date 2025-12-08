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
        # "Excess Variability Smoothed",
        "Variability Excess Adjacent"
        # "Variability Excess Smoothed Adjacent",
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
        "-pp",
        "--pp",
        type=str,
        default="default",
        help="The name of the processing parameters used",
    )

    parser.add_argument(
        "-t",
        "--tex",
        type=str,
        default="table",
        help="The name of the output latex table file",
    )
    args = parser.parse_args()
    return args


def filter_by_pipe_and_variability_type(
    hash: str, df: DataFrame, pipes: list, variability_types: list
):
    in_reportable_pipes = df["pipeline_id"].isin(pipes)
    in_allowed_var_types = df["variability_type"].isin(variability_types)

    use = (in_reportable_pipes) & (in_allowed_var_types)
    return df.loc[use]


def p2ab_table(
    csv: DataFrame,
    pp: ProcessingParameters,
    outfile: str,
    pipes: list,
    variability_types: list,
    caption: str,
    label: str = None,
    standing: bool = False,
):
    # Produce a list of the observations that belong to a variability type that
    # fulfills the criteria of specificy and sensitivity
    table = filter_by_pipe_and_variability_type(
        pp.get_hash(), csv, variability_types=variability_types, pipes=pipes
    )

    assert table.shape[0] > 0, "No observations types the pipe and var type criteria"

    table = table.sort_values(["pipeline_id"])

    columns_keep = [
        "pipeline_id",
        #  "variability_type",
        "p2_slope_A",
        "p2_slope_B",
        "p2a_slope_p",
        "p2a_verdict",
        "p2b_verdict",
    ]

    columns_labels = [
        "Pipeline",
        # "Variability type",
        "Slope test",
        "Slope null",
        "Diff p-value",
        "P2a significance",
        "P2b hardening",
    ]
    if len(variability_types) > 1:
        columns_keep.append("variability_type")
        columns_labels.append("Variability type")
    else:
        caption += f" ({variability_types[0]})"

    write_as_latex_table(
        df=table,
        columns_keep=columns_keep,
        column_labels=columns_labels,
        filename=outfile,
        caption=caption,
        label=label,
    )

    print(f"Finished creating table {outfile} runs using {pp.id} processing parameters")


def p1de_table(
    csv: DataFrame,
    pp: ProcessingParameters,
    outfile: str,
    pipes: list,
    variability_types: list,
    caption: str,
    label: str = None,
    standing: bool = False,
):
    # Produce a list of the observations that belong to a variability type that
    # fulfills the criteria of specificy and sensitivity
    table = filter_by_pipe_and_variability_type(
        pp.get_hash(), csv, variability_types=variability_types, pipes=pipes
    )

    assert table.shape[0] > 0, "No observations types the pipe and var type criteria"

    table = table.sort_values(["pipeline_id"])

    columns_keep = [
        "pipeline_id",
        # "variability_type",
        "p1d_neg_test_peak",
        "p1d_pos_test_peak",
        "p1d_neg_null_peak",
        "p1d_pos_null_peak",
        "p1d_verdict",
        "p1e_mass_ratio_test",
        "p1e_mass_ratio_null",
        "p1e_verdict",
    ]

    columns_labels = [
        "Pipeline",
        # "Variability type",
        "Neg peak (A)",
        "Pos peak (A)",
        "Neg peak (B)",
        "Pos peak (B)",
        "Shifted peak",
        "Pos/Neg Prob mass (A)",
        "Pos/Neg Prob mass (B)",
        "Skewness verdict",
    ]
    if len(variability_types) > 1:
        columns_keep.append("variability_type")
        columns_labels.append("Variability type")
    else:
        caption += f" ({variability_types[0]})"

    write_as_latex_table(
        df=table,
        columns_keep=columns_keep,
        column_labels=columns_labels,
        filename=outfile,
        caption=caption,
        label=label,
    )

    print(f"Finished creating table {outfile} runs using {pp.id} processing parameters")


def p1bc_table(
    csv: DataFrame,
    pp: ProcessingParameters,
    outfile: str,
    pipes: list,
    variability_types: list,
    caption: str,
    label: str = None,
    standing: bool = False,
):
    # Produce a list of the observations that belong to a variability type that
    # fulfills the criteria of specificy and sensitivity
    table = filter_by_pipe_and_variability_type(
        pp.get_hash(), csv, variability_types=variability_types, pipes=pipes
    )

    assert table.shape[0] > 0, "No observations types the pipe and var type criteria"

    table = table.sort_values(["pipeline_id"])

    columns_keep = [
        "pipeline_id",
        #   "variability_type",
        "p1b_test_mean_var",
        "p1b_null_mean_var",
        "p1b_verdict_mean",
        "p1c_test_median_var",
        "p1c_null_median_var",
        "p1c_verdict_median",
    ]

    columns_labels = [
        "Pipeline",
        #  "Variability type",
        "Mean var A",
        "Mean var B",
        "Mean",
        "Median var A",
        "Median var B ",
        "Median",
    ]

    if len(variability_types) > 1:
        columns_keep.append("variability_type")
        columns_labels.append("Variability type")
    else:
        caption += f" ({variability_types[0]})"

    write_as_latex_table(
        df=table,
        columns_keep=columns_keep,
        column_labels=columns_labels,
        filename=outfile,
        caption=caption,
        label=label,
    )

    print(f"Finished creating table {outfile} runs using {pp.id} processing parameters")


def p1a_table(
    csv: DataFrame,
    pp: ProcessingParameters,
    outfile: str,
    pipes: list,
    variability_types: list,
    caption: str = "",
    label: str = None,
    standing: bool = False,
):
    assert (
        len(pipes) > 0
    ), "Need to provide at least one pipeline to include in the table"
    table = filter_by_pipe_and_variability_type(
        pp.get_hash(), csv, pipes=pipes, variability_types=variability_types
    )

    assert table.shape[0] > 0, "No observations passed type pipeline and type criteria"

    table = table.sort_values(["variability_type", "p1a_p_value_binomial"])

    columns_keep = [
        "pipeline_id",
        # "variability_type",
        "p1a_p_value_binomial",
        "p1a_threshold_q",
        "p1a_exceedances_x",
        "p1a_verdict",
    ]

    columns_labels = [
        "Pipeline",
        # "Variability type",
        "p (binomial)",
        "Threshold",
        "Exceedances",
        "Verdict",
    ]

    if len(variability_types) > 1:
        columns_keep.append("variability_type")
        columns_labels.append("Variability type")
    else:
        caption += f" ({variability_types[0]})"
    print(columns_keep)

    write_as_latex_table(
        df=table,
        columns_keep=columns_keep,
        column_labels=columns_labels,
        filename=outfile,
        caption=caption,
        label=label,
    )

    print(f"Finished summarizing runs using  {pp.id} processing parameters")


def validate_csv_dataframe(df: DataFrame):
    cols = [
        "date",
        "pp_hash",
        "pipeline_id",
        "variability_type",
        "A",
        "B",
        "A_expected_null",
        "B_expected_null",
        "pp",
        "date_run",
        "p1a_threshold_q",
        "p1a_null_n",
        "p1a_test_m",
        "p1a_exceedances_x",
        "p1a_exceed_rate_p_hat",
        "p1a_null_rate_p0",
        "p1a_enrichment_p_hat_over_p0",
        "p1a_p_value_binomial",
        "p1a_ci95_p_hat",
        "p1a_verdict",
        "p1b_test_mean_var",
        "p1b_null_mean_var",
        "p1c_test_median_var",
        "p1c_null_median_var",
        "p1b_verdict_mean",
        "p1c_verdict_median",
        "p1d_neg_test_peak",
        "p1d_pos_test_peak",
        "p1d_neg_null_peak",
        "p1d_pos_null_peak",
        "p1d_verdict",
        "p1e_mass_ratio_test",
        "p1e_mass_ratio_null",
        "p1e_verdict",
        "p2_slope_A",
        "p2_slope_B",
        "p2a_slope_p",
        "p2a_verdict",
        "p2b_verdict",
    ]

    for c in cols:
        assert c in df.columns, f"Require {c} in columns"


def get_positive_controls(
    csv: DataFrame, test_controls: list, ignore_pipelines: list = []
):
    # Identify controls where A is does not have nulls but B is
    filter = (~csv["A_expected_null"]) & (csv["B_expected_null"])
    positive_controls = csv.loc[filter]
    pos_pipes = positive_controls["pipeline_id"].unique().tolist()

    pos_pipes_filtered = [p for p in pos_pipes if p not in test_controls]
    pos_pipes_filtered = [p for p in pos_pipes_filtered if p not in ignore_pipelines]
    return pos_pipes_filtered


def get_negative_controls(csv: DataFrame, ignore_pipelines: list = []):
    # Identify controls where both A or B are expected to have nulls
    filter = (csv["A_expected_null"]) & (csv["B_expected_null"])
    negative_controls = csv.loc[filter]
    neg_pipes = negative_controls["pipeline_id"].unique().tolist()
    neg_pipes_filtered = [p for p in neg_pipes if p not in ignore_pipelines]
    return neg_pipes_filtered


def run_pipeline(pp: ProcessingParameters, outfile: str | None = None):
    try:
        assert pp is not None, "Processing parameters cannot be None"
        csv_file = f"files/statistical_tests.csv"
        csv = read_from_csv(csv_file)

        validate_csv_dataframe(csv)

        if outfile is None:
            outfile = "table"

        # List of pipelines
        positive_controls = get_positive_controls(
            csv, test_controls, ignore_pipelines=ignore_pipelines
        )
        negative_controls = get_negative_controls(
            csv, ignore_pipelines=ignore_pipelines
        )

        # Positive control datasets. These should all show variability in dataset A
        p1a_table(
            csv,
            pp,
            outfile=f"tables/{outfile}_p1a_POS.tex",
            pipes=positive_controls,
            variability_types=variability_types,
            caption="P1a Statistical test results for positive controls. A is expected to contain significantly more than the 100 exceedances present in dataset B.",
            label="table:p1a_POS.tex",
            standing=True,
        )
        p1bc_table(
            csv,
            pp,
            outfile=f"tables/{outfile}_p1bc_POS.tex",
            pipes=positive_controls,
            variability_types=variability_types,
            caption="P1b and P1c Variance test results for anisogen datasets. Verdicts are based on comparison of mean and median variance between test and null datasets.",
            label="table:p1bc_POS.tex",
            standing=True,
        )
        p1de_table(
            csv,
            pp,
            outfile=f"tables/{outfile}_p1de_POS.tex",
            pipes=positive_controls,
            variability_types=variability_types,
            caption="P1d Skewness and P1e Mass Ratio test results for Antares dataset. Verdicts are based on comparison of skewness and mass ratio between test and null datasets.",
            label="table:p1de_POS.tex",
            standing=True,
        )
        p2ab_table(
            csv,
            pp,
            outfile=f"tables/{outfile}_p2ab_POS.tex",
            pipes=positive_controls,
            variability_types=variability_types,
            caption="P2a and P2b Hardening test results for the Positive controls. Probability of slope difference being due to chance. The slope in A is negative, consistent with the hardening encoded in the Anisogen generation parameters",
            label="table:p2ab_POS.tex",
            standing=True,
        )

        # Negative control datasets. These should never show significant deviations
        p1a_table(
            csv,
            pp,
            outfile=f"tables/{outfile}_p1a_NEG.tex",
            pipes=negative_controls,
            variability_types=variability_types,
            caption="P1a Statistical test results for negative controls. A and B are both poissonized, and should not be significantly different.",
            label="table:p1a_NEG.tex",
            standing=True,
        )
        p1bc_table(
            csv,
            pp,
            outfile=f"tables/{outfile}_p1bc_NEG.tex",
            pipes=negative_controls,
            variability_types=variability_types,
            caption="P1b and P1c Variance test results for fully randomized dataset. Verdicts are based on comparison of mean and median variance between test and null datasets.",
            label="table:p1b_NEG.tex",
            standing=True,
        )
        p1de_table(
            csv,
            pp,
            outfile=f"tables/{outfile}_p1de_NEG.tex",
            pipes=negative_controls,
            variability_types=variability_types,
            caption="P1d Skewness and P1e Mass Ratio test results for Negative controls. Verdicts are based on comparison of skewness and mass ratio between test and null datasets.",
            label="table:p1de_NEG.tex",
            standing=True,
        )
        p2ab_table(
            csv,
            pp,
            outfile=f"tables/{outfile}_p2ab_NEG.tex",
            pipes=negative_controls,
            variability_types=variability_types,
            caption="P2a and P2b Hardening test results for the Negative controls. High probability of slope difference being due to chance. The slope in A and B are both flat, consistent with no systematic variability correlation with wavelength.",
            label="table:p2ab_NEG.tex",
            standing=True,
        )

        # Test datasets
        p1a_table(
            csv,
            pp,
            outfile=f"tables/{outfile}_p1a_TEST.tex",
            pipes=test_controls,
            variability_types=variability_types,
            caption="P1a Statistical test results for Antares dataset. Statistically signficant exceedances in Antares dataset compared to control is evidence for flickering (excess temporal variability) in Antares dataset .",
            label="table:p1a_TEST.tex",
            standing=True,
        )
        p1bc_table(
            csv,
            pp,
            outfile=f"tables/{outfile}_p1bc_TEST.tex",
            pipes=test_controls,
            variability_types=variability_types,
            caption="P1b and P1c Variance test results for Antares dataset. Verdicts are based on comparison of mean and median variance between test and null datasets.",
            label="table:p1bc_TEST.tex",
            standing=True,
        )
        p1de_table(
            csv,
            pp,
            outfile=f"tables/{outfile}_p1de_TEST.tex",
            pipes=test_controls,
            variability_types=variability_types,
            caption="P1d Skewness and P1e Mass Ratio test results for Antares dataset. Verdicts are based on comparison of skewness and mass ratio between test and null datasets.",
            label="table:p1de_TEST.tex",
        )
        p2ab_table(
            csv,
            pp,
            outfile=f"tables/{outfile}_p2ab_TEST.tex",
            pipes=test_controls,
            variability_types=variability_types,
            caption="P2a and P2b Hardening test results for Antares dataset. Low probability of slope difference being due to chance, hardening is consistent with the predicted hardening",
            label="table:p2ab_TEST.tex",
            standing=True,
        )
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
