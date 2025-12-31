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
        "Variability Excess Adjacent",
        "Variability Excess Smoothed Adjacent",
        #   "Fano Excess Global Variability",
    ]
)

# Define the test control.s
test_pipelines = [
    "antares_vs_rnd_antares",  # Antares default time bins without flattening
    "antares_vs_rnd_antares_f",  # Antares flattened and default time bins
    "antares_vs_rnd_antares_fl",  # Antares flattened and longer duration time bins
    "antares_vs_rnd_antares_lt",  # Antares longer duration time bins and thinned
    "antares_vs_rnd_antares_t",  # Antares default time bins and thinned
]
# Define pipeline to ignore for the preparation of the table
ignore_globally = [
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


def filter_and_group(
    hash: str,
    df: DataFrame,
    variability_types: list,
    group_by_cols: list = None,
    agg_types: list = ["mean", "median", "count"],
):
    if group_by_cols is None:
        group_by_cols = []

    # --- filtering ---
    use = df["variability_type"].isin(variability_types)
    filtered_df = df.loc[use]

    if not group_by_cols:
        return filtered_df

    numeric_cols = filtered_df.select_dtypes(include="number").columns.difference(
        group_by_cols
    )

    grouped = (
        filtered_df.groupby(group_by_cols, dropna=False)[numeric_cols]
        .agg(["mean", "median"])
        .reset_index()
    )

    # flatten MultiIndex columns: (empirical_q, mean) -> empirical_q_mean
    grouped.columns = [
        c[0]
        if isinstance(c, tuple) and c[1] == ""
        else f"{c[0]}_{c[1]}"
        if isinstance(c, tuple)
        else c
        for c in grouped.columns
    ]

    # one group count
    grouped["n"] = (
        filtered_df.groupby(group_by_cols, dropna=False).size().reset_index(drop=True)
    )

    return grouped


def p2a_table(
    csv: DataFrame,
    pp: ProcessingParameters,
    outfile: str,
    variability_types: list,
    caption: str,
    label: str = None,
    standing: bool = False,
):
    # Produce a list of the observations that belong to a variability type that
    # fulfills the criteria of specificy and sensitivity

    table = filter_and_group(
        pp.get_hash(),
        csv,
        variability_types=variability_types,
        group_by_cols=["pipeline_id", "p2a_verdict"],
        agg_types=["mean", "median"],
    )

    assert table.shape[0] > 0, "No observations types the pipe and var type criteria"

    table = table.sort_values(["pipeline_id"])

    columns_keep = [
        "pipeline_id",
        #  "variability_type",
        "p2_slope_A_median",
        # "p2_slope_B_median",
        "p2a_slope_p_median",
        "p2a_verdict",
        "n",
    ]

    columns_labels = [
        "Pipeline",
        # "Variability type",
        "Slope test",
        # "Slope null",
        "P2a significance",
        "P2b hardening",
        "Runs",
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
        standing=standing,
        order_by="p2a_slope_p_median",
    )

    print(f"Finished creating table {outfile} runs using {pp.id} processing parameters")



def p1d_table(
    csv: DataFrame,
    pp: ProcessingParameters,
    outfile: str,
    variability_types: list,
    caption: str,
    label: str = None,
    standing: bool = False,
):
    # Produce a list of the observations that belong to a variability type that
    # fulfills the criteria of specificy and sensitivity

    table = filter_and_group(
        pp.get_hash(),
        csv,
        variability_types=variability_types,
        group_by_cols=["pipeline_id", "p1d_verdict"],
        agg_types=["mean", "median"],
    )

    assert table.shape[0] > 0, "No observations types the pipe and var type criteria"

    table = table.sort_values(["pipeline_id"])

    columns_keep = [
        "pipeline_id",
        # "variability_type",
        "p1d_neg_test_peak_median",
        "p1d_pos_test_peak_median",
        "p1d_neg_null_peak_median",
        "p1d_pos_null_peak_median",
        "p1d_verdict",
        "n",
    ]

    columns_labels = [
        "Pipeline",
        # "Variability type",
        "Neg peak (A)",
        "Pos peak (A)",
        "Neg peak (B)",
        "Pos peak (B)",
        "Shifted peak",
        "Runs",
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
        standing=standing,
        order_by=None,
    )

    print(f"Finished creating table {outfile} runs using {pp.id} processing parameters")


def p1e_table(
    csv: DataFrame,
    pp: ProcessingParameters,
    outfile: str,
    variability_types: list,
    caption: str,
    label: str = None,
    standing: bool = False,
):
    # Produce a list of the observations that belong to a variability type that
    # fulfills the criteria of specificy and sensitivity

    table = filter_and_group(
        pp.get_hash(),
        csv,
        variability_types=variability_types,
        group_by_cols=["pipeline_id", "p1e_verdict"],
        agg_types=["mean", "median"],
    )

    assert table.shape[0] > 0, "No observations types the pipe and var type criteria"

    table = table.sort_values(["pipeline_id"])

    columns_keep = [
        "pipeline_id",
        # "variability_type",
        "p1e_mass_ratio_test_median",
        "p1e_mass_ratio_null_median",
        "p1e_verdict",
        "n",
    ]

    columns_labels = [
        "Pipeline",
        # "Variability type",
        "Pos/Neg Prob mass (A)",
        "Pos/Neg Prob mass (B)",
        "Skewness verdict",
        "Runs",
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
        standing=standing,
        order_by=None,
    )

    print(f"Finished creating table {outfile} runs using {pp.id} processing parameters")


def p1bc_table(
    csv: DataFrame,
    pp: ProcessingParameters,
    outfile: str,
    variability_types: list,
    caption: str,
    label: str = None,
    standing: bool = False,
):
    # Produce a list of the observations that belong to a variability type that
    # fulfills the criteria of specificy and sensitivity

    table = filter_and_group(
        pp.get_hash(),
        csv,
        variability_types=variability_types,
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
        standing=standing,
        order_by=None,
    )

    print(f"Finished creating table {outfile} runs using {pp.id} processing parameters")


def p1a_table(
    csv: DataFrame,
    pp: ProcessingParameters,
    outfile: str,
    variability_types: list,
    caption: str = "",
    label: str = None,
    standing: bool = False,
):
    table = filter_and_group(
        pp.get_hash(),
        csv,
        variability_types=variability_types,
        group_by_cols=["pipeline_id", "p1a_verdict"],
        agg_types=["mean", "median"],
    )

    assert table.shape[0] > 0, "No observations passed type pipeline and type criteria"

    table = table.sort_values(["p1a_p_value_binomial_mean"])
    print(table.columns)
    columns_keep = [
        "pipeline_id",
        # "variability_type",
        "p1a_p_value_binomial_median",
        "p1a_threshold_q_median",
        "p1a_exceedances_x_median",
        "p1a_verdict",
        "n",
    ]

    columns_labels = [
        "Pipeline",
        # "Variability type",
        "Median p (bin)",
        "Median Thres",
        "Median Excess.",
        "Verdict",
        "Runs",
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
        standing=standing,
        order_by="p1a_p_value_binomial_median",
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


def get_test_controls(csv: DataFrame, test_pipelines: list):
    mask = csv["empirical_q"] > 1e-100

    mask &= csv["pipeline_id"].isin(test_pipelines)

    return csv.loc[mask].copy()



def get_positive_controls(csv: DataFrame, ignore_pipelines: list | None = None) -> DataFrame:
    if ignore_pipelines is None:
        ignore_pipelines = []

    # A is NOT expected null, B IS expected null, and empirical_q is finite/nonzero-ish
    mask = (
        (~csv["A_expected_null"])
        & (csv["B_expected_null"])
        & (csv["empirical_q"] > 1e-100)
    )

    # Exclude ignored pipelines
    if ignore_pipelines:
        mask &= ~csv["pipeline_id"].isin(ignore_pipelines)

    return csv.loc[mask].copy()



def get_negative_controls(csv: DataFrame, ignore_pipelines: list = []):
    # Identify controls where both A or B are expected to have nulls
    filter = (csv["A_expected_null"]) & (
        csv["B_expected_null"] & csv["empirical_q"] > 1e-100
    )
    neg_df = csv.loc[filter]
    # neg_pipes = neg_df["pipeline_id"].unique().tolist()
    # neg_pipes_filtered = [p for p in neg_pipes if p not in ignore_pipelines]
    return neg_df


def run_pipeline(
    pp: ProcessingParameters,
    separate_table_per_var_type=True,
    outfile: str | None = None,
    standing: bool = False,
):
    try:
        assert pp is not None, "Processing parameters cannot be None"
        csv_file = f"files/baseliner.csv"
        csv = read_from_csv(csv_file)

        validate_csv_dataframe(csv)

        if outfile is None:
            outfile = "table"

        ignore = test_pipelines + ignore_globally
        #
        pos_df = get_positive_controls(csv, ignore_pipelines=ignore)

        neg_df = get_negative_controls(csv, ignore_pipelines=ignore_globally)

        test_df = get_test_controls(csv, test_pipelines)
        print(test_df["pipeline_id"].head(200))
        # Clever but hard to read nesting hack to make this optional
        if not separate_table_per_var_type:
            variability_set = [variability_types]
        else:
            variability_set = variability_types

        # For each variability type
        for vart in variability_set:
            var_sig = "".join([word[0] for word in vart.split()])
            vart = [vart]
            outfile = f"table_{var_sig}"

            # Positive control datasets. These should all show variability in dataset A
            p1a_table(
                pos_df,
                pp,
                outfile=f"tables/{outfile}_p1a_POS.tex",
                variability_types=vart,
                caption="P1a Statistical test results for positive controls. A is expected to contain significantly more than the 100 exceedances present in dataset B.",
                label=f"table:{var_sig}_p1a_POS",
                standing=standing,
            )
            p1bc_table(
                pos_df,
                pp,
                outfile=f"tables/{outfile}_p1bc_POS.tex",
                variability_types=vart,
                caption="P1b and P1c Variance test results for anisogen datasets. Verdicts are based on comparison of mean and median variance between test and null datasets.",
                label=f"table:{var_sig}_p1bc_POS",
                standing=standing,
            )
            p1d_table(
                pos_df,
                pp,
                outfile=f"tables/{outfile}_p1d_POS.tex",
                variability_types=vart,
                caption="P1d Skewness and P1e Mass Ratio test results for Antares dataset. Verdicts are based on comparison of skewness and mass ratio between test and null datasets.",
                label=f"table:{var_sig}_p1de_POS",
                standing=standing,
            )
            p1e_table(
                pos_df,
                pp,
                outfile=f"tables/{outfile}_p1e_POS.tex",
                variability_types=vart,
                caption="P1d Skewness and P1e Mass Ratio test results for Antares dataset. Verdicts are based on comparison of skewness and mass ratio between test and null datasets.",
                label=f"table:{var_sig}_p1de_POS",
                standing=standing,
            )
            p2a_table(
                pos_df,
                pp,
                outfile=f"tables/{outfile}_p2a_POS.tex",
                variability_types=vart,
                caption="P2a Hardening test results for the Positive controls. Probability of slope difference being due to chance. The slope in A is negative, consistent with the hardening encoded in the Anisogen generation parameters",
                label=f"table:{var_sig}_p2ab_POS",
                standing=standing,
            )
           

            # Negative control datasets. These should never show significant deviations
            p1a_table(
                neg_df,
                pp,
                outfile=f"tables/{outfile}_p1a_NEG.tex",
                variability_types=vart,
                caption="P1a Statistical test results for negative controls. A and B are both poissonized, and should not be significantly different.",
                label=f"table:{var_sig}_p1a_NEG",
                standing=standing,
            )
            p1bc_table(
                neg_df,
                pp,
                outfile=f"tables/{outfile}_p1bc_NEG.tex",
                variability_types=vart,
                caption="P1b and P1c Variance test results for fully randomized dataset. Verdicts are based on comparison of mean and median variance between test and null datasets.",
                label=f"table:{var_sig}_p1b_NEG",
                standing=standing,
            )
            p1d_table(
                neg_df,
                pp,
                outfile=f"tables/{outfile}_p1de_NEG.tex",
                variability_types=vart,
                caption="P1d Skewness and P1e Mass Ratio test results for Negative controls. Verdicts are based on comparison of skewness and mass ratio between test and null datasets.",
                label=f"table:{var_sig}_p1de_NEG",
                standing=standing,
            )
            p2a_table(
                neg_df,
                pp,
                outfile=f"tables/{outfile}_p2a_NEG.tex",
                variability_types=vart,
                caption="P2a Hardening test results for the Negative controls. High probability of slope difference being due to chance. The slope in A and B are both flat, consistent with no systematic variability correlation with wavelength.",
                label=f"table:{var_sig}_p2a_NEG",
                standing=standing,
            )
            

            # Test datasets
            p1a_table(
                test_df,
                pp,
                outfile=f"tables/{outfile}_p1a_TEST.tex",
                variability_types=vart,
                caption="P1a Statistical test results for Antares dataset. Statistically signficant exceedances in Antares dataset compared to control is evidence for flickering (excess temporal variability) in Antares dataset .",
                label=f"table:{var_sig}_p1a_TEST",
                standing=standing,
            )
            p1bc_table(
                test_df,
                pp,
                outfile=f"tables/{outfile}_p1bc_TEST.tex",
                variability_types=vart,
                caption="P1b and P1c Variance test results for Antares dataset. Verdicts are based on comparison of mean and median variance between test and null datasets.",
                label=f"table:{var_sig}_p1bc_TEST",
                standing=standing,
            )
            p1d_table(
                test_df,
                pp,
                outfile=f"tables/{outfile}_p1de_TEST.tex",
                variability_types=vart,
                caption="P1d Skewness and P1e Mass Ratio test results for Antares dataset. Verdicts are based on comparison of skewness and mass ratio between test and null datasets.",
                label=f"table:{var_sig}_p1de_TEST",
            )
            p2a_table(
                test_df,
                pp,
                outfile=f"tables/{outfile}_p2a_TEST.tex",
                variability_types=vart,
                caption="P2a Hardening test results for Antares dataset. Low probability of slope difference being due to chance, hardening is consistent with the predicted hardening",
                label=f"table:{var_sig}_p2ab_TEST",
                standing=standing,
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
    run_pipeline(pp, outfile=args.tex, standing=True)


if __name__ == "__main__":
    main()
