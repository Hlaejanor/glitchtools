import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import binomtest
from scipy.stats import norm
from common.generate_data import generate_synthetic_telescope_data
import argparse
import numpy as np
import datetime


from pandas import DataFrame
from event_processing.plotting import (
    compute_time_variability_async,
    take_max_variability_per_wbin,
    take_top_wavelengths_per_timescale,
    take_max_variability,
    plot_top_variability_timescales,
    plot_wbin_time_heatmap,
    plot_ccd_bin,
    plot_wavelength_counts_histogram,
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
from event_processing.plotting import (
    plot_chunk_variability_excess,
    plot_chunk_variability_shape,
)
from common.helper import (
    ensure_pipeline_folders_exists,
    randomly_sample_from,
    get_duration,
    write_result_to_csv,
    get_wavelength_bins,
    ensure_path_exists,
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
)
from event_processing.binning import (
    get_binned_datasets,
    load_or_compute_chunk_variability_observations,
    load_source_data,
)


version = 1.0
"""
TLD;DR This script runs the variability analysis, and is responsible for calling the plotting functions

The variability analysis

"""

defaultpipe = "rnd_antares_vs_rnd_antares"
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
        "-c",
        "--chunk",
        type=str,
        default="false",
        help="Generate chunk data",
    )
    parser.add_argument(
        "-pl",
        "--plot",
        type=str,
        default="true",
        help="--plot true for plotting, otherwise will not plot ",
    )

    parser.add_argument(
        "-p",
        "--pipe",
        type=str,
        default=defaultpipe,
        help="Select the experiment pipe name ",
    )
    """
    parser.add_argument(
        "-a",
        "--meta_a",
        type=str,
        default=None,
        help="Path to first metadata JSON file (default: meta_1.json)",
    )
    parser.add_argument(
        "-b",
        "--meta_b",
        type=str,
        default=None,
        help="Path to B fits-file JSON file (default: best synthetic metadata)",
    )
    """
    parser.add_argument(
        "-pp",
        "--pp",
        type=str,
        default=None,
        help="Processing Params Processing (default)",
    )

    parser.add_argument(
        "-g",
        "--gen",
        type=str,
        default=None,
        help="Generation Parameters (default to best)",
    )

    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        default=None,
        help="Number of samples for downsampling (default: 100000)",
    )

    args = parser.parse_args()
    return args


def get_variability_peaks(test_vals, null_vals, bins=100):
    # Combine ranges
    min_edge = min(test_vals.min(), null_vals.min())
    max_edge = max(test_vals.max(), null_vals.max())

    # Create bins and centers
    variability_bins = np.linspace(min_edge, max_edge, bins + 1)
    bin_centers = 0.5 * (variability_bins[:-1] + variability_bins[1:])

    # Histogram both datasets
    test_hist, _ = np.histogram(test_vals, bins=variability_bins)
    null_hist, _ = np.histogram(null_vals, bins=variability_bins)

    # Separate negative and positive masks
    neg_mask = bin_centers < 0
    pos_mask = bin_centers >= 0

    # --- Test dataset peaks ---
    if neg_mask.any():
        neg_peak_rel = np.argmax(test_hist[neg_mask])  # index within masked subset
        neg_peak_test_idx = np.where(neg_mask)[0][neg_peak_rel]  # convert to full index
        neg_peak_test_count = test_hist[neg_peak_test_idx]
    else:
        neg_peak_test_count = None

    if pos_mask.any():
        pos_peak_rel = np.argmax(test_hist[pos_mask])
        pos_peak_test_idx = np.where(pos_mask)[0][pos_peak_rel]
        pos_peak_test_count = test_hist[pos_peak_test_idx]
    else:
        pos_peak_test_count = None

    # --- Null dataset peaks ---
    if neg_mask.any():
        neg_peak_rel = np.argmax(null_hist[neg_mask])
        neg_peak_null_idx = np.where(neg_mask)[0][neg_peak_rel]
        neg_peak_null_count = null_hist[neg_peak_null_idx]
    else:
        neg_peak_null_count = None

    if pos_mask.any():
        pos_peak_rel = np.argmax(null_hist[pos_mask])
        pos_peak_null_idx = np.where(pos_mask)[0][pos_peak_rel]
        pos_peak_null_count = null_hist[pos_peak_null_idx]
    else:
        pos_peak_null_count = None

    return (
        neg_peak_test_count,
        pos_peak_test_count,
        neg_peak_null_count,
        pos_peak_null_count,
    )


def prediction_1_average_test(test_vals, null_vals):
    test_mean = np.mean(test_vals)
    test_median = np.median(test_vals)

    null_mean = np.mean(null_vals)
    null_median = np.median(null_vals)

    if test_mean > null_mean:
        verdict_mean = "CONFIRMED"
    else:
        verdict_mean = "FAILED"

    if test_median > null_median:
        verdict_median = "CONFIRMED"
    else:
        verdict_median = "FAILED"

    return {
        "p1b_test_mean_var": test_mean,
        "p1b_null_mean_var": null_mean,
        "p1c_test_median_var": test_median,
        "p1c_null_median_var": null_median,
        "p1b_verdict_mean": verdict_mean,
        "p1c_verdict_median": verdict_median,
    }


def prediction_1_skewed_mass(test_vals, null_vals):
    try:
        print(
            "Prediction_1 : Evaluating if the probabilty density mass is skewed more than 20%"
        )
        neg_mask_test = test_vals < 0
        pos_mask_test = test_vals > 0

        neg_mask_null = null_vals < 0
        pos_mask_null = null_vals > 0

        neg_mass_test = np.sum(neg_mask_test)
        pos_mass_test = np.sum(pos_mask_test)
        assert (
            neg_mass_test > 0
        ), "Test set : The negative probability mass must exceed 0 for this test to make sense"
        assert (
            pos_mass_test > 0
        ), "Test set : The positive probability mass must exceed 0 for this test to make sense"
        neg_mass_null = np.sum(neg_mask_null)
        pos_mass_null = np.sum(pos_mask_null)
        assert (
            neg_mass_null > 0
        ), "Null set : The negative probability mass must exceed 0 for this test to make sense"
        assert (
            pos_mass_null > 0
        ), "Null set : The positive probability mass must exceed 0 for this test to make sense"
        null_ratio = neg_mass_null / pos_mass_null
        test_ratio = neg_mass_test / pos_mass_test
        if test_ratio is None:
            print(f"   ERROR : Could not compare probability mass%% ")
        elif test_ratio < null_ratio * 0.5:
            print(
                f"   STRONGLY CONFIRMED : The probability mass ratio is skewed positive by more than 50%% "
            )
            verdict = "STRONGLY CONFIRMED"
        elif test_ratio < null_ratio * 0.8:
            print(
                "   CONFIRMED : The probability mass ratio is skewed positive by more than 20%"
            )
            verdict = "CONFIRMED"
        elif test_ratio < null_ratio:
            print(f"  TREND : Skew present but weak")
            verdict = "TREND"
        elif test_ratio > null_ratio:
            print(
                f"   CONTRAINDICATED : Negative mass neg_mass_test/neg_mass_null : {neg_mass_test}/{neg_mask_null}, positive mass  {pos_mass_test}/{pos_mask_null}"
            )
            verdict = "CONTRAINDICATED"
        return {
            "p1e_mass_ratio_test": test_ratio,
            "p1e_mass_ratio_null": null_ratio,
            "p1e_verdict": verdict,
        }
    except Exception:
        return {
            "p1e_mass_ratio_test": None,
            "p1e_mass_ratio_null": None,
            "p1e_verdict": "ERROR",
        }


def prediction_1_shifted_peak_test(test_vals, null_vals):
    """
    The function isolates the four peaks, the negative and the positive for both the test and the null dataset.
    If the negative test peak is lower and the positive test peak is higher than before (and this is not replicated in the control), then
    """
    try:
        print(
            "Prediction_1 : Evaluating if the negative peak is lower than the null and the positive peak is higher than the null"
        )
        (
            neg_test_peak,
            pos_test_peak,
            neg_null_peak,
            pos_null_peak,
        ) = get_variability_peaks(test_vals, null_vals)
        if neg_null_peak is None:
            neg_peak_deviation = 1.0
        else:
            neg_peak_deviation = neg_test_peak / neg_null_peak
        pos_peak_deviation = pos_test_peak / pos_null_peak

        sensitivity = 0.05

        print(f"Neg peak  ({neg_peak_deviation}, pos peak {pos_peak_deviation})")
        if neg_null_peak is None:
            print(
                f"   ANOMALOUS : THere was no negative peak,  {sensitivity*100:.2f}%. Correlate with the variability type and distribution shape"
            )
            verdict = "ANOMALOUS"

        elif neg_peak_deviation < (1 - sensitivity * 4) and pos_peak_deviation > (
            1 + sensitivity * 4
        ):
            print(
                f"   STRONGLY CONFIRMED : The negative peak was depressed more than {(sensitivity*4)*100:.2f}% and the positive peak enhanced more than {(sensitivity*4)*100:.2f}%"
            )
            verdict = "STRONGLY CONFIRMED"
        elif neg_peak_deviation < (1 - sensitivity) and pos_peak_deviation > (
            1 + sensitivity
        ):
            print(
                f"   CONFIRMED : The negative peak was depressed more than {(sensitivity*4)*100:.2f}% and the positive peak enhanced more than {sensitivity*100:.2f}% "
            )
            verdict = "STRONGLY CONFIRMED"
        elif (
            neg_peak_deviation < (1 - sensitivity)
            and np.abs(1 - pos_peak_deviation) < sensitivity
        ):
            print(
                f"   PARTIAL : The negative peak was depressed more than {(sensitivity*4)*100:.2f}%, but the positive peak within sensitivity of p/m {sensitivity*100:.2f}%  "
            )
            verdict = "PARTIAL"
        elif (
            np.abs(1 - neg_peak_deviation) < sensitivity
            and 1 - pos_peak_deviation > sensitivity
        ):
            print(
                f"   PARTIAL : The negative peak was silent, but the positive peak was increased more than {sensitivity*100:.2f} "
            )
            verdict = "PARTIAL"
        elif neg_peak_deviation < (1 - sensitivity) and pos_peak_deviation < (
            1 - sensitivity
        ):
            print(
                f"   ANOMALOUS : Both negative and positive peaks was depressed more than {sensitivity*100:.2f}%. Correlate with the distribution shape"
            )
            verdict = "ANOMALOUS"
        elif pos_peak_deviation > (1 + sensitivity * 4) and neg_peak_deviation > (
            1 + sensitivity * 4
        ):
            print(
                f"   ANOMALOUS : Both negative and positive peaks was increased more than {(sensitivity*4)*100:.2f}%. Correlate with the distribution shape"
            )
            verdict = "ANOMALOUS"
        elif (
            np.abs(1 - neg_peak_deviation) < sensitivity
            and (1 - np.abs(pos_peak_deviation)) < sensitivity
        ):
            print(
                f"   SILENT : Both negative and positive peaks were so close (< {sensitivity*100:.2f}%) that no determination could be made"
            )
            verdict = "SILENT"
        else:
            print(
                f"   CONTRAINDICATED : (or anomalous) Shifted peak not present. Neg peak test/null {neg_test_peak}/{neg_null_peak}, Pos peak test/null : {pos_test_peak}/{pos_null_peak}"
            )
            verdict = "CONTRAINDICATED"
        return {
            "p1d_neg_test_peak": neg_test_peak,
            "p1d_pos_test_peak": pos_test_peak,
            "p1d_neg_null_peak": neg_null_peak,
            "p1d_pos_null_peak": pos_null_peak,
            "p1d_verdict": verdict,
        }
    except Exception as e:
        print("Expceiton ", e)
        return {
            "p1d_neg_test_peak": None,
            "p1d_pos_test_peak": None,
            "p1d_neg_null_peak": None,
            "p1d_pos_null_peak": None,
            "p1d_verdict": "ERROR",
        }


def check_prediction_1(
    pp: ProcessingParameters,
    test_vals,
    null_vals,
    empirical_q=None,
):
    try:
        null_vals = np.asarray(null_vals, dtype=float)
        test_vals = np.asarray(test_vals, dtype=float)
        n = null_vals.size
        m = test_vals.size
        # Merge in the columns from the tail exceedance test
        p1a_tail = prediction_1_tail_exceedance_test(
            test_vals, null_vals, pp.percentile, empirical_q=empirical_q
        )
        p1bc_average = prediction_1_average_test(test_vals, null_vals)
        p1d_peak = prediction_1_shifted_peak_test(test_vals, null_vals)
        p1e_skew = prediction_1_skewed_mass(test_vals, null_vals)
        return p1a_tail | p1bc_average | p1d_peak | p1e_skew
    except Exception as e:
        print(e)
        raise e


def compute_q(
    variability_obs: DataFrame, variability_type: str, percentile: float = 0.001, N=100
):
    q = np.quantile(
        variability_obs[variability_type], 1.0 - percentile, method="higher"
    )  # or 'nearest'/'linear'

    return q


def prediction_1_tail_exceedance_test(
    test_vals, null_vals, percentile: float = 0.001, empirical_q: float | None = None
):
    null_vals = np.asarray(null_vals, dtype=float)
    test_vals = np.asarray(test_vals, dtype=float)
    n = null_vals.size
    m = test_vals.size
    if empirical_q is None:
        raise ValueError("q must be provided to prediction_1_tail_exceedance_test")

    # Exceedances in the A dataset compared to the specified empirical Q
    x = int(np.sum(test_vals >= empirical_q))
    p_hat = x / m

    # Exact binomial one-sided test (H1: p > p0)
    test = binomtest(x, m, p=percentile, alternative="greater")

    # Clopper–Pearson 95% CI for p_hat
    ci_low, ci_high = test.proportion_ci(confidence_level=0.95, method="exact")

    return {
        "p1a_threshold_q": float(empirical_q),
        "p1a_null_n": int(n),
        "p1a_test_m": int(m),
        "p1a_exceedances_x": x,
        "p1a_exceed_rate_p_hat": p_hat,
        "p1a_null_rate_p0": percentile,
        "p1a_enrichment_p_hat_over_p0": (p_hat / percentile)
        if percentile > 0
        else np.inf,
        "p1a_p_value_binomial": test.pvalue,
        "p1a_ci95_p_hat": (float(ci_low), float(ci_high)),
        "p1a_verdict": "SIGNIFICANT" if test.pvalue < percentile else "NOT SIGNIFICANT",
    }


def check_prediction_2(
    chunked_variability_A_top,
    chunked_variability_B_top,
    pp: ProcessingParameters,
    variability_type: str,
    pipe: ComparePipeline,
    perc: float = None,
    saveplot: bool = False,
):
    results = {}
    (
        a_A,
        b_A,
        a_B,
        b_B,
        se_a_A,
        se_b_A,
        se_a_B,
        se_b_B,
        p2_p,
    ) = plot_wavelength_counts_histogram(
        event_data_A=chunked_variability_A_top,
        event_data_B=chunked_variability_B_top,
        column="Wavelength Center",
        pp=pp,
        filename=f"{pipe.id}/var_concentration_{variability_type}.png",
        A_handle=pipe.A_fits_id,
        B_handle=pipe.B_fits_id,
        title="High variability concentration",
        perc=perc,
        saveplot=saveplot,
    )
    results["p2_slope_A"] = b_A
    results["p2_slope_B"] = b_B
    if p2_p is None:
        print("  ERROR : The slope difference were not computed ")
        p2a_significance_verdict = "ERROR"
    elif p2_p < 0.05:
        print(
            "  CONFIRMED : The slope difference in the curves is statistically significant "
        )
        p2a_significance_verdict = "SIGNIFICANT"
    else:
        p2a_significance_verdict = "NOT SIGNIFICANT"

    results["p2a_slope_p"] = p2_p
    results["p2a_verdict"] = p2a_significance_verdict
    if b_A is None:
        print("  ERROR : The slope of the curve had an ERROR ")
        p2b_verdict = "ERROR"
    elif b_A < 0:
        print(f" CONFIRMED : The slope of the curve is negative {b_A}")
        p2b_verdict = "CONFIRMED"
    else:
        print(f" CONTRAINDICATED: The slope of the curve was positive {b_A}")
        p2b_verdict = "CONTRAINDICATED"

    # If the exponential slope of the non-poissonian is higher than
    results["p2b_verdict"] = p2b_verdict

    return results


def compute_chunk_var(
    meta: FitsMetadata, pp: ProcessingParameters, handle: str, force: bool = False
):
    chunkvar_id = f"{meta.id}_{pp.id}"
    chunk_meta = load_chunk_metadata(chunkvar_id)

    chunkvar_is_cached, cached_filename = get_cached_filename("chunkvar", meta, pp)

    if chunkvar_is_cached and not force:
        return [f"Already cached {cached_filename}"]
    else:
        log, meta, source = load_source_data(meta, pp)
        log, meta, binned_datasets, time_bin_widths = get_binned_datasets(
            source_data=source, meta=meta, pp=pp, handle=handle
        )
        # The name of the chunk dataset for this observation
        variability_observations, chunk_meta = compute_time_variability_async(
            binned_datasets=binned_datasets,
            meta=meta,
            pp=pp,
            time_bin_widths=time_bin_widths,
        )

        # print(variability_observations.head(200))
        fits_save_cache(cached_filename, variability_observations)
        save_chunk_metadata(chunk_meta)
        return [f"Saved chunk file {cached_filename}"]


def get_empirical_q_from_results_file(pipeline_id: str, variability_type: str):
    """
    Docstring for get_empirical_q_from_results_file
    Read CSV file and determine the average value in the q_threshold_this_run columns that satisfies filtering criteria
    :param pipeline_id: Description
    :type pipeline_id: str
    :param variability_type: Description
    :type variability_type: str
    """
    filename = f"files/baseliner.csv"
    try:
        # Read csv file
        df = pd.read_csv(filename)

        idxs = (df["variability_type"] == variability_type) & (
            df["pipeline_id"] == pipeline_id
        )

        q_observations = pd.to_numeric(
            df.loc[idxs, "q_threshold_this_run"], errors="coerce"
        ).mean()

        print(
            f"Read empirical q {q_observations.mean()} from baseline file {filename} for variability type {variability_type}"
        )
        return q_observations.mean()
    except Exception as e:
        print(f"Error reading baseline file {filename}: {e}")
        return 0.0


def prediction_1(
    pipe: ComparePipeline,
    metaA: FitsMetadata,
    metaB: FitsMetadata,
    pp: ProcessingParameters,
    variability_type: str,
    chunked_variability_A: DataFrame,
    chunked_variability_B: DataFrame,
    plot=False,
):
    # Prepare data for PREDICITON 1

    # PREDICTION 1 tests if the number of tail exceedances in the current A dataset, exceeds a specified q-thresholds
    # This q-threshold can vary a great deal between runs, because of the the randomized bin width, and other random effects.
    # So each trial needs to be checked against
    # the average q value, orhterwise the confidence will fluctutate a lot (at least 1000x) per run, making the
    # confidence difficult to pin down.

    # First compute the q-threshold for the knockout dataset for this run
    #  This marks the 0.999 percentile cutoff on the knockout dataset for this particular randomization.

    q_threshold_this_run = compute_q(
        chunked_variability_B,
        variability_type=variability_type,
        percentile=pp.percentile,
    )

    # Then load all the computed q threshold from a reference set called baseliner.csv,
    # and use that empirical q to compute the confidence of this particular trial.

    empirical_q = get_empirical_q_from_results_file(pipe.id, variability_type)

    # Extract the variability metrics of the current type
    test_vals = chunked_variability_A[variability_type]

    # Extract the variability metrics for this variability type
    null_vals = chunked_variability_B[variability_type]

    # If both A and B are poissonian, we expect a null results
    # Special case if the pipe.id is named isogen_vs_rnd_isogen, this is a generated dataset with enough Lane density to assume that it is isotropic
    if "poissonize" in pipe.A_tasks or pipe.id == "isogen_vs_rnd_isogen":
        A_expected_null = True
    else:
        A_expected_null = False

    if "poissonize" in pipe.B_tasks:
        B_expected_null = True
    else:
        B_expected_null = False

    # Prepare the output dictionary
    result = {
        "A_expected_null": A_expected_null,
        "B_expected_null": B_expected_null,
        "q_threshold_this_run": q_threshold_this_run,
        "empirical_q": empirical_q,
    }

    print(f"{pipe.id} Testing prediction 1 ")
    # Merge in the columns returned from the actual prediction
    result = result | check_prediction_1(
        pp=pp, test_vals=test_vals, null_vals=null_vals, empirical_q=empirical_q
    )

    if plot:
        print(
            f"Generating PREDICTION 1 plots for {variability_type} in pipeline {pipe.id} "
        )
        # Plot the histogram shape of the variability of these two datasets
        # This is used to separate isotropic, temporal variability from

        min_y_A = np.min(test_vals)
        max_y_A = np.max(test_vals)
        min_y_B = np.min(null_vals)
        max_y_B = np.max(null_vals)

        plot_chunk_variability_shape(
            variabilityA=chunked_variability_A,
            variabilityB=chunked_variability_B,
            pp=pp,
            variability_type=variability_type,
            filename=f"{pipe.id}/hist_{variability_type}.png",
            show=False,
            handleA=metaA.star,
            handleB=metaB.star,
        )

        plot_chunk_variability_excess(
            variability=chunked_variability_A,
            pp=pp,
            variability_type=variability_type,
            filename=f"{pipe.id}/A_{variability_type}.png",
            show=False,
            handle="A",
            ylim=(np.min([min_y_A, min_y_B]), np.max([max_y_A, max_y_B])),
        )

        plot_chunk_variability_excess(
            variability=chunked_variability_B,
            pp=pp,
            filename=f"{pipe.id}/B_{variability_type}.png",
            variability_type=variability_type,
            show=False,
            handle="B",
            ylim=(np.min([min_y_A, min_y_B]), np.max([max_y_A, max_y_B])),
        )

    return result


def perc_diff(a: DataFrame, b: DataFrame, not_more_than: float):
    perc_diff = abs(1 - (a.shape[0] / b.shape[0]))
    return perc_diff > not_more_than, perc_diff


def prediction_2(
    pipe: ComparePipeline,
    metaA: FitsMetadata,
    metaB: FitsMetadata,
    pp: ProcessingParameters,
    variability_type: str,
    chunked_variability_A: DataFrame,
    chunked_variability_B: DataFrame,
    plot: bool = False,
):
    # PREDICITON 2
    # Prepare data for prediction 2, varibility / hardening
    # Prediction to is easier to see when focusing on the shape of high-variability tail
    # Take the top highest variability metrics, and compare the shape.
    max_length_diff = 0.01

    halt, diff = perc_diff(
        chunked_variability_A, chunked_variability_B, max_length_diff
    )
    assert (
        not halt
    ), f"The shape of dataset A {chunked_variability_A.shape[0]} and B {chunked_variability_A.shape[0]} was {diff:2f}, exceeding safeguard of {max_length_diff:2f}"

    # Ensure that we are using a value
    if pp.variability_percentile is None:
        print("WARNING : Variability precentile not specified, default to top 10%")
        pp.variability_percentile = 0.01

    # Check if the size of the chunked variability datasets are the same. They should be!
    percdiff = abs(1 - chunked_variability_A.shape[0] / chunked_variability_B.shape[0])
    assert (
        percdiff < 0.001
    ), f"The size of chunked variability datasets differs by {percdiff}%. A{chunked_variability_A.shape[0]} vs B: {chunked_variability_B.shape[0]}. This has downstream effects, and should be exactly the same!"

    # Take the top variability from A dataset
    take_obs_A = int(pp.variability_percentile * chunked_variability_A.shape[0])
    take_obs_B = int(pp.variability_percentile * chunked_variability_B.shape[0])
    percdiff = abs(1 - take_obs_A / take_obs_B)
    assert (
        percdiff < 0.01
    ), f"The top observation counts in datasets vary by more than 1 percent! Diff {percdiff} %"

    chunked_variability_A_top = take_max_variability(
        take_obs_A,
        chunked_variability_A,
        variability_type,
    )

    chunked_variability_B_top = take_max_variability(
        take_obs_B,
        chunked_variability_B,
        variability_type,
    )

    # Find the clamp

    print(f"{pipe.id} Testing prediction 12")
    result = check_prediction_2(
        pp=pp,
        chunked_variability_A_top=chunked_variability_A_top,
        chunked_variability_B_top=chunked_variability_B_top,
        variability_type=variability_type,
        pipe=pipe,
        perc=pp.variability_percentile,
        saveplot=plot,
    )

    if plot:
        min_y_A = np.min(chunked_variability_A_top)
        max_y_A = np.max(chunked_variability_A_top)
        min_y_B = np.min(chunked_variability_B_top)
        max_y_B = np.max(chunked_variability_B_top)

        plot_top_variability_timescales(
            variability=chunked_variability_A_top,
            pp=pp,
            variability_type=variability_type,
            filename=f"{pipe.id}/time_scales_{variability_type}_A.png",
            show=False,
            fit_curve=False,
            handle="A",
        )

        plot_top_variability_timescales(
            variability=chunked_variability_B_top,
            pp=pp,
            variability_type=variability_type,
            filename=f"{pipe.id}/time_scales_{variability_type}_B.png",
            show=False,
            fit_curve=False,
            handle="B",
        )

    return result


def compare_two_sets(
    pipe: ComparePipeline,
    metaA: FitsMetadata,
    metaB: FitsMetadata,
    pp: ProcessingParameters,
    plot=False,
):
    ensure_pipeline_folders_exists(pipe)
    print(
        f"Comparing two datasets : {metaA.id}: {metaA.star} vs {metaB.id} {metaB.star}"
    )

    log = []

    (
        A_log,
        chunked_variability_A,
        cached_filename_A,
    ) = load_or_compute_chunk_variability_observations(
        pipe,
        meta=metaA,
        pp=pp,
        handle="A",
    )

    ensure_path_exists(f"./plots/{pipe.id}")
    log.extend(A_log)

    # If B is set in the pipeline, we use this as a comparion

    (
        B_log,
        chunked_variability_B,
        cached_filename_B,
    ) = load_or_compute_chunk_variability_observations(
        pipe,
        meta=metaB,
        pp=pp,
        handle="B",
    )
    log.extend(B_log)
    if cached_filename_A == cached_filename_B:
        print()
        log.append(
            f"WARNING : A {metaA.id} = B {metaB.id} has the same cache file. Comparing the same cached dataset is going to be a problem"
        )

    logA, metaA, source_A = load_source_data(meta=metaA, pp=pp)
    logB, metaB, source_B = load_source_data(meta=metaB, pp=pp)

    print(f"{metaA.id} contains {source_A.shape[0]} rows")
    print(f"{metaB.id} contains {source_B.shape[0]} rows")
    assert (
        pp.variability_type in chunked_variability_A.columns
    ), f"Expected column {pp.variability_type} in columns for chunked_variability_A"
    assert (
        pp.variability_type in chunked_variability_B.columns
    ), f"Expected column {pp.variability_type} in columns for chunked_variability_B"
    results = []
    for variability_type in variability_types:
        try:
            assert (
                variability_type in chunked_variability_B.columns
            ), f"Expected column {variability_type} in columns for chunked_variability_B"

            assert (
                variability_type in chunked_variability_A.columns
            ), f"Expected column {variability_type} in columns for chunked_variability_A"

            # Extract the variability metrics
            test_vals = chunked_variability_A[variability_type]
            null_vals = chunked_variability_B[variability_type]

            result = {
                "date": datetime.datetime.now(),
                "pp_hash": pp.get_hash(),
                "pipeline_id": pipe.id,
                "variability_type": variability_type,
                "A": metaA.id,
                "B": metaB.id,
                "pp": pp.id,
                "date_run": datetime.datetime.now(),
            }

            # Merge the data from prediction_1 into the dataset
            result = result | prediction_1(
                pipe,
                metaA,
                metaB,
                pp,
                variability_type,
                chunked_variability_A,
                chunked_variability_B,
                plot=False,
            )
            result = result | prediction_2(
                pipe,
                metaA,
                metaB,
                pp,
                variability_type,
                chunked_variability_A,
                chunked_variability_B,
                plot=True,
            )

            # Write result of these tests to statistical tests
            ensure_pipeline_folders_exists(pipe)
            csv_file = "files/statistical_tests.csv"
            write_result_to_csv(result, csv_file)
            results.append(result)

        except Exception as e:
            print(" Exception caused the image generation to stop ", e)
            print(e)
            raise e
    return log, results


def run_all_pipelines(search: str = None, plot: bool = True, force_chunk: bool = False):
    """
    Discover all JSON pipeline metadata files and run each pipeline.
    Returns a dict with 'success' and 'failed' lists.
    """
    pipelines_dir = os.path.join(os.getcwd(), "meta_files", "pipeline")
    if not os.path.isdir(pipelines_dir):
        raise FileNotFoundError(f"Pipeline directory not found: {pipelines_dir}")

    json_files = sorted(
        f
        for f in os.listdir(pipelines_dir)
        if f.lower().endswith(".json")
        and os.path.isfile(os.path.join(pipelines_dir, f))
    )

    summary = {"success": [], "failed": []}

    for fname in json_files:
        pipeline_id = os.path.splitext(fname)[0]
        if search is not None:
            if search not in pipeline_id:
                print(f"Skipping {pipeline_id}, did not match search string {search}")
                continue
        try:
            print(f"Loading pipeline pipeline_id: {pipeline_id}")
            pipeline_meta = load_pipeline(pipeline_id)
        except Exception as e:
            summary["failed"].append(
                {"pipeline_id": pipeline_id, "stage": "load_pipeline", "error": str(e)}
            )
            continue

        try:
            result = run_pipeline(pipeline_meta, plot, force_chunk)
            summary["success"].append(
                {"pipeline_id": pipeline_meta.id, "result": result}
            )
        except Exception as e:
            summary["failed"].append(
                {
                    "pipeline_id": getattr(pipeline_meta, "id", pipeline_id),
                    "stage": "run_pipeline",
                    "error": str(e),
                }
            )

    return summary


def print_source_visualizations(
    event_data: DataFrame,
    pipeline: ComparePipeline,
    meta: FitsMetadata,
    pp: ProcessingParameters,
    handle: str,
):
    ensure_pipeline_folders_exists(meta=pipeline)

    wbins, wave_centers, wave_widths = get_wavelength_bins(pp)
    plot_wbin_time_heatmap(
        data=event_data,
        filename=f"{pipeline.id}/timeheatmap_{handle}.png",
        wbins=wbins,
        wbin_col="Wavelength (nm)",
        dt=100,
        show=False,
        tcol="time",
        log_scale=False,
        cmap="viridis",
        handle=handle,
    )
    """
    plot_ccd_bin(
        source_data,
        filename=f"{pipeline.id}/plot_{meta.id}_count_{pp.id}.png",
        show=False,
        handle=handle,
    )
    """


def run_pipeline(
    pipeline: ComparePipeline, plot: bool = True, force_chunk: bool = False
):
    print(f"Running pipeline {pipeline.id}")
    ensure_path_exists(f"./plots/{pipeline.id}")
    logs = []
    try:
        if pipeline.A_fits_id is None:
            raise Exception("Cannot run var_analysis with A slot empty in pipeline")
        if pipeline.B_fits_id is None:
            raise Exception("Cannot run var_analysis with B slot empty in pipeline")
        # if pipeline.source is not None:
        # source_meta = load_fits_metadata(pipeline.source)
        # source_data = fits_read(source_meta.raw_event_file)
        metaA = load_fits_metadata(pipeline.A_fits_id)
        metaB = load_fits_metadata(pipeline.B_fits_id)

        if metaA is None or metaB is None:
            return
        pp = load_processing_param(pipeline.pp_id)
        genmeta = load_gen_param(pipeline.gen_id)
        if force_chunk:
            logs = compute_chunk_var(metaA, pp, "A", force_chunk)
            logs.extend(compute_chunk_var(metaB, pp, "B", force_chunk))

        if plot:
            event_data_A = fits_read(metaA.raw_event_file)
            event_data_B = fits_read(metaB.raw_event_file)
            print(
                f"Loaded dataset {metaA.raw_event_file} with {event_data_A.shape[0]} observations"
            )
            if metaA.synthetic and event_data_A is None:
                print(
                    f"Synhetic data for {metaB.id} data (path {metaB.raw_event_file}) was not found. Must regenerate data"
                )
            elif event_data_A is None:
                raise Exception(
                    "Error : The meta file contained an invalid path to a fits file"
                )

            plot_wavelength_counts_histogram(
                event_data_A=event_data_A,
                event_data_B=event_data_B,
                column="pi",
                pp=pp,
                filename=f"{pipeline.id}/counts_histogram_pi.png",
                title=f"Spectral shape (pi column)",
                A_handle=f"A : {metaA.star}",
                B_handle=f"B : {metaB.star}",
                saveplot=plot,
            )
            plot_wavelength_counts_histogram(
                event_data_A=event_data_A,
                event_data_B=event_data_B,
                column="Wavelength (nm)",
                pp=pp,
                filename=f"{pipeline.id}/counts_histogram_w.png",
                title="Spectral shape (Wavelength nm)",
                A_handle=f"A : {metaA.star}",
                B_handle=f"B : {metaB.star}",
                saveplot=plot,
            )

            print_source_visualizations(
                event_data=event_data_A,
                pipeline=pipeline,
                meta=metaA,
                pp=pp,
                handle="A",
            )
            source_data_B = fits_read(metaB.raw_event_file)
            print_source_visualizations(
                event_data=event_data_B,
                pipeline=pipeline,
                meta=metaB,
                pp=pp,
                handle="B",
            )

            print_source_visualizations(
                event_data=source_data_B,
                pipeline=pipeline,
                meta=metaB,
                pp=pp,
                handle="B",
            )

        log, result = compare_two_sets(
            pipe=pipeline, metaA=metaA, metaB=metaB, pp=pp, plot=plot
        )
        logs.extend(log)
        print(f"Finished pipe {pipeline.id}")
    except Exception as e:
        print(f"ERROR in pipeline {pipeline.id}, aborting")
        print(e)
        raise e
    return logs


def main():
    log = []
    print("Variability analysis : Chunk Phased Sampling and plot generation ")
    args = parse_arguments()
    if args.chunk == "force":
        force_chunk = True
    else:
        force_chunk = False

    if args.plot:
        plot = True
    else:
        plot = False

    if args.pipe is not None:
        if args.pipe == "all":
            print("Will runn all pipelines found...")
            summary = run_all_pipelines(search=None, plot=plot, force_chunk=force_chunk)
            for result in summary:
                print(f"Summary {result}")
        elif args.pp is not None:
            raise Exception(
                f"You cannot use a different ProcessingParameter id {args.pp} than what is specified in the pipeline file. Try modifying /meta_files/pipelines/{args.pipe}.json instead"
            )
        else:
            summary = run_all_pipelines(args.pipe, plot, force_chunk=force_chunk)
            assert "success" in summary, "Expected key 'success' in dictionary"
            assert "failed" in summary, "Expected key 'failed' in dictionary"
            print("SUCCESS PIPELINES")
            for logline in summary["success"]:
                print(f"{logline['pipeline_id']}: {logline['result']}")
            print("FAILED PIPELINES")
            for logline in summary["failed"]:
                print(f"{logline['pipeline_id']}: {logline['result']}")
            successes = len(summary["success"])
            failures = len(summary["failed"])
            print(f"Successful pipes : {successes}")
            if failures == 0:
                print("Great news, no failed pipes!")
            else:
                print(f"WARNING : {failures} failed pipes")


if __name__ == "__main__":
    main()
