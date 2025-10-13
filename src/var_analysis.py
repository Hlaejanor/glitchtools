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
defaultpipe = "antares_vs_rnd_antares"
defaultpipe = "all"
defaultpipe = "anisogen2_vs_rnd_anisogen2"
defaultpipe = "anisogen_vs_rnd_anisogen"
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
        "-p",
        "--pipe",
        type=str,
        default=defaultpipe,
        help="Select the experiment pipe name ",
    )
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


def tail_exceedance_test(
    pipeline: ComparePipeline,
    metaA: FitsMetadata,
    metaB: FitsMetadata,
    pp: ProcessingParameters,
    comparing_to_self: bool,
    variability_type: str,
    test_vals,
    null_vals,
    significance_cutoff: float = 0.001,
):
    null_vals = np.asarray(null_vals, dtype=float)
    test_vals = np.asarray(test_vals, dtype=float)
    n = null_vals.size
    m = test_vals.size

    # 99.9% empirical threshold
    q = np.quantile(
        null_vals, 1.0 - pp.percentile, method="higher"
    )  # or 'nearest'/'linear'

    # Exceedances in Antares
    x = int(np.sum(test_vals >= q))
    p_hat = x / m

    # Exact binomial one-sided test (H1: p > p0)
    test = binomtest(x, m, p=pp.percentile, alternative="greater")

    # Clopper–Pearson 95% CI for p_hat
    ci_low, ci_high = test.proportion_ci(confidence_level=0.95, method="exact")

    result = {
        "date": datetime.datetime.now(),
        "pp_hash": pp.get_hash(),
        "pipeline_id": pipeline.id,
        "variability_type": variability_type,
        "A": metaA.id,
        "B": metaB.id,
        "threshold_q": float(q),
        "null_n": int(n),
        "antares_m": int(m),
        "exceedances_x": x,
        "exceed_rate_p_hat": p_hat,
        "null_rate_p0": pp.percentile,
        "enrichment_p_hat_over_p0": (p_hat / pp.percentile)
        if pp.percentile > 0
        else np.inf,
        "p_value_binomial": test.pvalue,
        "ci95_p_hat": (float(ci_low), float(ci_high)),
        "comparing_to_self": comparing_to_self,
        "significant": test.pvalue < significance_cutoff,
    }
    return result


def compare_two_sets(
    pipe: ComparePipeline,
    metaA: FitsMetadata,
    metaB: FitsMetadata,
    pp: ProcessingParameters,
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

    for variability_type in variability_types:
        try:
            assert (
                variability_type in chunked_variability_B.columns
            ), f"Expected column {variability_type} in columns for chunked_variability_B"

            assert (
                variability_type in chunked_variability_A.columns
            ), f"Expected column {variability_type} in columns for chunked_variability_A"

            if pp.take_top_variability_count is not None:
                assert (
                    pp.take_top_variability_count > 0
                ), "take_top_variability_count must be > 0"
                assert (
                    pp.take_top_variability_count < pp.time_bin_widths_count
                ), "take_top_variability_count must be < time_bin_widths_count, otherwise you will take all observations, and there will be no selection of the higest variability is "
                test_vals = take_top_wavelengths_per_timescale(
                    pp.take_top_variability_count,
                    chunked_variability_A,
                    variability_type,
                )
                null_vals = take_top_wavelengths_per_timescale(
                    pp.take_top_variability_count,
                    chunked_variability_B,
                    variability_type,
                )
            else:
                test_vals = chunked_variability_A[variability_type]
                null_vals = chunked_variability_B[variability_type]

            min_y_A = np.min(test_vals)
            max_y_A = np.max(test_vals)
            min_y_B = np.min(null_vals)
            max_y_B = np.max(null_vals)

            print(f"Generating plots for {variability_type} in pipeline {pipe.id} ")
            results = tail_exceedance_test(
                pipeline=pipe,
                variability_type=variability_type,
                metaA=metaA,
                metaB=metaB,
                pp=pp,
                test_vals=test_vals,
                null_vals=null_vals,
                comparing_to_self=cached_filename_A == cached_filename_B,
            )
            # Example usage
            ensure_pipeline_folders_exists(pipe)
            csv_file = f"files/tail_exceedance_test_{pp.get_hash()}.csv"
            write_result_to_csv(results, csv_file)

            print(results)
            plot_chunk_variability_excess(
                variability=chunked_variability_A,
                pp=pp,
                variability_type=variability_type,
                filename=f"{pipe.id}/{variability_type}_A_{metaA.id}.png",
                show=False,
                use_old=False,
                handle="A",
                ylim=(np.min([min_y_A, min_y_B]), np.max([max_y_A, max_y_B])),
            )

            plot_chunk_variability_excess(
                variability=chunked_variability_B,
                pp=pp,
                filename=f"{pipe.id}/{variability_type}_B_{metaB.id}.png",
                variability_type=variability_type,
                show=False,
                use_old=False,
                handle="B",
                ylim=(np.min([min_y_A, min_y_B]), np.max([max_y_A, max_y_B])),
            )

            test_vals_A = take_top_wavelengths_per_timescale(
                take_top_variability_count=100,
                variability_observations=chunked_variability_A,
                variability_type=variability_type,
            )

            test_vals_B = take_top_wavelengths_per_timescale(
                take_top_variability_count=100,
                variability_observations=chunked_variability_B,
                variability_type=variability_type,
            )

            plot_top_variability_timescales(
                variability=test_vals_A,
                pp=pp,
                variability_type=variability_type,
                filename=f"{pipe.id}/time_scales_{variability_type}_A_{metaA.id}.png",
                show=False,
                fit_curve=False,
                handle="A",
            )

            plot_top_variability_timescales(
                variability=test_vals_B,
                pp=pp,
                variability_type=variability_type,
                filename=f"{pipe.id}/time_scales_{variability_type}_B_{metaB.id}.png",
                show=False,
                fit_curve=False,
                handle="B",
            )

        except Exception as e:
            print(" Exception caused the image generation to sop")
            print(e)

    if False:
        logA, metaA, time_binned_dfs_A = add_time_binning(
            pipeline=pipe, source_data=source_A, meta=metaA, pp=pp, handle="A"
        )
        for binned_df in time_binned_dfs_A:
            make_standard_plots(
                pp=pp,
                meta=metaA,
                time_binned_source=binned_df,
                pipline_id=pipe.id,
                handle="A",
            )

        logB, metaB, time_binned_dfs_B = add_time_binning(
            pipeline=pipe, source_data=source_B, meta=metaB, pp=pp, handle="B"
        )
        for binned_df in time_binned_dfs_B:
            make_standard_plots(
                pp=pp,
                meta=metaB,
                time_binned_source=binned_df,
                pipline_id=pipe.id,
                handle="B",
            )


def run_all_pipelines():
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
        try:
            pipeline_meta = load_pipeline(pipeline_id)
        except Exception as e:
            summary["failed"].append(
                {"pipeline_id": pipeline_id, "stage": "load_pipeline", "error": str(e)}
            )
            continue

        try:
            result = run_pipeline(pipeline_meta)
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
    source_data: DataFrame,
    pipeline: ComparePipeline,
    meta: FitsMetadata,
    pp: ProcessingParameters,
    handle: str,
):
    ensure_pipeline_folders_exists(meta=pipeline)

    plot_pi_time_heatmap(
        data=source_data,
        filename=f"{pipeline.id}/plot_{meta.id}_energy_{pp.id}.png",
        dt=100,
        show=False,
        tcol="time",
        picol="pi",
        pi_bin_width=1,
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


def run_pipeline(pipeline: ComparePipeline):
    print(f"Running pipeline {pipeline.id}")
    try:
        if pipeline.A_fits_id is None:
            raise Exception("Cannot run var_analysis with A slot empty in pipeline")
        if pipeline.B_fits_id is None:
            raise Exception("Cannot run var_analysis with B slot empty in pipeline")

        metaA = load_fits_metadata(pipeline.A_fits_id)
        metaB = load_fits_metadata(pipeline.B_fits_id)
        pp = load_processing_param(pipeline.pp_id)
        genmeta = load_gen_param(pipeline.gen_id)
        source_data_A = fits_read(metaA.raw_event_file)

        if metaA.synthetic and source_data_A is None:
            print(
                f"Synhetic data for {metaB.id} data (path {metaB.raw_event_file}) was not found. Must regenerate data"
            )
        elif source_data_A is None:
            raise Exception(
                "Error : The meta file contained an invalid path to a fits file"
            )

        print_source_visualizations(
            source_data=source_data_A, pipeline=pipeline, meta=metaA, pp=pp, handle="A"
        )
        source_data_B = fits_read(metaB.raw_event_file)

        if metaB.synthetic and source_data_B is None:
            print(
                f"Synhetic data for {metaB.id} data (path {metaB.raw_event_file}) was not found. Must regenerate data"
            )

            events_B = generate_synthetic_telescope_data(
                genmeta=genmeta,
                wavelength_bins=pp.wavelength_bins,
            )
            # Save the the generated dataset as a fits file
            meta_B = fits_save_events_generated(events_B, genmeta)
            # Update the pipeline with reference to the new fits file
            pipeline.B_fits_id = meta_B.id
            # Save the pipeline
            save_pipeline(pipeline)
            source_data_B = fits_read(metaB.raw_event_file)
        elif source_data_B is None:
            raise Exception(
                "Error : The meta file contained an invalid path to a fits file"
            )

        print_source_visualizations(
            source_data=source_data_B, pipeline=pipeline, meta=metaB, pp=pp, handle="B"
        )
        compare_two_sets(pipe=pipeline, metaA=metaA, metaB=metaB, pp=pp)
        print(f"Finished pipe {pipeline.id}")
    except Exception as e:
        print(f"ERROR in pipeline {pipeline.id}, aborting")
        print(e)
        raise e
    return []


def main():
    log = []
    print("Lightlanefinder")
    args = parse_arguments()
    if args.pipe is not None:
        if args.pipe == "all":
            summary = run_all_pipelines()
            for result in summary:
                print(f"Summary {result}")

        elif args.meta_a is not None:
            raise Exception(
                f"You cannot both provide a pipe and a specific meta id, try modifying the /meta_files/pipeline/{args.pipe}.json"
            )
        elif args.pp is not None:
            raise Exception(
                f"You cannot use a different ProcessingParameter id {args.pp} than what is specified in the pipeline file. Try modifying /meta_files/pipelines/{args.pipe}.json instead"
            )
        else:
            pipeline_meta = load_pipeline(args.pipe)
            assert pipeline_meta is not None, "Pipeline metadata was 0"
            log_pipes = run_pipeline(pipeline_meta)
            log.extend(log_pipes)

        # B_gen = load_gen_param(pipeline_meta.gen_id)
    else:
        raise Exception("Missing pipe informatoin")
    print("LOG : ")
    for logline in log:
        print(logline)

    print("\n".join(log))


if __name__ == "__main__":
    main()
