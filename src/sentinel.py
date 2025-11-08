import sys
import os

from pandas import DataFrame
from common.helper import get_wavelength_bins, get_uneven_time_bin_widths
from event_processing.plotting import (
    compute_time_variability_async,
    take_max_variability_per_wbin,
)
from event_processing.vartemporal_plotting import harmonic_band_plot
from common.fitsread import (
    fits_save_events_with_pi_channel,
    read_event_data_crop_and_project_to_ccd,
    fits_save_chunk_analysis,
    fits_read,
    fits_save_cache,
    get_cached_filename,
    fits_read_cache_if_exists,
)
from event_processing.detectbands import (
    analyze_bands,
    find_band_times,
    lomb_scargle_periodogram,
    tally_votes_find_rige_peaks,
    test_rayleigh,
)
from common.helper import randomly_sample_from, get_duration
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
)
from event_processing.var_analysis_plots import (
    make_standard_plots,
    SDMC_plot,
    get_uneven_time_bin_widths,
)
from event_processing.binning import get_binned_datasets
from common.generate_data import generate_synthetic_telescope_data
import argparse

version = 1.0


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare two metadata files and optionally set downsampling."
    )
    parser.add_argument(
        "-p",
        "--pipe",
        type=str,
        default="antares_vs_isogen",
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
        type=int | int,
        default=None,
        help="Number of samples for downsampling (default: 100000)",
    )

    args = parser.parse_args()
    return args


def main():
    print(f"Chandra Lightlane Variability analysis v{version}")
    args = parse_arguments()
    N = args.n_samples
    log = []
    log.append(f"Both : Using N = {N}")

    if args.pipe is not None:
        pipeline_meta = load_pipeline(args.pipe)
        assert pipeline_meta is not None, "Pipeline metadata was 0"
        if args.meta_a is not None:
            raise Exception(
                f"You cannot both provide a pipe and a specific meta id, try modifying the /meta_files/pipelines/{pipeline_meta.id}.json"
            )
        if args.pp is not None:
            raise Exception(
                f"You cannot use a different ProcessingParameter id {args.pp} than what is specified in the pipeline file. Try modifying /meta_files/pipelines/{pipeline_meta.id}.json instead"
            )

        metaA = load_fits_metadata(pipeline_meta.A_fits_id)
        assert metaA.t_max > 0, "T_max has not been set. "
        pp = load_processing_param(pipeline_meta.pp_id)
        assert pp is not None, "Processing param was none"
        binning_is_cached, cached_filename = get_cached_filename("binning", metaA, pp)
        edges = []
        using_best = False

        if binning_is_cached:
            binned = fits_read_cache_if_exists(cache_filename_path=cached_filename)
            logA = [f"A : Loaded cached binned data from {cached_filename}"]
        else:
            logA, metaA, binned = get_binned_datasets(
                pipeline=pipeline_meta, meta=metaA, pp=pp, handle="A", N=N
            )
            fits_save_cache(cached_filename, binned)

        wave_edges, wave_centers, wave_widths = get_wavelength_bins(pp)
        time_bin_widths = get_uneven_time_bin_widths(pp)
        time_bin_widht_index = 0

        edges_is_cached, cached_filename = get_cached_filename("edges", metaA, pp)

        if edges_is_cached:
            edges = fits_read_cache_if_exists(cache_filename_path=cached_filename)
            logA = [f"A : Loaded cached edges data from {cached_filename}"]
        else:
            while time_bin_widht_index < len(time_bin_widths):
                tbinw_edges = analyze_bands(
                    binned_data=binned,
                    time_bin_widht_index=time_bin_widht_index,
                    time_bin_widths=time_bin_widths,
                )

                edges.extend(tbinw_edges)
                print(
                    f"Width index {time_bin_widht_index} with width {time_bin_widths[time_bin_widht_index]} had {len(tbinw_edges)} edges:"
                )
                print(tbinw_edges)

                harmonic_band_plot(
                    pipeline_meta,
                    metaA,
                    binned_data=binned,
                    wbin_widths=wave_widths,
                    time_bin_widht_index=time_bin_widht_index,
                    time_bin_widhts=time_bin_widths,
                    bands=tbinw_edges,
                )

                time_bin_widht_index += 1

            edges_df = DataFrame(edges, columns=["Time Bin Width Index", "Time"])

            fits_save_cache(cached_filename, edges_df)
            logA.append(f"A : Saved edges data to {cached_filename}")

    peak_times, peak_strengths = tally_votes_find_rige_peaks(edges)
    print("Detected peaks at times:")
    for t, s in zip(peak_times, peak_strengths):
        print(f"  Time {t} with strength {s}")
        log.append(f"Detected peak at time {t} with strength {s}")

    test_rayleigh(peak_times, peak_times[0])
    print("LOG : ")
    for logline in log:
        print(logline)

    print("\n".join(log))


if __name__ == "__main__":
    main()
