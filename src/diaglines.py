import sys
import os
import pandas as pd
import numpy as np
from pandas import DataFrame
from common.helper import get_wavelength_bins, get_uneven_time_bin_widths
from event_processing.plotting import (
    compute_time_variability_async,
    take_max_variability_per_wbin,
    plot_periodicity_winners,
)
from event_processing.vartemporal_plotting import (
    harmonic_band_plot,
    plot_image,
    vartemporal_plot,
)
from common.fitsread import (
    fits_save_events_generated,
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
    convert_to_2D_numpy_image,
    diag_run_detector_45,
    compute_beat_cube,
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
    save_fits_metadata,
    load_gen_param,
    save_chunk_metadata,
)
from event_processing.var_analysis_plots import (
    make_standard_plots,
    SDMC_plot,
    get_uneven_time_bin_widths,
)
from event_processing.binning import add_time_binning
from common.generate_data import generate_synthetic_telescope_data
import argparse

version = 1.0


def parse_arguments():
    default_pipeline = "antares_vs_anisogen"
    default_pipeline = "antares_vs_isogen"
    default_pipeline = "rnd_antares_vs_rnd_antares"
    parser = argparse.ArgumentParser(
        description="Compare two metadata files and optionally set downsampling."
    )
    parser.add_argument(
        "-p",
        "--pipe",
        type=str,
        default=default_pipeline,
        help="Select the experiment pipe name ",
    )
    parser.add_argument(
        "-a",
        "--meta_a",
        type=str,
        default=None,
        help="Path to first metadata JSON file (default: [].json)",
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


def wasserstein1_1d(p, q):
    """Wasserstein-1 in 1D for discrete distributions over ordered lags."""
    # assumes p,q nonnegative, sum to 1
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return float(np.abs(cdf_p - cdf_q).sum())


def cube_distance(A_acube, B_acube, normalize=True):
    """
    Distance between two beat-cubes:
      - build lag profiles P_A, P_B (W x L+1)
      - per-wavelength Wasserstein-1 along lag axis
      - average across wavelengths
    Returns:
      d_mean in [0,1] if normalize=True (divides by max possible distance = L)
    """
    assert len(A_acube.shape) == 3, "Expected shape for A to have length 3"
    assert len(B_acube.shape) == 3, "Expected shape for B to have length 3"

    P_A, _ = lag_profile(A_acube)
    P_B, _ = lag_profile(B_acube)
    assert P_A.shape == P_B.shape
    W, Lp1 = P_A.shape
    d_w = np.array([wasserstein1_1d(P_A[w], P_B[w]) for w in range(W)])
    d = d_w.mean()
    if normalize:
        d /= Lp1 - 1  # max W1 on 0..L is (L)
    return float(d), d_w


def structural_cosine(A_acube, B_acube):
    assert len(A_acube.shape) == 3, "Expected shape for A to have length 3"
    assert len(B_acube.shape) == 3, "Expected shape for B to have length 3"
    P_A, _ = lag_profile(A_acube)
    P_B, _ = lag_profile(B_acube)
    vA = (P_A - P_A.mean()).ravel()
    vB = (P_B - P_B.mean()).ravel()
    denom = np.linalg.norm(vA) * np.linalg.norm(vB) + 1e-12
    return float(np.dot(vA, vB) / denom)


def gini_from_probs(p):
    # p: (L,) nonnegative, sum=1
    L = p.size
    if L <= 1:
        return 0.0
    x = np.sort(p)
    cum = np.cumsum(x)
    # Gini for discrete probs
    g = 1 - 2 * (cum.sum() / (L * cum[-1])) + 1 / L
    return float(max(0.0, min(1.0, g)))


def temporal_coherence(beats_row):
    """
    beats_row: (T, L+1) for a fixed wavelength w, binary
    Returns a 0..1 coherence index: fraction of 1s that are part of 11 adjacent pairs.
    """
    if beats_row.ndim != 2:
        raise ValueError
    T, Lp1 = beats_row.shape
    if T < 2:
        return 0.0
    ones = beats_row.sum()
    if ones == 0:
        return 0.0
    adj = (beats_row[:-1] & beats_row[1:]).sum()
    # Count pairs per lag; weight by how many ones belong to a pair
    # Each 11 contributes 2 "paired ones"; cap at ones
    paired_ones = min(2 * adj, ones)
    return float(paired_ones / ones)


def detect_lightlane_signatures(
    pipe: ComparePipeline,
    meta: FitsMetadata,
    pp: ProcessingParameters,
    binned: DataFrame,
    handle: str,
    time_bin_widht_index: int,
    time_bin_widths: [],
):
    all_segments_dfs = None
    all_beat_cubes = []
    log = []
    while time_bin_widht_index < len(time_bin_widths):
        time_bin_width = time_bin_widths[time_bin_widht_index]
        for j in range(0, pp.phase_bins):
            image, wbin_axis, tbin_axis = convert_to_2D_numpy_image(
                binned_data=binned,
                time_bin_widht_index=time_bin_widht_index,
                phase_bin_index=j,
                time_bin_widths=time_bin_widths,
                deviation_from_mean=True,
                handle=handle,
            )
            # Compute the anisotropy cube

            tbw_anisotropy_cube = compute_beat_cube(pixelmap=image, max_lag=12, q=0.5)
            all_beat_cubes.append(tbw_anisotropy_cube)

            ridges, segments_dfs, arrows = diag_run_detector_45(
                image,
                time_bin_widht_index,
                time_bin_widths,
                sigma=3,
                tol_deg=2,
                min_len=10,
                max_gap=1,
                handle=handle,
            )

            if all_segments_dfs is None:
                all_segments_dfs = segments_dfs.copy()
            else:
                all_segments_dfs = pd.concat(
                    [all_segments_dfs, segments_dfs], ignore_index=True
                )

            title = f"Diagonal striations ({len(arrows)} {time_bin_width}s)"
            if len(arrows) > 0:
                plot_image(
                    image,
                    title,
                    "Time Bin Index",
                    "Wavelength Bin",
                    "Deviance from mean",
                    f"{pipe.id}/striations_{meta.id}_{time_bin_width:.0f}s.png",
                    show=False,
                    arrows=arrows,
                )

        time_bin_widht_index += 1

    # fits_save_cache(cached_filename, all_segments_dfs)

    return log, all_segments_dfs, all_beat_cubes


def compute_winning_beats(
    beat_sheet: np.array, time_bin_width_index, time_bin_widths, peak_threshold
) -> np.array:
    """
    Improved detection of periodicities per wavelength bin.

    Strategy:
      - ignore lag 0, smooth the lag-count profile to reduce single-bin noise
      - convert counts -> probabilities (mass at each lag)
      - pick up to three highest peaks whose fractional mass >= peak_threshold
      - convert lag indices -> seconds using the time bin width for the given index
      - return one row per wavelength: [w, lag1_s, p1, lag2_s, p2, lag3_s, p3]
    """
    import numpy as np

    arr = np.asarray(beat_sheet)
    if arr.ndim != 2:
        raise ValueError("beat_sheet must be a 2D array (wbin x lag_index)")

    dt = float(time_bin_widths[time_bin_width_index])
    n_wbins, Lp1 = arr.shape
    winners = []
    eps = 1e-12

    # small smoothing kernel to suppress single-bin spikes
    kernel = np.array([0.25, 0.5, 0.25], dtype=float)

    for w in range(n_wbins):
        row = np.asarray(arr[w], dtype=float).ravel()
        # require at least one lag (besides lag 0)
        if row.size <= 1 or np.nansum(row[1:]) == 0:
            continue

        counts = row[1:]  # ignore lag 0, counts indexed 0 -> lag 1
        total = counts.sum() + eps
        # normalize to probability mass function over lags
        p = counts / total

        # smooth the PMF to reduce noisy single-bin spikes (convolution, preserve length)
        padded = np.pad(p, (1, 1), mode="edge")
        smooth = np.convolve(padded, kernel, mode="valid")

        # descend order indices into smooth (relative indices -> lag = idx+1)
        desc_idx = np.argsort(-smooth)

        top_lags_s = []
        top_ps = []
        used = set()
        k = 0
        # select up to three peaks that meet threshold and are distinct
        for idx in desc_idx:
            if k >= 3:
                break

            k += 1
            # skip negligible bins
            if smooth[idx] <= 0:
                continue
            # fraction of mass at this lag (use original p to compute exact fraction)
            frac = float(p[idx])
            if frac < peak_threshold:
                # below absolute threshold, skip
                continue
            # simple de-duplication (avoid repeated neighboring picks)
            if any(abs(idx - u) <= 0 for u in used):
                # allow same bin only once
                continue
            used.add(idx)

            lag_idx = int(idx + 1)
            secs = float(lag_idx * dt)

            top_ps.append(float(frac))
            winners.append(
                np.array(
                    [int(w), lag_idx, secs, frac, len(top_lags_s)],
                    dtype=float,
                )
            )

        # pad to three entries

    return np.asarray(winners)


# ...existing code...


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

        pp = load_processing_param(pipeline_meta.pp_id)
        assert pp is not None, "Processing param was none"
        binning_is_cached, cached_filename = get_cached_filename("binning", metaA, pp)

        if binning_is_cached:
            binnedA = fits_read_cache_if_exists(cache_filename_path=cached_filename)

            logA = [f"A : Loaded cached binned data from {cached_filename}"]
        else:
            logA, metaA, binnedA = add_time_binning(
                pipeline=pipeline_meta, meta=metaA, pp=pp, handle="A", N=N
            )
            fits_save_cache(cached_filename, binnedA)

        wave_edges, wave_centers, wave_widths = get_wavelength_bins(pp)
        time_bin_widths = get_uneven_time_bin_widths(pp)
        time_bin_widht_index = 0

        logA, segments_dfs, A_beat_scores = detect_lightlane_signatures(
            pipe=pipeline_meta,
            meta=metaA,
            pp=pp,
            binned=binnedA,
            time_bin_widht_index=time_bin_widht_index,
            time_bin_widths=time_bin_widths,
            handle=f"A_{metaA.id}",
        )

        print(f"Beginning to read B")
        metaB = load_fits_metadata(pipeline_meta.B_fits_id)
        assert metaB.t_max > 0, "T_max has not been set. "
        pp = load_processing_param(pipeline_meta.pp_id)
        assert pp is not None, "Processing param was none"
        binning_is_cached, cached_filename = get_cached_filename("binning", metaB, pp)

        if binning_is_cached:
            binnedB = fits_read_cache_if_exists(cache_filename_path=cached_filename)
            logB = [f"B : Loaded cached binned data from {cached_filename}"]
        else:
            logB, metaA, binnedB = add_time_binning(
                pipeline=pipeline_meta, meta=metaA, pp=pp, handle="B", N=N
            )
            fits_save_cache(cached_filename, binnedB)

        time_bin_widths = get_uneven_time_bin_widths(pp)
        time_bin_widht_index = 0
        logB, segments_dfs, B_beat_scores = detect_lightlane_signatures(
            pipe=pipeline_meta,
            meta=metaB,
            pp=pp,
            binned=binnedB,
            time_bin_widht_index=time_bin_widht_index,
            time_bin_widths=time_bin_widths,
            handle=f"B_{metaB.id}",
        )

        assert len(B_beat_scores) == len(
            A_beat_scores
        ), "Expect beat cubes to have some number of rows!"

        A_winners = []
        B_winners = []
        peak_threshold = 0.15
        for i in range(0, len(A_beat_scores)):
            A_beat_score = A_beat_scores[i]
            B_beat_score = B_beat_scores[i]

            A_winner = compute_winning_beats(
                A_beat_score, time_bin_widht_index, time_bin_widths, peak_threshold
            )
            B_winner = compute_winning_beats(
                B_beat_score, time_bin_widht_index, time_bin_widths, peak_threshold
            )

            A_winners.extend(A_winner)
            B_winners.extend(B_winner)

        filename = f"plots/{pipeline_meta.id}/period_winners_A_{metaA.id}.png"
        plot_periodicity_winners(A_winners, None, wave_centers, filename, True)

        filename = f"plots/{pipeline_meta.id}/period_winners_B_{metaB.id}.png"
        plot_periodicity_winners(None, B_winners, wave_centers, filename, True)

    print("LOG : ")
    for logline in log:
        print(logline)

    print("\n".join(log))


if __name__ == "__main__":
    main()
