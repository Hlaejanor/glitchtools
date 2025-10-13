from common.fitsmetadata import (
    FitsMetadata,
    Spectrum,
    ChunkVariabilityMetadata,
    ProcessingParameters,
    ComparePipeline,
)
import os
from common.fitsread import (
    read_event_data_crop_and_project_to_ccd,
    get_cached_filename,
    fits_read_cache_if_exists,
    fits_save_cache,
    save_chunk_metadata,
)
import numpy as np
import pandas as pd

from common.metadatahandler import save_fits_metadata, load_chunk_metadata
from common.powerdensityspectrum import PowerDensitySpectrum, compute_spectrum_params
from common.helper import get_uneven_time_bin_widths
from tabulate import tabulate
from pandas import DataFrame
from event_processing.plotting import (
    compute_time_variability_async,
    take_max_variability_per_wbin,
)


def load_or_compute_chunk_variability_observations(
    pipe: ComparePipeline,
    meta: FitsMetadata,
    pp: ProcessingParameters,
    handle: str,
):
    assert pp is not None, "Processing param was none"
    log = []

    chunkvar_id = f"{meta.id}_{pp.id}"
    chunk_meta = load_chunk_metadata(chunkvar_id)

    chunkvar_is_cached, cached_filename = get_cached_filename("chunkvar", meta, pp)

    if not chunkvar_is_cached:
        print(
            "Must generate variability computations using Monte Carlo method, this can take some time"
        )

        log, meta, source = load_source_data(meta, pp)

        log, meta, binned_datasets, time_bin_widths = add_time_binning(
            pipeline=pipe, source_data=source, meta=meta, pp=pp, handle=handle
        )
        # The name of the chunk dataset for this observation

        variability_observations, chunk_meta = compute_time_variability_async(
            binned_datasets=binned_datasets,
            meta=meta,
            pp=pp,
            time_bin_widths=time_bin_widths,
        )
        print(variability_observations.head(200))
        fits_save_cache(cached_filename, variability_observations)
        save_chunk_metadata(chunk_meta)
        # set chunk_variability to the just-computed table so downstream code can use it
        chunk_variability = variability_observations
    else:
        print("Can use cached ChunkVar table!")
        # Load the chunk metadata, which is now either generated or loaded from disk
        chunk_meta = load_chunk_metadata(chunkvar_id)

        # Load the variability computed from chunks
        chunk_variability = fits_read_cache_if_exists(cached_filename)
        if chunk_variability is None:
            raise Exception("Cache was supposed to exist!")

    # Read the chunked variability observation data from the fits file

    return log, chunk_variability, cached_filename


def load_source_data(
    meta: FitsMetadata, pp: ProcessingParameters
) -> tuple[list, FitsMetadata, pd.DataFrame]:
    source_data_is_cached, cached_filename = get_cached_filename("source", meta, pp)
    log = []
    if source_data_is_cached:
        source_data = fits_read_cache_if_exists(cache_filename_path=cached_filename)
        log = [f"A : Loaded cached binned data from {cached_filename}"]
    else:
        # Add the wavelength binning

        success, meta, pp, source_data = read_event_data_crop_and_project_to_ccd(
            meta.id, pp.id
        )
        source_data = add_wavelength_bin_columns(meta, pp, source_data)
        meta.t_max = int(source_data["relative_time"].max())
        meta.t_min = int(source_data["relative_time"].min())
        # Compute the spectrum if for some reason this was not already set
        meta.apparent_spectrum, _ = compute_spectrum_params(meta, pp, source_data)
        # Save the spectrum params
        source_data_is_cached, cached_filename = get_cached_filename("source", meta, pp)
        fits_save_cache(cached_filename, source_data)
        save_fits_metadata(meta)

    return (
        log,
        meta,
        source_data,
    )


def add_time_binning(
    pipeline: ComparePipeline,
    source_data: DataFrame,
    meta: FitsMetadata,
    pp: ProcessingParameters,
    handle: str,
) -> list[list, FitsMetadata, DataFrame]:
    log = []
    log.append(
        f"{handle} : Load and process fits file - id:{meta.id} {meta.raw_event_file}"
    )

    # print(source_reduced.head(2000))
    # source_reduced = randomly_sample_from(source_data, N)
    log.append(f"{handle} : Process dataset - id:{meta.id} {meta.raw_event_file}")
    time_bin_widths = get_uneven_time_bin_widths(pp)
    binned_datasets, meta = binning_process_distributed(
        source_data, meta, pp, time_bin_widths
    )

    log.append(
        f"{handle} : Computing average variability per wavelength bin using :{pp.time_bin_seconds}s time bins and {pp.wavelength_bins} wavelength bins"
    )

    save_fits_metadata(meta)
    return log, meta, binned_datasets, time_bin_widths


def equalize_wavelength_bins(data: DataFrame):
    print(
        "Upsample strategy 'equalize' indicated. Will duplicate observations until all bins have the same nuber of counts as the maximum count"
    )
    bin_counts = data["Wavelength Bin"].value_counts().sort_index()
    max_count = bin_counts.max()
    # Step 2: Determine how many photons are missing in each bin
    padding_needed = max_count - bin_counts

    # Step 3: Create padded dataset
    padding_hits = []

    for bin_val, missing_count in padding_needed.items():
        if missing_count > 0:
            bin_df = data[data["Wavelength Bin"] == bin_val]
            if not bin_df.empty:
                sampled = bin_df.sample(n=missing_count, replace=True).copy()
                # Jitter time by +/- 10% of the time range in that bin
                jitter_range = 1.0 * (
                    sampled["relative_time"].max() - sampled["relative_time"].min()
                )
                sampled["relative_time"] += np.random.uniform(
                    -jitter_range, jitter_range, size=missing_count
                )
                padding_hits.append(sampled)

    # Step 4: Combine original and padding datasets
    if len(padding_hits) > 0:
        padding_df = pd.concat(padding_hits, ignore_index=True)
        data = pd.concat([data, padding_df], ignore_index=True)
    return data


def downsample_flatten(data: DataFrame):
    print(
        "Downsample strategy 'flatten' indicated. Will downsample within energy bounds and ensure that all wavelength bins have the same count"
    )
    bin_counts = data["Wavelength Bin"].value_counts().sort_index()
    min_count = bin_counts.min()
    # flattened_bins = pd.DataFrame(columns=data.columns)
    samples = []
    for bin_val in bin_counts.index:
        bin_df = data[data["Wavelength Bin"] == bin_val]
        samples.append(bin_df.sample(n=min_count, replace=False))

    return pd.concat(samples, ignore_index=True), min_count


def add_wavelength_bin_columns(
    meta: FitsMetadata, pp: ProcessingParameters, source_data: DataFrame
):
    assert (
        "Wavelength (nm)" in source_data.columns
    ), "Expected column  'Wavelength (nm)' in source data here"

    min_lambda = meta.get_min_wavelength()
    max_lambda = meta.get_max_wavelength()
    print(
        f"Filtering the dataset based on high and low wavelength. From [{min_lambda:.2f}, {max_lambda:.2f}]nm"
    )

    # Example output: [0.   1.   2.   3.   4.  ] if max_time was ~3.14

    # Filter wavelengths
    valid_range = source_data["Wavelength (nm)"] >= min_lambda
    source_data = source_data[valid_range].reset_index(drop=True)
    valid_range = source_data["Wavelength (nm)"] <= max_lambda

    # WBIN BINNING
    lamb_min = source_data["Wavelength (nm)"].min()
    lamb_max = source_data["Wavelength (nm)"].max()

    if np.isnan(lamb_min):
        raise Exception("Lamb min is nan")

    if np.isnan(lamb_max):
        raise Exception("Lamb min is nan")

    wave_edges = np.linspace(
        lamb_min,
        lamb_max,
        pp.wavelength_bins + 1,
    )
    print("Bin edges:", wave_edges)
    wave_centers = 0.5 * (wave_edges[:-1] + wave_edges[1:])
    wave_widths = np.diff(wave_edges)

    source_data["Wavelength Bin"] = pd.cut(
        source_data["Wavelength (nm)"],
        bins=wave_edges,
        labels=False,
        include_lowest=True,
    )

    source_data.sort_values(["Wavelength Bin"], inplace=True)
    source_data["Wavelength Center"] = source_data["Wavelength Bin"].apply(
        lambda i: wave_centers[i] if pd.notnull(i) else np.nan
    )

    source_data["Wavelength Width"] = source_data["Wavelength Bin"].apply(
        lambda i: wave_widths[i] if pd.notnull(i) else np.nan
    )

    source_data = source_data[valid_range].reset_index(drop=True)
    return source_data


def perform_downsampling(
    meta: FitsMetadata, pp: ProcessingParameters, source_data: DataFrame
):
    # DOWNSAMPLE :
    if pp.downsample_strategy == "flatten":
        source_data, min_count = downsample_flatten(source_data)
        assert all(source_data["Wavelength Bin"].value_counts() == min_count)
    elif pp.downsample_strategy == "reduce":
        source_data = source_data.sample(pp.downsample_target_count)

    if pp.padding_strategy == "equalize":
        source_data = equalize_wavelength_bins(source_data)
        # Step 1: Count photons in each bin

    return source_data


import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import Optional, Tuple


def cut_dataset_simple(
    source_data: DataFrame,
    time_column="time",
    time_bin_width: float = 1,
    out_col: str = "time_bin",
):
    t_min = float(source_data[time_column].min())
    t_max = float(source_data[time_column].max())
    start = t_min - time_bin_width
    stop = t_max + time_bin_width + 1e-12  # tiny epsilon so rightmost edge is included
    n_edges = int(np.floor((stop - start) / time_bin_width)) + 1
    base_edges = (start + np.arange(n_edges) * time_bin_width).astype(float)

    # Apply the scalar offset
    bin_edges = base_edges

    # Cut; include_lowest puts t == leftmost edge into bin 0
    codes = pd.cut(
        source_data[time_column].to_numpy(),
        bins=bin_edges,
        labels=False,
        include_lowest=True,
        right=False,  # half-open bins [left, right)
    )

    # At this point, padding should ensure no NaNs. If any, drop or fill safely.
    if pd.isna(codes).any():
        # Drop rows that fell outside due to numerical quirks
        mask = ~pd.isna(codes)
        df = df.loc[mask].copy()
        codes = codes[mask]

    source_data[out_col] = codes.astype(int)

    return source_data


def cut_dataset_with_time_bin_width_and_offset(
    source_data: DataFrame,
    time_bin_width: float,
    offset_fraction: Optional[float] = None,
    jitter_eps: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    time_col: str = "relative_time",
    out_col: str = "time_bin",
) -> Tuple[DataFrame, np.ndarray, float]:
    """
    Bin events into time bins of width `time_bin_width`, with a single scalar offset.
    - Pads the range by ±1 bin to avoid edge loss after offset.
    - Optionally applies multiplicative jitter to the bin width.
    - Returns: (dataset_with_bins, bin_edges_used, offset_seconds)

    Parameters
    ----------
    source_data : DataFrame
        Must contain a time column `time_col` in seconds (float).
    time_bin_width : float
        Nominal bin width (seconds).
    offset_fraction : Optional[float]
        Fraction of bin width in [-0.5, 0.5) to shift bin edges. If None, drawn U[-0.5, 0.5).
    jitter_eps : float
        Log-jitter magnitude. If > 0, actual Δt = Δt * exp(U[-eps, eps]).
    rng : Optional[np.random.Generator]
        For reproducibility. If None, uses np.random.default_rng().
    time_col : str
        Name of the time column.
    out_col : str
        Name of the output bin-index column.
    """

    if rng is None:
        rng = np.random.default_rng()

    df = source_data.copy()
    # Ensure sorted by time (pd.cut does not require it, but downstream often does)
    df = df.sort_values(time_col, kind="mergesort").reset_index(drop=True)

    t_min = float(df[time_col].min())
    t_max = float(df[time_col].max())
    if not np.isfinite(t_min) or not np.isfinite(t_max):
        raise ValueError("Time column contains non-finite values.")

    # Optional multiplicative log-jitter on the bin width
    if jitter_eps and jitter_eps > 0:
        jitter = np.exp(rng.uniform(-jitter_eps, jitter_eps))
        dt = time_bin_width * jitter
    else:
        dt = time_bin_width

    # Choose a scalar offset in fractions of dt
    if offset_fraction is None:
        off_frac = rng.uniform(-0.5, 0.5)
    else:
        off_frac = float(offset_fraction)
        if not (-0.5 <= off_frac < 0.5):
            raise ValueError("offset_fraction must be in [-0.5, 0.5).")
    offset = off_frac * dt

    # Pad by one bin on each side to avoid losing events after offset
    # Build edges aligned to t_min, then expand to cover [t_min - dt, t_max + dt]
    start = t_min - dt
    stop = t_max + dt + 1e-12  # tiny epsilon so rightmost edge is included
    n_edges = int(np.floor((stop - start) / dt)) + 1
    base_edges = (start + np.arange(n_edges) * dt).astype(float)

    # Apply the scalar offset
    bin_edges = base_edges + offset

    # Cut; include_lowest puts t == leftmost edge into bin 0
    codes = pd.cut(
        df[time_col].to_numpy(),
        bins=bin_edges,
        labels=False,
        include_lowest=True,
        right=False,  # half-open bins [left, right)
    )

    # At this point, padding should ensure no NaNs. If any, drop or fill safely.
    if pd.isna(codes).any():
        # Drop rows that fell outside due to numerical quirks
        mask = ~pd.isna(codes)
        df = df.loc[mask].copy()
        codes = codes[mask]

    df[out_col] = codes.astype(int)

    # Reindex time_bin to start at 0 (optional but nice)
    # This keeps relative bin numbering stable even if we padded.
    first_bin = int(df[out_col].min())
    df[out_col] = df[out_col] - first_bin
    bin_edges = bin_edges - bin_edges[first_bin]

    return df, bin_edges, offset


def binning_process_distributed(
    source_data: DataFrame,
    meta: FitsMetadata,
    pp: ProcessingParameters,
    time_bin_widths,
) -> tuple[DataFrame, FitsMetadata]:
    try:
        """Loads a metadata file"""
        # meta = load_fits_metadata(filename)
        # source_data = pd.read_csv(f"temp/{meta.source_filename}")
        # background_data = pd.read_csv(meta.background_filename)

        if pp.take_time_seconds:
            duration = pp.take_time_seconds
        else:
            duration = meta.t_max
        # min_time_in_data = source_data["relative_time"].min()
        max_time_in_data = source_data["relative_time"].max()

        if max_time_in_data < duration:
            raise Exception(
                f"Dataset shorter than duration request. Saw time {max_time_in_data} but only {duration} is allowed"
            )

        # Drop unecessary columns for this analysis
        # source_data.drop()
        if "Wavelength Bin" not in source_data.columns:
            print("Must add binning on wavelength here, this was not set")
            source_data = add_wavelength_bin_columns(meta, pp, source_data)

        source_data = perform_downsampling(meta, pp, source_data)
        if len(source_data) == 0:
            return source_data, meta

        datasets = []
        # Create bin_width offset cube
        for time_bin_width in time_bin_widths:
            df = source_data.copy()

            df, bin_edges, offset = cut_dataset_with_time_bin_width_and_offset(
                source_data, time_bin_width
            )
            datasets.append(df)

        return datasets, meta

    except Exception as e:
        print(f"Exception occured in binning process {repr(e)}")
        raise e

    return source_data, meta


def make_time_bins(duration, time_bin_width_seconds):
    time_edges = np.arange(0, duration, time_bin_width_seconds)

    return time_edges


def binning_process(
    source_data: DataFrame, meta: FitsMetadata, pp: ProcessingParameters
) -> tuple[DataFrame, FitsMetadata]:
    try:
        """Loads a metadata file"""
        # meta = load_fits_metadata(filename)
        # source_data = pd.read_csv(f"temp/{meta.source_filename}")
        # background_data = pd.read_csv(meta.background_filename)

        if pp.take_time_seconds:
            duration = pp.take_time_seconds
        else:
            duration = meta.t_max

        max_time_in_data = source_data["relative_time"].max()

        if max_time_in_data - 1 > duration:
            raise Exception(
                f"Dataset should have been cropped in time. Saw time {max_time_in_data} but only {duration} is allowed"
            )

        min_lambda = meta.get_min_wavelength()
        max_lambda = meta.get_max_wavelength()
        print(
            f"Filtering the dataset based on high and low wavelength. From [{min_lambda:.2f}, {max_lambda:.2f}]nm"
        )
        if np.abs(min_lambda) < 0.001:
            print(" Somehow the min lambda is too low!")
        if np.abs(max_lambda) < 0.001:
            print(" Somehow the max lambda is too low!")
        valid_range = source_data["Wavelength (nm)"] >= min_lambda
        source_data = source_data[valid_range].reset_index(drop=True)
        valid_range = source_data["Wavelength (nm)"] <= max_lambda

        source_data = source_data[valid_range].reset_index(drop=True)
        if len(source_data) == 0:
            return source_data, meta
        # Suppose we want 1-second time bins:
        print(source_data.head(200))
        bin_edges = make_time_bins(duration, pp.time_bin_seconds)
        if len(bin_edges) <= 1:
            raise Exception("Problem occured, needs more than one bin edge")
        print("Bin edges:", bin_edges)
        # Example output: [0.   1.   2.   3.   4.  ] if max_time was ~3.14

        # First we bin the events based on the number of bin edges
        source_data["Time Bin 0"] = pd.cut(
            source_data["relative_time"],
            bins=bin_edges,
            labels=False,
            include_lowest=True,
        )

        lamb_min = source_data["Wavelength (nm)"].min()
        lamb_max = source_data["Wavelength (nm)"].max()

        if np.isnan(lamb_min):
            raise Exception("Lamb min is nan")

        if np.isnan(lamb_max):
            raise Exception("Lamb min is nan")

        wave_edges = np.linspace(
            lamb_min,
            lamb_max,
            pp.wavelength_bins + 1,
        )
        print("Bin edges:", wave_edges)
        wave_centers = 0.5 * (wave_edges[:-1] + wave_edges[1:])
        wave_widths = np.diff(wave_edges)

        source_data["Wavelength Bin"] = pd.cut(
            source_data["Wavelength (nm)"],
            bins=wave_edges,
            labels=False,
            include_lowest=True,
        )

        source_data.sort_values(["Wavelength Bin"], inplace=True)
        source_data["Wavelength Center"] = source_data["Wavelength Bin"].apply(
            lambda i: wave_centers[i] if pd.notnull(i) else np.nan
        )

        source_data["Wavelength Width"] = source_data["Wavelength Bin"].apply(
            lambda i: wave_widths[i] if pd.notnull(i) else np.nan
        )

        print("Estimate spectrum before ")
        generated_spectrum, r_squared = compute_spectrum_params(
            meta=meta, pp=pp, source_data=source_data
        )
        meta.apparent_spectrum = generated_spectrum
        save_fits_metadata(meta)

        if pp.downsample_strategy == "flatten":
            source_data, min_count = downsample_flatten(source_data)
        elif pp.downsample_strategy == "reduce":
            source_data = source_data.sample(pp.downsample_target_count)

        if pp.padding_strategy == "equalize":
            source_data = equalize_wavelength_bins(source_data)
            # Step 1: Count photons in each bin

        print(f"Fitted Spectrum with residual error {r_squared:.3f}")
        print(generated_spectrum.to_string())
        # print(f"Estimating spectrum params from {len(count_per_lambda_bin)} obs")

        meta.apparent_spectrum = generated_spectrum

        save_fits_metadata(meta=meta)

    except Exception as e:
        print(f"Exception occured in binning process {repr(e)}")
        raise e
    return source_data, meta
