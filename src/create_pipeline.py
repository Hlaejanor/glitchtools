import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from common.helper import compare_dataframes, ensure_path_exists, get_duration
from event_processing.plotting import (
    plot_spectrum_vs_data,
    compute_time_variability_async,
)
from event_processing.binning import (
    load_source_data,
    cut_dataset_simple,
    get_binned_datasets,
)
from event_processing.chandratimeshift import chandrashift
from event_processing.binning import (
    add_wavelength_bin_columns,
    cut_dataset_with_time_bin_width_and_offset,
)
from common.powerdensityspectrum import (
    compute_spectrum_params,
    exp_count_per_sec,
    PowerDensitySpectrum,
)
from common.metadatahandler import (
    load_fits_metadata,
    save_fits_metadata,
    load_gen_param,
    save_gen_param,
    load_pipeline,
    save_pipeline,
    save_processing_metadata,
    load_processing_param,
    load_chunk_metadata,
    save_chunk_metadata,
)
from common.helper import (
    compare_variability_profiles,
    ensure_pipeline_folders_exists,
    ensure_path_exists,
    get_wavelength_bins,
)

from common.fitsread import (
    fits_save_events_with_pi_channel,
    fits_to_dataframe,
    pi_channel_to_wavelength_and_width,
    read_event_data_crop_and_project_to_ccd,
    fits_save_chunk_analysis,
    fits_read,
    fits_save_cache,
    get_cached_filename,
    fits_read_cache_if_exists,
    fits_save_event_file,
)

from common.fitsmetadata import (
    FitsMetadata,
    Spectrum,
    ProcessingParameters,
    ComparePipeline,
    GenerationParameters,
)
from common.helper import randomly_sample_from

from event_processing.binning import binning_process_distributed
from common.generate_data import generate_synthetic_telescope_data
import csv
import os
import sys
import itertools
import uuid
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import argparse
from pandas import DataFrame


def run_all_pipelines():
    """
    Discover all JSON pipeline metadata files and run each pipeline.
    Returns a dict with 'success' and 'failed' lists.
    """
    print(f"Running all pipes")
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
            logs = run_pipeline(pipeline_meta)
            summary["success"].append(
                {"pipeline_id": pipeline_meta.id, "success": True, "logs": logs}
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


def run_pipeline(pipe: ComparePipeline):
    assert pipe is not None, "Pipe needs to be something, cant run None pipe"
    log = []
    log.append(f"Creating pipeline {pipe.id}")
    log.append(f"Plan : ")
    if pipe.source is not None and pipe.gen_id is None:
        log.append(
            f"1. Load source dataset {pipe.source}, make a copy called {pipe.A_fits_id}"
        )
    elif pipe.source is None and pipe.gen_id is not None:
        log.append(
            f"1. Generate dataset in slot A using gen_id {pipe.gen_id}, call it {pipe.A_fits_id}"
        )
    else:
        log.append(
            f"1.Use Spectrum from source {pipe.gen_id}, call it {pipe.A_fits_id}"
        )

    pp = load_processing_param(pipe.pp_id)
    assert (
        pp is not None
    ), "Processing param was none, use default or make sure that meta file exists"
    log.append(
        f"2. Use {pipe.pp_id} processing parameters , with the following settings : max_wavelength {pp.max_wavelength} min_wavelength {pp.min_wavelength} wavelength_bins {pp.wavelength_bins}"
    )
    log.append(f"3. Apply tasks {pipe.A_tasks} to dataset A")
    log.append(f"4. Apply tasks {pipe.B_tasks} to dataset B")
    log.append(
        f"5. Save the resulting datasets as {pipe.A_fits_id} and {pipe.B_fits_id}"
    )
    log.append(
        f"6. Save plots of the spectrums of A and B to the plots/<pipe.id>/ folder"
    )

    log.append(
        f"Note : This will prepare the datasets only. To make plots, run var_analysis run var_analysis.sh -p {pipe.id}"
    )
    # Load the processing params

    assert (
        pp is not None
    ), "Processing param was none, use default or make sure that meta file exists"

    # Load metadata for target set A
    # meta_A = load_fits_metadata(pipe.A_fits_id)
    # assert pp is not None, "A slot in pipeline cannot be None"
    meta_A, meta_B = task_handler(pipe, pp)
    pipe.A_fits_id = meta_A.id
    pipe.B_fits_id = meta_B.id
    save_pipeline(pipe)
    log.append(
        f"{pipe.id} : Saved pipeline A : {pipe.A_fits_id} and B : {pipe.B_fits_id}"
    )

    log_A, meta_A, source_A = load_source_data(meta_A, pp)
    log_B, meta_B, source_B = load_source_data(meta_B, pp)
    log.extend(log_A)
    log.extend(log_B)
    log.append(
        f"Loaded A: {source_A.shape[0]} events from {meta_A.id} file {meta_A.raw_event_file}"
    )
    log.append(
        f"Loaded B: {source_B.shape[0]} events from {meta_B.id} file {meta_A.raw_event_file}"
    )

    ensure_pipeline_folders_exists(pipe)
    filename = f"plots/{pipe.id}/plot_spectrum_A_vs_B.png"
    # Show that the computed spectrum matches the actual data
    log.append(f"{pipe.id} : Plotting spectrum")
    plot_spectrum_vs_data(
        meta_A=meta_A,
        source_A=source_A,
        meta_B=meta_B,
        source_B=source_B,
        filename=filename,
        show=False,
    )

    log.append(f"Saved spectrum comparison {filename}")
    return log


def equalize_wavelength_bins(data: DataFrame, pp: ProcessingParameters):
    print(
        "Upsample strategy 'equalize' indicated. Will duplicate observations until all bins have the same nuber of counts as the maximum count"
    )
    bin_counts = data["Wavelength Bin"].value_counts().sort_index()
    max_count = bin_counts.max()
    # Step 2: Determine how many photons are missing in each bin
    padding_needed = max_count - bin_counts

    # Step 3: Create padded dataset
    padding_hits = []
    jitter_half_bin = pp.time_bin_seconds / 2
    for bin_val, missing_count in padding_needed.items():
        if missing_count > 0:
            bin_df = data[data["Wavelength Bin"] == bin_val]
            if not bin_df.empty:
                sampled = bin_df.sample(n=missing_count, replace=True).copy()
                # Jitter time by +/- 10% of the time range i./n that bin
                sampled["time"] += np.random.uniform(
                    -jitter_half_bin, jitter_half_bin, size=missing_count
                )
                padding_hits.append(sampled)

    # Step 4: Combine original and padding datasets
    if len(padding_hits) > 0:
        padding_df = pd.concat(padding_hits, ignore_index=True)
        data = pd.concat([data, padding_df], ignore_index=True)
    return data


def downsample_flatten(data: DataFrame, pp: ProcessingParameters):
    print(
        "Downsample strategy 'flatten' indicated. Will downsample within energy bounds and ensure that all wavelength bins have the same count"
    )
    assert (
        "Wavelength Bin" in data.columns
    ), "Probably missing wbin operation before flatten. Reason : Wavelength Bin column must be present to perform flatten operation. Try wbin,flatten sequence in tasks"
    # data = add_wavelength_bin_columns(source_data=data, pp=pp)

    bin_counts = data["Wavelength Bin"].value_counts().sort_index()
    min_count = bin_counts.min()
    # flattened_bins = pd.DataFrame(columns=data.columns)
    samples = []
    for bin_val in bin_counts.index:
        bin_df = data[data["Wavelength Bin"] == bin_val]
        samples.append(bin_df.sample(n=min_count, replace=False))

    return pd.concat(samples, ignore_index=True), min_count


def poissonize_homogeneous(
    source_data: pd.DataFrame,
    poissonisation_percent: float,
    meta: FitsMetadata,
    time_column: str = "time",
    rng=None,
):
    """
    Generate homogeneous Poisson event times, but preserve spectrum shape and count
    on [t_min, t_max).
    If N is given, draw exactly n_fixed times i.i.d.
    """

    t_min = float(source_data[time_column].min())
    t_max = float(source_data[time_column].max())

    # Unconditional simulation via exponential gaps

    take = int(source_data.shape[0] * poissonisation_percent)

    if poissonisation_percent == 100:
        poissonize = source_data
        leftover = None
    else:
        poissonize = source_data.sample(take)
        leftover = source_data.sample(source_data.shape[0] - take)

    N = poissonize.shape[0]
    rng = np.random.default_rng() if rng is None else rng
    times = rng.uniform(t_min, t_max, size=N)

    poissonize.sort_values(time_column, inplace=True)
    poissonize[time_column] = times
    if leftover is not None:
        source_data = pd.concat(poissonize, leftover)
    else:
        source_data = poissonize

    source_data.sort_values(time_column, inplace=True)
    meta.t_min = t_min
    meta.t_max = t_max
    return source_data, meta


def parse_arguments(use_defaults: bool):
    parser = argparse.ArgumentParser(
        description="Create mimic dataset: Generate synthetic data variants."
    )
    if use_defaults:
        parser.add_argument(
            "-p",
            "--pipe",
            required=not use_defaults,
            default="antares_vs_rnd_antares_f100",
            type=str,
            help="Set the pipeline you want to run",
        )
        parser.add_argument(
            "-ch",
            "--chunk",
            required=not use_defaults,
            default="force",
            type=str,
            help="True for running chunking if necessary, force for running all chunking steps otherwise do not chunk",
        )
        parser.add_argument(
            "-c",
            "--create",
            required=False,
            default=False,
            type=str,
            help="Pass -c to create pipeline by specifying all other parameters, otherwise will just read -pipe from disk",
        )
        parser.add_argument(
            "-m",
            "--multiply",
            required=False,
            default=None,
            type=str,
            help="Pass to multiply a given pipe, and modify the parameters",
        )

        parser.add_argument(
            "-s",
            "--source",
            required=not use_defaults,
            default=None,
            type=str,
            help="What is the name of thse source dataset",
        )
        parser.add_argument(
            "-a",
            "--a_id",
            required=not use_defaults,
            default=None,
            type=str,
            help="Set the A dataset you want to test",
        )
        parser.add_argument(
            "-b",
            "--b_id",
            required=not use_defaults,
            default=None,
            type=str,
            help="Set the B dataset you want to use as a null",
        )
        parser.add_argument(
            "-pp",
            "--pp_id",
            required=not use_defaults,
            default=None,
            type=str,
            help="Set the Processing parameters, will default to 'default'",
        )
        parser.add_argument(
            "-g",
            "--gen_id",
            required=not use_defaults,
            default=None,
            type=str,
            help="Set the generation parameters",
        )
        parser.add_argument(
            "-tA",
            "--A_tasks",
            required=not use_defaults,
            default=[],
            type=str,
            help="Tasks and filters to perform for dataset A",
        )
        parser.add_argument(
            "-tB",
            "--B_tasks",
            required=not use_defaults,
            default=[],
            type=str,
            help="Tasks and filters to perform for dataset B",
        )

    args = parser.parse_args()

    return args


def convert_pi_channel_to_wavelength(
    data: DataFrame, pp: ProcessingParameters
) -> DataFrame:
    assert "pi" in data.columns, "PI channel column 'pi' is missing from data"
    print("Applying pi channel to wavelength mapping")
    # Convert pi channel to wavelength center
    wavelength_nm, delta_lambda_nm, pi_channel = pi_channel_to_wavelength_and_width(
        data["pi"]
    )
    # Jitter wavelength a bit to avoid bin edge effects

    wavelength_nm += (
        np.random.uniform(-0.5, 0.5, size=len(wavelength_nm)) * delta_lambda_nm
    )

    # Add wavelength columns
    data["Wavelength (nm)"] = wavelength_nm

    return data


def apply_task(
    task: str,
    events: DataFrame | None,
    event_meta: FitsMetadata | None,
    new_meta_id: str,
    pp: ProcessingParameters,
    genparam: GenerationParameters | None = None,
    event_data_A: DataFrame | None = None,
    handle: str = None,
) -> tuple[FitsMetadata, DataFrame]:
    print(f"Applying task {task} to dataset {new_meta_id}")
    if task not in [
        "generate",
        "prunecolumns",
        "chandrashift",
        "wbin",
        "wrange",
        "trange",
        "flatten",
        "tthinning",
        "downsample",
        "equalize",
        "poissonize",
        "copy",
    ]:
        raise ValueError(f"Unknown task {task}")

    if task in ["generate"]:
        assert (
            genparam is not None
        ), "Generation parameters must be provided when generating"
    elif task in [
        "chandrashift",
        "prunecolumns",
        "flatten",
        "reduce",
        "equalize",
        "poissonize",
        "copy",
    ]:
        assert (
            event_meta is not None and events is not None
        ), f"Both event metadata and events must be provided when calling {task}"

    if task in ["copyA"]:
        assert event_data_A is not None, "Event data A must be provided to copy A"
    if task == "prunecolumns":
        assert events is not None, "Dataframe empty"

        keep_columns = ["time", "x", "y", "pi"]
        keep_str = ",".join(keep_columns)
        print(f"Dropping all columns except {keep_str}")
        # Build list of columns to drop without mutating the DataFrame during iteration
        dropcolumns = [col for col in events.columns if col not in keep_columns]
        if len(dropcolumns) > 0:
            events = events.drop(columns=dropcolumns)
            print(f"Dropped columns: {', '.join(dropcolumns)}")
        return event_meta, events

    if task == "tthinning":
        events, bin_edges, offset = cut_dataset_with_time_bin_width_and_offset(
            events, pp.time_bin_seconds
        )

        # Group by wavelength bin
        timebin_groups = events.groupby("time_bin")

        # Compute mean count per wavelength bin
        timebin_counts = timebin_groups.size()
        avg_timebin_count = int(np.round(timebin_counts.mean()))

        # Downsample each wavelength bin group to the average count
        thinned_subsets = []
        for wl, group in timebin_groups:
            n = len(group)
            if n > avg_timebin_count:
                subset = group.sample(avg_timebin_count, random_state=42)
            else:
                subset = group  # keep all if already below mean
            thinned_subsets.append(subset)

        thinned_events = pd.concat(thinned_subsets, ignore_index=True)

        return event_meta, thinned_events

    if task == "trange":
        t0 = events["time"].min()
        events["time"] = events["time"] - t0
        print(f"Length 2.0 {len(events)}")

        if pp.end_time_seconds is not None:
            assert (
                pp.start_time_seconds is not None
            ), "If end_time_seconds is set, then from_time_seconds must also be set"
            print(
                f"Taking only events with timestamp between in range [{pp.start_time_seconds},  {pp.end_time_seconds}] seconds"
            )
            t_mask = events["time"].between(pp.start_time_seconds, pp.end_time_seconds)
            return event_meta, events[t_mask]
        else:
            return event_meta, events

    if task == "generate":
        if event_meta is not None and event_meta.apparent_spectrum is not None:
            genparam.spectrum = event_meta.apparent_spectrum
            save_gen_param(genparam)

        event_meta = FitsMetadata(
            id=f"{new_meta_id}",
            raw_event_file=f"fits/{new_meta_id}.fits",
            synthetic=True,
            source_pos_x=0.0,
            source_pos_y=0.0,
            max_energy=genparam.get_maximum_energy(),
            min_energy=genparam.get_minimum_energh(),
            source_count=None,
            star=genparam.star,
            t_min=genparam.t_min,
            t_max=genparam.t_max,
            gen_id=genparam.id,
            ascore=None,
            apparent_spectrum=event_meta.apparent_spectrum
            if event_meta is not None
            else None,
        )

        save_fits_metadata(event_meta)

        event_list = generate_synthetic_telescope_data(genparam, pp)

        # Save the the generated dataset as a fits file
        events, event_meta = fits_save_events_with_pi_channel(
            event_list, genparam, use_this_id=event_meta.id
        )
        print(f"Generated {len(events)} events for dataset {event_meta}")

        return event_meta, events
    if task == "chandrashift":
        assert (
            events is not None
        ), "Source data must be provided to chandrashift. You should probably specify the source dataset (and it should be from Chandra)"
        events = chandrashift(events)
        print(f"Chandra time shift applied to {event_meta.id}")
        event_meta = fits_save_event_file(events, event_meta)
        return event_meta, events
    if task == "wbin":
        assert (
            events is not None
        ), "Source data must be provided to add wavelength bins."
        if "Wavelength (nm)" not in events.columns:
            events = convert_pi_channel_to_wavelength(events, pp)
        if "Wavelength Bin" not in events.columns:
            events = add_wavelength_bin_columns(pp=pp, source_data=events)
        print(
            f"Added wavelength bins to dataset {event_meta.id}, now has {events.shape[0]} events"
        )
        plot_spectrum_vs_data(
            meta_A=event_meta,
            source_A=events,
            filename=f"plots/spectra/{event_meta.id}.png",
            show=False,
        )

        spectrum, r2 = compute_spectrum_params(event_meta, events)
        flux_per_sec_12 = exp_count_per_sec(
            1.0,
            spectrum.A,
            spectrum.lambda_0,
            spectrum.sigma,
            spectrum.C,
        )
        print(
            f"Flux here should be similar to the source spectrum which is about 18, is {flux_per_sec_12}"
        )
        print(
            f"Apparent spectrum of dataset {event_meta.id}: A={spectrum.A} lambda_0={spectrum.lambda_0} sigma={spectrum.sigma} C={spectrum.C} with R2={r2}"
        )
        event_meta.apparent_spectrum = spectrum
        save_fits_metadata(event_meta)

        return event_meta, events
    if task == "wrange":
        assert event_meta.id == new_meta_id, "Meta id must match new meta id"
        assert events is not None, "Source data must be provided to select wbinrange."
        if "Wavelength (nm)" not in events.columns:
            events = convert_pi_channel_to_wavelength(events, pp)

        wave_edges, wave_centers, wave_widths = get_wavelength_bins(pp)

        events_mask = events["Wavelength (nm)"].between(wave_edges[0], wave_edges[-1])
        events = events[events_mask].copy()
        print(f"Cut dataset {event_meta.id} to {events.shape[0]} events")
        event_meta.source_count = events.shape[0]
        event_meta = fits_save_event_file(events, event_meta)
        return event_meta, events
    if task == "flatten":
        assert (
            events is not None
        ), "Source data must be provided to flatten. Perhaps you need to generate first?"
        assert (
            event_meta.id == new_meta_id
        ), "Meta id must match new meta id, try adding copy or generate"
        if "Wavelength (nm)" not in events.columns:
            events = convert_pi_channel_to_wavelength(events, pp)
        if "Wavelength Bin" not in events.columns:
            events = add_wavelength_bin_columns(pp=pp, source_data=events)

        events, min_count = downsample_flatten(data=events, pp=pp)
        assert all(events["Wavelength Bin"].value_counts() == min_count)
        print(f"Downsampled dataset to {min_count} counts per wavelength bin")

        event_meta.source_count = events.shape[0]
        event_meta = fits_save_event_file(events, event_meta)

        return event_meta, events
    if task == "downsample":
        assert (
            events is not None
        ), "Source data must be provided to downsample. Perhaps you need to generate first?"
        events = events.sample(pp.downsample_target_count)
        print(f"Downsampled dataset to {events.shape[0]} counts")
        event_meta.source_count = events.shape[0]
        event_meta = fits_save_event_file(events, event_meta)
        return event_meta, events
    if task == "equalize":
        assert (
            events is not None
        ), "Source data must be provided to equalize. Perhaps you need to generate first?"
        events = equalize_wavelength_bins(events)
        event_meta.source_count = events.shape[0]
        event_meta = fits_save_event_file(events, event_meta)
        return event_meta, events
    elif task in ["poissonize", "poissonize100", "poissonize80", "poissonize50"]:
        assert (
            events is not None
        ), "Source data must be provided to poissonize. Perhaps you need to generate first?"
        print(f"Homogeneous Poissonization of {event_meta.id} using {task}")
        if task == "poissonize50":
            poissoniation_percent = 100.0
        elif task == "poissonize80":
            poissoniation_percent = 80.0
        elif task == "poissonize20":
            poissoniation_percent = 20.0
        else:
            poissoniation_percent = 100
        events, event_meta = poissonize_homogeneous(
            events, poissoniation_percent, time_column="time", meta=event_meta
        )
        event_meta.star = f"{event_meta.star} (Poissonized)"
        fits_save_event_file(events, event_meta)
        print("Done poissonizing")
        save_fits_metadata(meta=event_meta)
        return event_meta, events
    elif task == "copy":
        assert events is not None, "Source data must be provided to copy source"

        event_meta = FitsMetadata(
            id=f"{new_meta_id}",
            raw_event_file=f"fits/{new_meta_id}.fits",
            synthetic=event_meta.synthetic,
            source_pos_x=event_meta.source_pos_x,
            source_pos_y=event_meta.source_pos_y,
            max_energy=event_meta.max_energy,
            min_energy=event_meta.min_energy,
            source_count=event_meta.source_count,
            star=event_meta.star,
            t_min=event_meta.t_min,
            t_max=event_meta.t_max,
            gen_id=event_meta.gen_id,
            ascore=event_meta.ascore,
            apparent_spectrum=event_meta.apparent_spectrum,
        )
        fits_save_event_file(events, event_meta)
        save_fits_metadata(event_meta)
        return event_meta, events.copy()


def task_handler(pipe: ComparePipeline, pp: ProcessingParameters):
    """Handle the tasks in the pipeline sequentially"""
    if pipe.source is not None:
        source_meta = load_fits_metadata(pipe.source)
        source_data = fits_read(source_meta.raw_event_file)

    else:
        source_data = None
        source_meta = None

    genparam = load_gen_param(pipe.gen_id)

    assert not (
        source_data is None and genparam is None
    ), "Either source data or generation parameters must be provided"

    print("Applying filters to dataset B")
    event_data_A = source_data
    event_meta_A = source_meta
    if pipe.A_fits_id is None:
        pipe.A_fits_id = f"{pipe.id}_A"
    try:
        for task in pipe.A_tasks:
            if task == "prunecolumns":
                print(f"Task {event_data_A.columns}")
            event_meta_A, event_data_A = apply_task(
                task=task,
                events=event_data_A,
                event_meta=event_meta_A,
                pp=pp,
                genparam=genparam,
                new_meta_id=pipe.A_fits_id,
                handle="A",
            )
            print(f"After task : {task}: Size {event_data_A.shape[0]}")
            save_fits_metadata(event_meta_A)
    except Exception as e:
        print(f"Error while processing A tasks: {e}")
        raise e
    fits_save_event_file(event_data_A, event_meta_A)

    event_meta_B = None
    event_data_B = None
    try:
        print("Applying filters to dataset B")
        if pipe.B_fits_id is None:
            pipe.B_fits_id = f"{pipe.id}_B"
        for task in pipe.B_tasks:
            # If the special command copyA is given, use the data from A in place of source
            if task == "copyA":
                event_data_B = event_data_A
                event_meta_B = event_meta_B
                task = "copy"
            elif task == "generate":
                assert genparam is not None, "Generation parameters must be provided"
                event_data_B = None
                event_meta_B = None
            elif event_data_B is None:
                event_data_B = source_data
                event_meta_B = source_meta

            event_meta_B, event_data_B = apply_task(
                task=task,
                events=event_data_B,
                event_meta=event_meta_B,
                pp=pp,
                genparam=genparam,
                new_meta_id=pipe.B_fits_id,
                handle="B",
            )
            print(f"After task : {task}: Size {event_data_B.shape[0]}")
        save_fits_metadata(event_meta_B)
    except Exception as e:
        print(f"Error while processing B tasks: {e}")
        raise e

    fits_save_event_file(event_data_B, event_meta_B)
    # meta_A = fits_save_events_generated(event_data, genparam, use_this_id=pipe.A_fits_id)
    return event_meta_A, event_meta_B


def main(use_defaults=True, overload_defaults: bool = True):
    print("Create a poissonizied and store that as B in the pipeline")

    #  Ensure that the default pipeline is present
    # create_default_pipeline()

    # Parse command line arguments
    args = parse_arguments(use_defaults)
    if args.chunk == "force":
        print("Will force chunk, even if cache exists")
        args.chunk = True
    else:
        print("Will try to use cached chunks is present")
        args.chunk = False

    assert (
        args.pipe is not None
    ), "Pipeline name missing, cannot create pipeline if you won't say the name!"
    # Load the default or specified pipeline
    if args.create:
        print("Creating new pipeline from command line arguments")
        if args.source is not None:
            if args.gen_id is not None:
                raise ValueError(
                    "Cannot specify both a source dataset and instruct generation of the source, pick either source (-s) or gen_id (-g) arguments!"
                )
        pipe = ComparePipeline(
            args.pipe,
            args.source,
            args.a_id,
            args.b_id,
            args.pp_id,
            args.gen_id,
            args.A_tasks.split(",") if args.A_tasks is str else args.A_tasks,
            args.B_tasks.split(",") if args.B_tasks is str else args.B_tasks,
        )
        save_pipeline(pipe)
        print("Creating new pipeline : Details")
    elif args.multiply is not None:
        N = int(args.multiply)
        assert args.pipe != "all", "Cannot duplicate all pipes, select one"
        assert args.pipe is not None, "Pipe cannot be empty when calling multiply"
        print(f"Will create {N} copies of {args.pipe} ")
        pipe = load_pipeline(args.pipe)
        if args.gen_id is not None:
            genmeta = load_gen_param(args.gen_id)
        else:
            genmeta = load_gen_param(pipe.gen_id)

        if args.pp_id is not None:
            pp = load_processing_param(args.pp)
            assert pp is not None, "Processing params needs to be here"
        else:
            pp = load_processing_param(pipe.pp_id)
            assert pp is not None, "Processing params needs to be here"

        for i in range(N):
            pipe_id = f"{pipe.id}_{i}"
            genid = f"{pipe.gen_id}_{i}"

            if genmeta.empirical:
                luc = 1e32 * np.random.uniform(0.5, 1.5)
                r_e = 6232 * np.random.uniform(0.2, 2.0)
            else:
                luc = genmeta.lucretius
                r_e = np.random.uniform(0.2, 1.1)
            gen = GenerationParameters(
                id=genid,
                alpha=genmeta.alpha,
                empirical=genmeta.empirical,
                lucretius=luc,
                r_e=r_e,
                theta=np.random.uniform(0, np.pi),
                theta_change_per_sec=genmeta.theta_change_per_sec
                * (np.random.uniform(1, 20) / np.random.uniform(1, 20)),
                t_min=genmeta.t_min,
                t_max=genmeta.t_max,
                perp=(genmeta.perp * np.random.uniform(-1, 1) * genmeta.t_max)
                * genmeta.velocity,
                phase=(genmeta.phase * np.random.uniform(-1, 1) * genmeta.t_max)
                * genmeta.velocity,
                max_wavelength=genmeta.max_wavelength,
                min_wavelength=genmeta.min_wavelength,
                star=genmeta.star,
                raw_event_file=None,
                spectrum=genmeta.spectrum,
                velocity=genmeta.velocity,
            )

            save_gen_param(gen)

            nupipe = ComparePipeline(
                id=pipe_id,
                source=pipe.source,
                A_fits_id=f"{pipe_id}_A",
                B_fits_id=f"{pipe_id}_B",
                gen_id=genid,
                pp_id=pp.id,
                A_tasks=pipe.A_tasks,
                B_tasks=pipe.B_tasks,
            )
            save_pipeline(nupipe)

    elif args.pipe == "all":
        summary = run_all_pipelines()
        if "failed" in summary and len(summary.failed) > 0:
            print("FAILED PIPELINES")
            for sumar in summary.failed:
                print(f"Pipeline {sumar.pipeline_id} with error {sumar.error}")
    else:
        pipe = load_pipeline(args.pipe)
        logs = run_pipeline(pipe)
        # print(f"{pipe.id} : Results ")


if __name__ == "__main__":
    main(use_defaults=True, overload_defaults=False)
