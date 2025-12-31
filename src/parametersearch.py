import pandas as pd
import numpy as np
from common.helper import compare_dataframes, get_uneven_time_bin_widths
from event_processing.plotting import (
    plot_spectrum_vs_data,
    compute_time_variability_async,
)
from common.helper import compare_variability_profiles, get_duration
from common.fitsmetadata import GenerationParameters
from common.powerdensityspectrum import compute_spectrum_params, PowerDensitySpectrum
from common.fitsread import (
    load_fits_metadata,
    save_fits_metadata,
    load_processing_param,
    fits_save_events_with_pi_channel,
    read_event_data_crop_and_project_to_ccd,
    chandra_like_pi_mapping,
)
import random as rnd
import uuid
from common.metadatahandler import load_summaries, save_gen_param
from event_processing.plotting import plot_spectrum_vs_data
from common.fitsmetadata import FitsMetadata, Spectrum
from common.helper import randomly_sample_from
from event_processing.var_analysis_plots import (
    binning_process,
    experiment_exists,
)
from common.generate_data import generate_synthetic_telescope_data
import csv
import os
import sys
import itertools
import uuid
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import argparse


def save_summary_to_csv(summary_dict, csv_path):
    """
    Save a single summary dictionary to a CSV file.
    Appends to the file if it exists, or creates a new one with headers.

    Parameters
    ----------
    summary_dict : dict
        Dictionary of metrics and parameters to save
    csv_path : str
        Path to the output CSV file
    """
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=summary_dict.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(summary_dict)


def make_genparam_variant(
    modify_this_gen: GenerationParameters,
    param_update_dict,
) -> GenerationParameters:
    # Load as FitsMetadata instance

    modify_this_gen.id = str(uuid.uuid4())[:8]

    # Apply parameter updates using setattr
    for key, val in param_update_dict.items():
        if hasattr(modify_this_gen, key):
            setattr(modify_this_gen, key, val)
        else:
            print(f"⚠️ Unknown parameter '{key}' in base metadata")

    # Update output-specific fields

    modify_this_gen.star = "Synthetic CXB"
    modify_this_gen.raw_event_file = (
        f"synthetic_lightlane/gen_{modify_this_gen.id}.fits"
    )

    # Save to JSON (assumes your class has .to_dict())
    save_gen_param(modify_this_gen)

    return modify_this_gen


def refine_search(
    target_meta_id: str,
    processing_parameters_id: str,
    parameter_summaries_csv,
    N=10,
):
    # Load and process real Chandra dataset once
    chandra_meta = load_fits_metadata(target_meta_id)

    pp = load_processing_param(processing_parameters_id)
    duration = get_duration(chandra_meta, pp)
    success, fits_meta, pp, chandra_data = read_event_data_crop_and_project_to_ccd(
        chandra_meta.id, pp.id
    )
    if not success:
        return Exception("Could not prepare source data for refinement search")
    # chandra_reduced = randomly_sample_from(chandra_data, 100000)
    default_gen = GenerationParameters(
        id="template",
        alpha=1.0,
        lucretius=-1,
        velocity=1,
        theta_change_per_sec=0.0,
        r_e=1,
        theta=0.2,
        t_max=duration,
        perp=0.0,
        phase=0.0,
        max_wavelength=pp.max_wavelength,
        min_wavelength=pp.min_wavelength,
        raw_event_file=None,
        spectrum=chandra_meta.apparent_spectrum,
        star=chandra_meta.star,
    )
    for i in range(0, N):
        print("Running another")
        summaries = load_summaries(
            path=parameter_summaries_csv, target_meta_id=target_meta_id
        )
        if len(summaries == 0):
            raise Exception(
                f"Cannot run -refine mode until we have entries in file {parameter_summaries_csv} to learn from. Run parameter-search with the -search"
            )

        param_combo = learn_then_choose(target_meta_id=target_meta_id)

        # Prepare output
        results = []
        Path("meta_variants").mkdir(exist_ok=True)

        # Grid search across all parameter combinations
        genparam_id = str(uuid.uuid4())[:8]  # unique short id

        genmeta = make_genparam_variant(default_gen, param_combo)

        if experiment_exists(
            summaries,
            param_combo["r_e"],
            param_combo["theta"],
            param_combo["alpha"],
            param_combo["lucretius"],
        ):
            print("This parameter combination has already been tested.")
        else:
            print("This is a new parameter combination - go ahead!")

        # Find the best match

        print(f"[{i+1}/{N}] Running: {param_combo}")

        try:
            # Generate + process synthetic data
            print(f"Fits file {genmeta.raw_event_file}")
            events = generate_synthetic_telescope_data(genmeta)
            new_fits_meta = fits_save_events_with_pi_channel(
                events=events, genmeta=genmeta
            )
            (
                success,
                new_fits_meta,
                pp,
                synth_data,
            ) = read_event_data_crop_and_project_to_ccd(
                new_fits_meta.id, processing_param_id=processing_parameters_id
            )
            if not success:
                print(
                    f"Experiment failed when trying to crop and project to CCD, see log"
                )
                continue
            # synth_reduced = randomly_sample_from(synth_data, downsample_target)

            print(f"Counts Synth {len(synth_data)}, real {len(chandra_data)}")

            synth_binned, new_fits_meta = binning_process(
                source_data=synth_data, meta=new_fits_meta, pp=pp
            )

            real_binned, chandra_meta = binning_process(
                source_data=chandra_data, meta=chandra_meta, pp=pp
            )

            synth_var = compute_time_variability_async(
                binned_datasets=synth_binned,
                duration=duration,
                time_bin_chunk_length=pp.time_bin_chunk_length,
            )

            real_var = compute_time_variability_async(
                binned_datasets=real_binned,
                duration=duration,
                time_bin_chunk_length=pp.time_bin_chunk_length,
            )

            # Compare to real data
            summary = compare_variability_profiles(
                A_df=real_var,
                B_df=synth_var,
                A_meta=chandra_meta,
                B_meta=new_fits_meta,
                A_gen=None,
                B_gen=genmeta,
            )
            summary.update(param_combo)
            summary["meta_id"] = genparam_id
            results.append(summary)
            save_summary_to_csv(summary, parameter_summaries_csv)
        except Exception as e:
            print(f"⚠️  Failed for {param_combo}: {e}")
            raise Exception(e)
            continue


def sample_random_params(param_grid, n_samples):
    keys = list(param_grid.keys())
    return [
        dict(zip(keys, [rnd.choice(param_grid[k]) for k in keys]))
        for _ in range(n_samples)
    ]


def get_variant_gen_params(
    param_grid,
    target_meta_id: str,
    parameter_summaries_csv: str,
    default_gen: GenerationParameters,
    N: int,
):
    genmetas = []
    summaries = load_summaries(parameter_summaries_csv, target_meta_id=target_meta_id)

    random_params = sample_random_params(param_grid, 3 * N)
    i = 0
    for random_param in random_params:
        genmeta = make_genparam_variant(default_gen, random_param)

        if summaries is not None:
            if experiment_exists(
                summaries,
                random_param["r_e"],
                random_param["theta"],
                random_param["alpha"],
                random_param["lucretius"],
            ):
                print("This parameter combination has already been tested.")
            else:
                print("This is a new parameter combination - go ahead!")
                genmetas.append(genmeta)
                i += 1
                if i >= N:
                    break
            # Find the best match

        print(f"[{i+1}/{N}] Running: {random_param}")

    return genmetas


def parameter_search(
    target_meta_id: str,
    processing_parameters_id: str,
    parameter_summaries_csv,
    param_grid,
    N=10,
    random: bool = False,
):
    # Load and process real Chandra dataset once
    chandra_meta = load_fits_metadata(target_meta_id)

    pp = load_processing_param(processing_parameters_id)
    duration = get_duration(chandra_meta, pp)
    success, chandra_meta, pp, chandra_data = read_event_data_crop_and_project_to_ccd(
        fits_id=target_meta_id, processing_param_id="test_1"
    )
    # chandra_reduced = randomly_sample_from(chandra_data, 100000)

    default_gen = GenerationParameters(
        id="template",
        alpha=1.0,
        lucretius=-1,
        velocity=1.0,
        theta_change_per_sec=0.0,
        r_e=1,
        theta=0.2,
        t_max=duration,
        perp=0.0,
        phase=0.0,
        max_wavelength=pp.max_wavelength,
        min_wavelength=pp.min_wavelength,
        raw_event_file=None,
        star=chandra_meta.star,
        spectrum=chandra_meta.apparent_spectrum,
    )

    try:
        results = []
        # Generate + process synthetic data
        genmetas = get_variant_gen_params(
            param_grid=param_grid,
            target_meta_id=target_meta_id,
            parameter_summaries_csv=parameter_summaries_csv,
            default_gen=default_gen,
            N=N,
        )
        for genmeta in genmetas:
            synth_data = generate_synthetic_telescope_data(genmeta)

            new_fits_meta = fits_save_events_with_pi_channel(
                events=synth_data, genmeta=genmeta
            )
            (
                success,
                new_fits_meta,
                pp,
                synth_data,
            ) = read_event_data_crop_and_project_to_ccd(
                new_fits_meta.id, processing_param_id=processing_parameters_id
            )

            print(f"Counts Synth {len(synth_data)}, real {len(chandra_data)}")

            # downsample_target = len(synth_data)

            # chandra_reduced = randomly_sample_from(chandra_data, downsample_target)
            compare_dataframes(
                synth_data, chandra_data, "Synth", "Real", show_plot=False
            )

            chandra_binned, chandra_meta = binning_process(
                source_data=chandra_data, meta=chandra_meta, pp=pp
            )

            synth_binnned, new_fits_meta = binning_process(
                source_data=synth_data, meta=new_fits_meta, pp=pp
            )
            time_bin_widths = get_uneven_time_bin_widths(pp)
            if "Wavelength Bin" not in synth_binnned.columns:
                print("We found a problem ,Wavelength Bin was not in synth_binned")
                continue

            chandra_var = compute_time_variability_async(
                binned_datasets=chandra_binned,
                meta=chandra_meta,
                pp=pp,
                time_bin_widths=time_bin_widths,
            )

            synth_var = compute_time_variability_async(
                binned_datasets=synth_binnned,
                meta=new_fits_meta,
                pp=pp,
                time_bin_widths=time_bin_widths,
            )
            # var_analysis_plot(synth_var)
            # Compare to real data

            summary = compare_variability_profiles(
                A_df=chandra_var,
                B_df=synth_var,
                A_meta=chandra_meta,
                B_meta=new_fits_meta,
                A_gen=None,
                B_gen=genmeta,
            )

            spectrum_real, r_squared = compute_spectrum_params(
                meta=chandra_meta, pp=pp, source_data=chandra_binned
            )
            spectrum_synth, r_squared = compute_spectrum_params(
                meta=new_fits_meta, pp=pp, source_data=synth_binnned
            )

            plot_spectrum_vs_data(
                meta_A=chandra_meta,
                binned_data_A=chandra_binned,
                processing_params=pp,
                filename=f"plot_{genmeta.id}_PDS.png",
                meta_B=new_fits_meta,
                binned_data_B=synth_binnned,
            )

            results.append(summary)
            save_summary_to_csv(summary, parameter_summaries_csv)
    except Exception as e:
        print(f"⚠️  Failed for {e}")


def sample_params():
    """
    Randomly sample a valid parameter combination for the Lightlane simulation.

    Returns:
        r_e (float): emitter radius
        D (float): spacing scale (inversely proportional to grid density)
        theta (float): angle of motion (radians)
        velocity (float): speed across the sheet (units per second)
        lucretius : the slope of the grid_distance growth curve
    """

    # Emitter radius - typical range from ~1 to 4
    r_e = np.random.uniform(1.0, 4.0)

    # Theta - any angle on unit circle
    theta = np.random.uniform(0, 2 * np.pi)

    # Velocity - slow to moderate tracing
    alpha = np.random.uniform(0.1, 5.0)

    lucretius = np.random.uniform(-1, -2)

    return r_e, theta, alpha, lucretius


def learn_then_choose(target_meta_id: str):
    filename = "temp/parameter_search.csv"
    summaries = load_summaries(filename, target_meta_id=target_meta_id)
    if summaries.empty:
        raise Exception(f"File {filename} was empty, cannot continue")

    features = ["r_e", "theta", "alpha", "lucretius"]

    target = "Combined MSE"

    X = summaries[features]
    y = summaries[target]

    model = RandomForestRegressor()
    model.fit(X, y)

    # Predict over a grid or randomly sampled candidates
    candidates = [sample_params() for _ in range(100000)]
    X_candidates = pd.DataFrame(candidates, columns=features)
    preds = model.predict(X_candidates)

    best_idx = np.argmin(preds)
    return X_candidates.iloc[best_idx].to_dict()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Parameter search: Generate synthetic data variants."
    )

    parser.add_argument(
        "-mode",
        choices=["refine", "search"],
        help="Mode of search: '-refine' for refining parameters or '-combo' for exploring combinations.",
    )
    parser.add_argument(
        "-N",
        nargs="?",
        type=int,
        default=100,
        help="Optional: Number of iterations to run (default: 100)",
    )

    args = parser.parse_args()
    return args


def main():
    print("Parameter search: Generate synthetic data")

    args = parse_arguments()

    refine = args.mode == "-refine"

    N = args.N

    print(f"Running mode: {'Refine' if refine else 'Combo'}, Iterations: {N}")

    param_grid = {
        "r_e": np.linspace(0.1, 20, 100),
        "alpha": np.linspace(0.1, 10, 10),
        "theta": np.linspace(0, np.pi / 6, 10),
        "lucretius": [-1.0],
    }

    if refine:
        print("Generating synthetic Lightlane dataset using Random Forest  ")
        refine_search(
            target_meta_id="default",
            processing_parameters_id="test_1",
            parameter_summaries_csv="temp/parameter_search.csv",
            N=N,
        )
        print(
            f"  Finished running refine parameter search, produced {N} synthetic experiments"
        )
    else:
        parameter_search(
            target_meta_id="default",
            processing_parameters_id="test_1",
            parameter_summaries_csv="temp/parameter_search.csv",
            param_grid=param_grid,
            N=N,
        )
        print(
            f"  Finished param-combo parameter search, produced {N} synthetic experiments"
        )


if __name__ == "__main__":
    main()
