import numpy as np
from pandas import DataFrame
from common.lanesheetMetadata import (
    LanesheetMetadata,
    load_lanesheet_metadata,
    save_lanesheet_metadata,
)
from common.fourier import (
    reconstruct_offset_from_phase_perp,
    generate_freq_signature,
    save_to_csv,
)
from pandas import DataFrame
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from common.helper import filter_by_wavelength_bin, get_wavelength_bins
from common.fitsmetadata import FitsMetadata, GenerationParameters, ProcessingParameters
from common.powerdensityspectrum import exp_count_per_sec
from common.fitsread import fits_read, read_crop_and_project_to_ccd
from common.metadatahandler import (
    load_pipeline,
    load_processing_param,
    load_gen_param,
    load_fits_metadata,
    save_fits_metadata,
)
from common.generate_data import (
    generate_dataset_for_experiment,
)
from lanesheetestimator.lanesheet_fitter import (
    refine_grid_spacing_2,
    refine_angle,
    tighten_offset_voronoi,
    refine_re,
)
import argparse
from lanesheetestimator.learnlanesheet import LanesheetParamEstimator
from parametersearch import get_variant_gen_params


def parse_arguments(use_defaults: bool):
    default_pipeline = "fullfat"
    parser = argparse.ArgumentParser(
        description="Create mimic dataset: Generate synthetic data variants."
    )
    if use_defaults:
        parser.add_argument(
            "-p",
            "--pipeline",
            required=not use_defaults,
            default=default_pipeline if use_defaults else None,
            type=str,
            help="Set the pipeline you want to run",
        )

    args = parser.parse_args()

    return args


def g_model(lamb, alpha, lucretius):
    return alpha * np.exp(lucretius * lamb)


# Evaluate all matches and global fit
def evaluate_lanesheet_predictions(true_lanesheets, est_lanesheets):
    # Create lookup dictionaries
    true_dict = {ls.lambda_center: ls for ls in true_lanesheets}
    est_dict = {ls.lambda_center: ls for ls in est_lanesheets}

    # Compare each matching pair
    comparisons = []
    for lamb in sorted(true_dict.keys()):
        if lamb in est_dict:
            comparisons.append(compare_lanesheets(true_dict[lamb], est_dict[lamb]))

    df = DataFrame(comparisons)

    # Fit alpha and lucretius to estimated g values
    lambda_vals = df["lambda_center"].values
    g_est_vals = df["g_est"].values
    popt, _ = curve_fit(g_model, lambda_vals, g_est_vals, p0=[1.0, -1.0])
    alpha_fit, lucretius_fit = popt

    # Optionally compute error to one of the true sets (take the first one)
    alpha_true = true_lanesheets[0].alpha
    lucretius_true = true_lanesheets[0].lucretius
    alpha_error = abs(alpha_fit - alpha_true)
    lucretius_error = abs(lucretius_fit - lucretius_true)

    # Add fit results
    fit_result = {
        "alpha_fit": alpha_fit,
        "lucretius_fit": lucretius_fit,
        "alpha_error": alpha_error,
        "lucretius_error": lucretius_error,
        "mean_g_error": df["g_error"].mean(),
        "std_g_error": df["g_error"].std(),
    }

    return df, fit_result


# Compare a true and estimated lanesheet
def compare_lanesheets(true_ls, est_ls):
    g_true = true_ls.alpha * np.exp(true_ls.lucretius * true_ls.lambda_center)
    g_est = est_ls.alpha * np.exp(est_ls.lucretius * est_ls.lambda_center)
    return {
        "lambda_center": true_ls.lambda_center,
        "r_e_error": abs(true_ls.r_e - est_ls.r_e),
        "theta_error": abs(true_ls.theta - est_ls.theta),
        "perp_error": abs(true_ls.perp - est_ls.perp),
        "phase_error": abs(true_ls.phase - est_ls.phase),
        "g_true": g_true,
        "g_est": g_est,
        "g_error": abs(g_true - g_est),
    }


def build_training_data(
    meta: FitsMetadata,
    pp: ProcessingParameters,
    genparam: GenerationParameters,
    plot: bool = False,
) -> list[LanesheetMetadata]:
    # Set true parameters and simulate

    wave_edges, wave_centers, wave_widths = get_wavelength_bins(pp)
    true_lanesheets = []

    for i in range(pp.wavelength_bins):
        exp_flux_per_sec = exp_count_per_sec(
            wave_centers[i],
            genparam.spectrum.A,
            genparam.spectrum.lambda_0,
            genparam.spectrum.sigma,
            genparam.spectrum.C,
        )

        true_lanesheet = LanesheetMetadata(
            id=meta.id,
            truth=True,
            lambda_center=wave_centers[i],
            lambda_width=wave_widths[i],
            exp_flux_per_sec=exp_flux_per_sec,
            r_e=genparam.r_e,
            theta=genparam.theta,
            perp=genparam.perp,
            phase=genparam.phase,
            alpha=genparam.alpha,
            lucretius=genparam.lucretius,
        )
        true_lanesheets.append(true_lanesheet)

        points_ll = generate_dataset_for_experiment(true_lanesheet)

        signature = generate_freq_signature(
            meta_id=meta.id, points_ll=points_ll, plot=plot, true_vals=true_lanesheet
        )
        save_to_csv(
            true_ls=true_lanesheet,
            signature=signature,
        )
    return true_lanesheets


def estimate_lanestack(
    meta: FitsMetadata, pp: ProcessingParameters, genparam: GenerationParameters
) -> list[LanesheetMetadata]:
    # Load the photon events (real or simulated)
    events = fits_read(meta.raw_event_file)

    # Crop and process the data to in standard way
    success, meta, pp, events = read_crop_and_project_to_ccd(meta.id, pp.id)
    if not success:
        raise Exception("Processing the data failed")

    meta.source_count = events.shape[0]  # Update the source count
    save_fits_metadata(meta)  # Commit to disk

    # Load the training data for the estimator
    estimator = LanesheetParamEstimator("beat/beat_metadata.csv")
    estimator.train_model_base(diagnostics=True)
    estimator.train_model_full(diagnostics=True)
    # For each wavelength bin
    wave_edges, wave_centers, wave_widths = get_wavelength_bins(pp)
    lanesheets = []
    for i in range(len(wave_centers)):
        exp_flux_per_sec = exp_count_per_sec(
            wave_centers[i],
            meta.apparent_spectrum.A,
            meta.apparent_spectrum.lambda_0,
            meta.apparent_spectrum.sigma,
            meta.apparent_spectrum.C,
        )
        # Filter for this wavelength only
        events_for_lambda_bin = filter_by_wavelength_bin(
            events, wave_centers[i], wave_widths[i]
        )
        # Make into a numpy array
        events_for_lambda_bin = events_for_lambda_bin.to_numpy()
        signature = generate_freq_signature(
            meta_id=meta.id, points_ll=events_for_lambda_bin, true_vals=None, plot=False
        )
        # Predict the Lanesheet values for this dataset using the ML method
        est_vals = estimator.predict(signature)
        assert est_vals is not None, "Estimator returned empty prediction"
        # Generate a lanesheet for this
        lanesheet = LanesheetMetadata(
            id=meta.id,
            truth=False,
            lambda_center=wave_centers[i],
            lambda_width=wave_widths[i],
            exp_flux_per_sec=exp_flux_per_sec,
            alpha=est_vals["alpha"],
            lucretius=est_vals["lucretius"],
            r_e=est_vals["r_e"],
            theta=est_vals["theta"],
            perp=est_vals["perp"],
            phase=est_vals["phase"],
        )
        lanesheets.append(lanesheet)

    return lanesheets


def ruiuasd():
    fine_tune = False
    runs = 1000

    results = []
    """
    # true_meta.id = f"ll_{i}"

    # true_meta.g = np.random.uniform(1, 8)
    # true_meta.theta = np.random.uniform(0.1, (np.pi / 2) * 0.97)
    # true_meta.offset = np.random.uniform(-true_meta.g / 2, true_meta.g / 2, 2)
    # true_meta.r_e = np.random.uniform(0.5, true_meta.g * 2)
    # true_meta.r_e = 0.8

    if not fine_tune:
        print("Fast building training data")
        continue
    if estimator.trained:
        print("1. Estimate values from beat signature")
        est_vals = estimator.predict(signature)
        errors = estimator.score(true_vals, est_vals)
        print(
            f"  -- Guessed values : Theta: {est_vals['theta']}, r_e:{est_vals['r_e']},      g : {est_vals['g']}, Phase: {est_vals['phase']} Perp: {est_vals['perp']} "
        )
        print(
            f"  -- Errors : Theta: {errors['theta']},   r_e:{errors['r_e']},g : {errors['g']}, Phase: {errors['phase']} Perp: {errors['perp']} "
        )

        offset_estimate = reconstruct_offset_from_phase_perp(est_vals)
        print(
            f"2. Refining angle estimate : We have {est_vals['theta']} and want to seek tou {true_meta.theta}"
        )
        best_theta, best_theta_score, best_theta_tol = refine_angle(
            points_ll,
            true_meta.offset,
            est_vals["theta"],
            true_meta.dt,
            est_vals["g"],
            est_vals["r_e"],
        )
        print(f"  -- Found best theta {best_theta}")
        ll_e.theta = best_theta
        ll_e.theta_tolerance = best_theta_tol
        if best_theta_tol > estimator.theta_rmse:
            raise Warning(
                f"Theta tolerance {best_theta_tol} higher than RMS of estimator {estimator.theta_rmse}"
            )

        print(
            f"3. Refine grid estimate : Starting at {est_vals['g']}, and seeking {true_meta.g}"
        )
        best_g, best_g_score, best_g_tol = refine_grid_spacing_2(
            points_ll,
            true_meta.offset,
            best_theta,
            true_meta.dt,
            est_vals["g"],
            true_meta.r_e,
        )
        print(f"  -- Found best g {best_g}")
        ll_e.g = best_g
        ll_e.g_tolerance = best_g_tol

        if best_g_tol > estimator.g_rmse:
            raise Warning(
                f"G tolerance {best_g_tol} higher than RMS of estimator {estimator.g_rmse}"
            )

        print(
            f"4. Refine r_e estimate : Starting at {est_vals['r_e']}, and seeking {true_meta.r_e}"
        )
        best_r_e, best_r_e_score = refine_re(
            hits=points_ll,
            g_guess=true_meta.g,
            theta_guess=true_meta.theta,
            offset_guess=true_meta.offset,
            dt=true_meta.dt,
            r_e=true_meta.r_e,
        )
        ll_e.r_e = best_r_e

        print(
            f"4. Refine offset estime : Starting at {offset_estimate}, and seeking {true_meta.offset}"
        )
        best_offset, best_offset_tol = tighten_offset_voronoi(
            points_ll,
            offset_estimate,
            best_g,
            best_theta,
            true_meta.dt,
            est_vals["r_e"],
        )

        print(
            f"  -- Found offset {best_offset}, true offset {true_meta.offset}. Diff {(true_meta.offset - best_offset)}"
        )
        ll_e.offset = best_offset
        ll_e.offset_tolerance = best_offset_tol

        if (
            best_offset_tol > estimator.phase_rmse
            or best_offset_tol > estimator.perp_rmse
        ):
            raise Warning(
                f"G tolerance {best_g_tol} higher than RMS of estimator {estimator.g_rmse}"
            )

"""


if __name__ == "__main__":
    print("Starting Lanesheet fitter")
    parser = argparse.ArgumentParser(description="Generate Lightlane video.")
    parser.add_argument(
        "-p",
        "--pipeline",
        required=False,
        default="default",
        type=str,
        help="Set the pipeline you want to run",
    )
    parser.add_argument(
        "-n",
        "--N",
        required=False,
        default=1,
        type=int,
        help="Set the n amount of training",
    )

    args = parser.parse_args()

    assert args.pipeline is not None, "Must specify a pipeline index"

    pipeline_meta = load_pipeline(args.pipeline)
    assert (
        pipeline_meta is not None
    ), "Pipeline meta missing. Try to run create_default_pipeline.sh"
    # Load metadata for target set A

    meta = load_fits_metadata(pipeline_meta.A_fits_id)
    pp = load_processing_param(pipeline_meta.pp_id)
    # Load the generation parameter from the pipeline
    beat_training_data = "beat/beat_metadata.csv"

    default_gen = GenerationParameters(
        id="template",
        alpha=1.0,
        lucretius=-1,
        r_e=1,
        theta=0.2,
        t_max=1000,
        perp=0.0,
        phase=0.0,
        max_wavelength=pp.max_wavelength,
        min_wavelength=pp.min_wavelength,
        raw_event_file=None,
        star="Fitting",
        spectrum=meta.apparent_spectrum,
    )

    param_grid = {
        "r_e": np.linspace(0.1, 20, 100),
        "alpha": np.linspace(0.1, 10, 10),
        "theta": np.linspace(0, np.pi / 6, 10),
        "lucretius": np.linspace(-2, -0.5, 10),
    }

    genmetas = get_variant_gen_params(
        param_grid=param_grid,
        target_meta_id=None,
        parameter_summaries_csv=beat_training_data,
        default_gen=default_gen,
        N=args.N,
    )

    for genmeta in genmetas:
        true_lanesheets = build_training_data(meta, pp, genmeta)

    # The purpos of the lanestack is to estimate a value for g, theta and r_e, perp and phase that matches
    # the observed data. When the lanestack has been compiled, we can try to fit the
    estimated_lanesheets = estimate_lanestack(meta, pp, genmeta)

    # If the lanestack is based on a generated dataaset, then we may compare the values
    df, results = evaluate_lanesheet_predictions(true_lanesheets, estimated_lanesheets)
    print(df)
    print(results)
