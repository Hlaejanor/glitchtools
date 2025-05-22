import numpy as np
from amplpy import AMPL, ampl_notebook
import matplotlib.pyplot as plt
# ampl = ampl_notebook(
#    modules=["highs", "cbc", "gurobi", "cplex"], # pick from over 20 modules including most commercial and open-source solvers
#    license_uuid="bc7d32e2-e20c-49f9-a56f-1c5f7b2f0336")
from common.fitsread import fits_read, load_fits_metadata, process_dataset
from common.metadatahandler import (
    load_gen_param,
    GenerationParameters,
    compare_generation_params,
)
from scipy.optimize import differential_evolution
from common.helper import randomly_sample_from, sample_evenly_from_wavelength_bins
from functools import partial

def evaluate_lightlane_score(params, df, g_bias_weight=1.0, use_median=True):
    """
    Score based on distance to nearest lightlane + bias toward larger g values.

    Parameters
    ----------
    params : list
        [x0, y0, theta, alpha, lucretius]
    df : pd.DataFrame
        Input photon dataset with columns 'relative_time', 'CCD X', 'CCD Y', 'Wavelength (nm)'
    g_bias_weight : float
        Controls how strongly to penalize small g values.
    use_median : bool
        Use median distance instead of mean (robust to outliers).

    Returns
    -------
    loss : float
        Score to be minimized.
    """
    times = df["relative_time"].values
    xs = df["CCD X"].values
    ys = df["CCD Y"].values
    lambdas = df["Wavelength (nm)"].values

    distances, g_vals = distance_to_nearest_lightlane(
        times=times,
        xs=xs,
        ys=ys,
        lambdas=lambdas,
        x0=params[0],
        y0=params[1],
        theta=params[2],
        alpha=params[3],
        lucretius=params[4],
    )
   
    # 1. Loss from distance
    if use_median:
        distance_score = np.median(distances)
    else:
        distance_score = np.mean(
            distances**2
        )  # squared for larger penalty on outliers

    # 2. Penalty for small g (mean over g_vals)
    g_penalty = np.mean(1.0 / (g_vals + 1e-6))

    # 3. Total loss: prioritize distance but penalize small g
    loss = distance_score + g_bias_weight * g_penalty

    return loss


def objective_with_data(params, data):
    # Compute distances to nearest lightlane
    distances = evaluate_lightlane_score(params, data)
    if distances is not None:
        var_score = np.median(distances)
        return var_score
    return 0


def distance_to_nearest_lightlane(
    times, xs, ys, lambdas, x0, y0, theta, alpha, lucretius
):
    """
    Compute Euclidean distance to nearest lightlane on a hexagonal grid.
    """
    try:
        g_vals = alpha * np.exp(lucretius * lambdas)
        v = np.array([np.cos(theta), np.sin(theta)])  # direction of motion
        v_perp = np.array([-np.sin(theta), np.cos(theta)])  # perpendicular

        # Estimate telescope position at time t
        positions = np.stack([xs, ys], axis=1) + np.outer(times, v)
        delta = positions - np.array([x0, y0])

        # Project onto (u, v) grid coordinates
        along = delta @ v
        perp = delta @ v_perp

        # Compute g for each wavelength
        g = alpha * lambdas * np.exp(lucretius)
        dy = (np.sqrt(3) / 2) * g

        # Convert to hex grid index space
        row = np.round(perp / dy).astype(int)
        is_odd_row = (row % 2) != 0
        col = np.round((along - is_odd_row * 0.5 * g) / g).astype(int)

        # Reconstruct nearest lightlane center
        lane_x = col * g + is_odd_row * 0.5 * g
        lane_y = row * dy

        lane_pos = np.outer(lane_x, v) + np.outer(lane_y, v_perp)
        camera_pos = positions

        # Compute Euclidean distance to closest lightlane center
        dists = np.linalg.norm(camera_pos - lane_pos, axis=1)

        return np.sum(dists), g_vals
    except Exception as ex:
        print("EXCEPTION WITH PARAMS ", ex)


if __name__ == "__main__":
    N = 10000
    max_lamb = 0.8
    min_lamb = 0.3
    meta = load_fits_metadata("1a8c5343")
    used_gen = load_gen_param(meta.gen_id)

    fits = fits_read(meta.raw_event_file)

    data = process_dataset(fits, 1024, max_lamb, min_lamb, False, None)
    data = sample_evenly_from_wavelength_bins(data, N, 100)
    assert data is not None, "Data cannot be none"
    used_gen.max_wavelength = max_lamb
    used_gen.min_wavelength = min_lamb

    # Sample parameters (to be optimized)
    x0, y0 = 0.0, 0.0
    theta = np.pi / 4
    alpha = 0.1
    lucretius = -1.0
    if False:
        alphas = np.linspace(0.01, 5.0, 200)
        losses = [evaluate_lightlane_score([x0, y0, theta, a, -1], data) for a in alphas]

        plt.plot(alphas, losses)
        plt.xlabel("Alpha")
        plt.ylabel("Loss")
        plt.title("Loss Landscape for Alpha with True Parameters Else Fixed")
        plt.show()
        # Parameters: x0, y0, theta, alpha, lucretius
    bounds = [
        (0, 1),  # x0, mod g
        (0, 1),  # y0, mod g
        (0, np.pi),  # theta
        (0.01, 10.0),  # alpha
        (-1, -1),  # lucretius
    ]
    objective = partial(objective_with_data, data=data)
    result = differential_evolution(
        func=objective,
        bounds=bounds,
        strategy="best1bin",
        maxiter=100,
        popsize=15,
        tol=1e-5,
        mutation=(0.5, 1),
        recombination=0.7,
        polish=True,
        disp=True,
        workers=-1,  # multiprocessing OK now
    )
    best_params = result.x  # This is a 5-element array

    recovered_gen = GenerationParameters(
        id=f"{meta.id}_rec",
        alpha=best_params[3],
        lucretius=best_params[4],
        r_e=0.8,  # fixed
        theta=best_params[2],
        t_max=meta.t_max,
        phase=best_params[1],
        perp=best_params[0],
        max_wavelength=max_lamb,
        min_wavelength=min_lamb,
        star=f"{used_gen.star}_recovered",
    )
    compare_generation_params(used_gen, recovered_gen, "Actual", "Recovered")

    print(f"Result : ", result)


