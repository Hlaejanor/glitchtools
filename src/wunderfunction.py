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
from common.helper import (
    randomly_sample_from,
    sample_evenly_from_wavelength_bins,
    split_into_time_bins,
)
from functools import partial
import os
import csv


def save_to_csv(row, csv_path: str):
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                ["phase", "perp", "theta", "alpha", "lucretirus", "r_e", "losses"]
            )  # header
        writer.writerow(row)


def evaluate_lightlane_score(
    params, data_array, i: int, g_bias_weight=1.0, use_median=True
):
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

    times = data_array[:, 0]
    xs = data_array[:, 1]
    ys = data_array[:, 2]
    lambdas = data_array[:, 3]

    mindist_array, lanecount, g_vals = count_lightlanes_intersecting_camera_vectorized(
        times=times,
        particle_lambdas=lambdas,
        phase=params[0],
        perp=params[1],
        theta=params[2],
        alpha=params[3],
        lucretius=params[4],
        r_e=params[5],
    )
    # The loss is highest when lanecount = 0, because then we are sampling impossible events
    # Then the loss is the lowest when lanecount is 1, because then the minimal explanation
    # Then the loss is smaller as we increase the lane denstiy
    median_g = np.median(g_vals)
    penalty = 0.2 / median_g
    mindist_term = mindist_array
    clutter_term = 0.3 * lanecount**2 * mindist_array

    loss_per_sample = mindist_term + clutter_term
    median_loss = np.median(loss_per_sample)

    # Final loss includes the penalty for small g
    
    return median_loss + penalty


def objective_with_data(params, data, i):
    # This returns the loss for each parameter combination
    #print(f"Evaluating time bin {i}")
    loss = evaluate_lightlane_score(params, data, i)

    return loss


def count_lightlanes_intersecting_camera_vectorized(
    times, phase, perp, theta, alpha, lucretius, particle_lambdas, r_e
):
    """
    Vectorized version to compute how many lightlanes intersect the camera center at each time t.
    """

    # Camera velocity vector
    v = np.array([np.cos(theta), np.sin(theta)])

    # Camera positions at each time
    positions = np.stack([phase + times * v[0], perp + times * v[1]], axis=1)  # shape (N, 2)

    # Grid spacing g at each time
    g_vals = alpha / (particle_lambdas ** lucretius)

    # Preallocate result
    lanecounts = np.zeros_like(times, dtype=int)
    mindist_array = np.zeros(len(times))

    for i in range(len(times)):
        pos = np.array(positions[i], dtype=np.float64)
        g = g_vals[i]
        
        # Determine the number of rows/cols to check
        M = int(np.ceil((2 * r_e) / g))
        if M > 10:
            est_lanecount = (np.pi * r_e**2 * g * M**2 * 2) / ((M * g) ** 2)
            # print(f"Size of lanebox {M}, L: {est_lanecount}")
            lanecounts[i] = est_lanecount

            mindist_array[i] = g / 2
        else:
            m_vals = np.arange(-M, M + 1)
            n_vals = np.arange(-M, M + 1)
            # Generate meshgrid for base grid
            m_grid, n_grid = np.meshgrid(m_vals, n_vals, indexing="ij")
            m_flat = m_grid.ravel()
            n_flat = n_grid.ravel()

            # Base grid centers
            base_x = m_flat * g
            base_y = n_flat * g
            base_centers = np.stack([base_x, base_y], axis=1)

            # Offset grid centers
            offset_x = base_x + 0.5 * g
            offset_y = base_y + (np.sqrt(3) / 2) * g
            offset_centers = np.stack([offset_x, offset_y], axis=1)

            # Compute distances and count
            vec_base = -base_centers + pos
            vec_offset = -offset_centers + pos
            dist_base = np.linalg.norm(vec_base, axis=1)
            dist_offset = np.linalg.norm(vec_offset, axis=1)
            dists_combined = np.stack([dist_base, dist_offset], axis=1)

            # Take the minimum along the second axis
            mindist = np.min(np.linalg.norm(dists_combined))
            count = np.count_nonzero(dist_base < r_e) + np.count_nonzero(
                dist_offset < r_e
            )
            mindist_array[i] = mindist
            lanecounts[i] = count

    # Return the median minimum distance for this parameter set over these time bins
    return mindist_array, lanecounts, g_vals


def count_lightlanes_intersecting_camera(
    times, x0, y0, theta, alpha, lucretius, particle_lambdas, r_e
):
    """
    Compute how many lightlanes intersect the camera center at each time t,
    based on a hexagonal grid modeled as two interleaved square grids.

    Parameters:
        times      : tibme bins shape (N,)
        x0, y0     : initial camera position
        theta      : angle of velocity vector
        alpha      : scalar parameter (for g = alpha / (lambda * lucretius))
        lucretius  : scalar parameter (for g = alpha / (lambda * lucretius))
        lambdas    : np.ndarray of shape (N,)
        r_e        : lightlane radius

    Returns:
        counts     : np.ndarray of shape (N,) with number of overlapping lightlanes
        g_vals     : np.ndarray of shape (N,) with grid spacing values
    """

    v = np.array([np.cos(theta), np.sin(theta)])  # camera motion direction
    positions = np.stack([x0 + times * v[0], y0 + times * v[1]], axis=1)  # shape (N, 2)

    g_vals = alpha / (particle_lambdas * lucretius)
    counts = np.zeros_like(times, dtype=int)

    for i in range(len(times)):
        pos = positions[i]
        g = g_vals[i]

        # Compute sampling range in each direction
        M = int(np.ceil((2 * r_e) / g))

        # Generate base grid and offset grid
        ms = np.arange(-M, M + 1)
        ns = np.arange(-M, M + 1)

        # First grid: (m g, n g)
        for m in ms:
            for n in ns:
                base_center = np.array([m * g, n * g])
                offset_center = base_center + np.array([0.5 * g, np.sqrt(3) / 2 * g])

                # Count base
                if np.linalg.norm(pos - base_center) < r_e:
                    counts[i] += 1

                # Count offset
                if np.linalg.norm(pos - offset_center) < r_e:
                    counts[i] += 1

    return counts, g_vals


if __name__ == "__main__":
    N = 312000
    max_lamb = 0.8
    min_lamb = 0.3
    meta = load_fits_metadata("default")
    used_gen = load_gen_param(meta.gen_id)

    fits = fits_read(meta.raw_event_file)

    data = process_dataset(fits, 1024, max_lamb, min_lamb, False, None)
    data = sample_evenly_from_wavelength_bins(data, N, 100)

    time_bin_indices = split_into_time_bins(data, 100)

    assert data is not None, "Data cannot be none"
    if used_gen is not None:
        used_gen.max_wavelength = max_lamb
        used_gen.min_wavelength = min_lamb

    # Sample parameters (to be optimized)
    x0, y0 = 0.0, 0.0
    theta = np.pi / 4
    alpha = 0.1
    lucretius = -1.0
    if False:
        alphas = np.linspace(0.01, 5.0, 200)
        losses = [
            evaluate_lightlane_score([x0, y0, theta, a, -1], data) for a in alphas
        ]

        plt.plot(alphas, losses)
        plt.xlabel("Alpha")
        plt.ylabel("Loss")
        plt.title("Loss Landscape for Alpha with True Parameters Else Fixed")
        plt.show()
        # Parameters: x0, y0, theta, alpha, lucretius
    bounds = [
        (0, 100),  # phase, mod g
        (0, 100),  # perp, mod g
        (0, np.pi/6),  # theta
        (0.01, 1000.0),  # alpha
        (0.1, 2),  # lucretius
        (0, 3),  # r_e
    ]
    if used_gen is not None:
        base_params = [
            used_gen.phase,
            used_gen.perp,
            used_gen.theta,
            used_gen.alpha,
            used_gen.lucretius,
            used_gen.r_e,
        ]

        print(f"Debug {0}")
        mask = time_bin_indices[0]
        bind = data.iloc[mask].to_numpy()
        losses = evaluate_lightlane_score(base_params, bind, i=0)

    best_params = []

    for i in range(0, len(time_bin_indices)):
        print(f"Iteration {i}")
        mask = time_bin_indices[i]
        bind = data.iloc[mask].to_numpy()
        # distances = evaluate_lightlane_score(base_params, bind)
        print(f"Time bin {time_bin_indices[i]}")
        objective = partial(objective_with_data, data=bind, i=i)
        result = differential_evolution(
            func=objective,
            bounds=bounds,
            strategy="best1bin",
            maxiter=1,
            popsize=15,
            tol=1e-1,
            mutation=(0.5, 1),
            recombination=0.7,
            polish=True,
            disp=True,
            workers=-1,  # multiprocessing OK now
        )
        losses = evaluate_lightlane_score(result.x, bind, i=0)
        row_with_loss = np.append(result.x, losses)
        best_params.append(row_with_loss)
        save_to_csv(row_with_loss,  "wunder.csv")

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

    best_params = np.array(best_params)
    recovered_gen = np.median(best_params, axis=1)

    compare_generation_params(used_gen, recovered_gen, "Actual", "Recovered")

    print(f"Result : ", result)
