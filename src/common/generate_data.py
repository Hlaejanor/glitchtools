import numpy as np
import matplotlib.pyplot as plt
from common.powerdensityspectrum import PowerDensitySpectrum, exp_count_per_sec
from common.fitsmetadata import FitsMetadata, GenerationParameters, ProcessingParameters
from common.fitsread import (
    fits_file_exists,
    fits_read,
    fits_save_events_with_pi_channel,
    pi_channel_to_wavelength_and_width,
)
from scipy.spatial import cKDTree
from common.lanesheetMetadata import LanesheetMetadata
import imageio


from common.helper import estimate_lanecount_fast, estimate_lanecount, get_duration


def generate_synthetic_telescope_data(
    genmeta: GenerationParameters, pp: ProcessingParameters
):
    events = []
    assert genmeta.t_max is not None, "t_max must be set and > 0"
    assert genmeta.spectrum is not None, "Spectrum must be set to generate data"
    assert (
        genmeta.spectrum.A > 0
    ), "Spectrum cannot be flat to generate data. A was {genmeta.spectrum.A}"
    assert (
        genmeta.spectrum.lambda_0 > 0
    ), "Spectrum cannot be flat to generate data. lambda_0 was {genmeta.spectrum.A}"

    t = 0.0  # TODO - have t0 as a parameter
    t = 0
    # The number of steps required to use 100 second generation chunks
    steps = get_duration(genmeta) / 100
    # Since steps need to be integer, compute the time step so that it exactly matches the length of the dataset
    dt = get_duration(genmeta) / np.floor(steps)
    # Generate a synthetic event file using genparam, which include the spectrum and duration

    sheet_size = (
        np.max(np.array([np.cos(genmeta.theta), np.sin(genmeta.theta)])) * genmeta.t_max
    )
    velvec = np.array([np.cos(genmeta.theta), np.sin(genmeta.theta)])

    lambda_nm = np.flip(
        np.linspace(genmeta.max_wavelength, genmeta.min_wavelength, pp.wavelength_bins)
    )
    lambda_bin_width = np.ones(pp.wavelength_bins) * np.abs(
        (lambda_nm[0] - lambda_nm[1])
    )

    lambdas = np.stack((lambda_nm, lambda_bin_width), axis=1)
    l_idx = 0
    total_added = 0
    last_bin = lambdas[-1][0]
    try:
        for lamb, lambda_bin_width in lambdas:
            t = 0.0
            flux_per_sec = exp_count_per_sec(
                lamb,
                genmeta.spectrum.A,
                genmeta.spectrum.lambda_0,
                genmeta.spectrum.sigma,
                genmeta.spectrum.C,
            )
            assert (
                lambda_bin_width > 0
            ), f"Lambda bin must be > 0, was {lambda_bin_width}"
            print(f"Flux per {lamb}:  {flux_per_sec} per sec from empirical lambda ")

            ls = LanesheetMetadata(
                id=f"{genmeta.id}_bin_{l_idx}",
                truth=False,
                empirical=genmeta.empirical,
                lambda_center=lamb,
                lambda_width=lambda_bin_width,
                exp_flux_per_sec=flux_per_sec,
                alpha=genmeta.alpha,
                # theta=genmeta.theta,
                r_e=genmeta.r_e,
                perp=genmeta.perp,
                phase=genmeta.phase,
                lucretius=genmeta.lucretius,
                lucretius_tolerance=None,
                alpha_tolerance=None,
                perp_tolerance=None,
                phase_tolerance=None,
                theta_tolerance=None,
            )
            # A dioburst will appear when many lightlanes become commensurate
            # In the generator, this nominally occurs when phase = 0.0, because that corresponds to when the lanesheet is initialized. As time progresses,
            # The lanes drift out of sync, and eventually into sync again. When the lanes are fully synced, Diophantine Bursts (diobursts) are created.
            # To avoid the simulation always starting in the middle of a burst, we can vary the phase and perp, indicating the number of seconds before and a diobursts
            # Var they velocity parameter to speed up how quickly the generator moves between bursts
            perp_dir = np.array([-velvec[1], velvec[0]])

            pos = perp_dir * (genmeta.perp * genmeta.alpha)
            pos += velvec * (genmeta.phase * genmeta.alpha)
            # Use the same x-ray distribution as the chandra dataset, to obtain an estimate for the number of photons to expect on average for this time bin length
            theta = genmeta.theta_change_per_sec * t + genmeta.theta
            velvec = np.array([np.cos(theta), np.sin(theta)]) * genmeta.velocity
            if ls.empirical:
                print(
                    f"Using empirical lanecount estimator. Note that this requires that the genmeta paramters alpha and r_e  are given in light-seconds, and velocity as share of c."
                )
                exp_lanecount, g = estimate_lanecount(
                    ls
                )  # Use the new, empirical computation of lanecount which has parity with
                tree = generate_triangular_grid_along_path_empirical(
                    pos=pos, velvec=velvec, dt=dt, g=g, r_e=ls.r_e
                )

            else:
                print(
                    f"Using dimensionelss lanecount estimator. The parameters alpha and r_e are dimensionless and velocity ought to be 1.0"
                )
                exp_lanecount, g = estimate_lanecount_fast(
                    ls
                )  # Use the old, dimensionless computation of lanecount
                tree = generate_triangular_grid_along_path_empirical(
                    pos=pos,
                    velvec=velvec,
                    dt=dt,
                    r_e=ls.r_e,
                    g=g,
                )
            # The double
            g_y = (np.sqrt(3) / 2) * g * 2
            noise_percent = 0.01

            lanecount_homogenous = 1 / (noise_percent**2)

            while t < genmeta.t_max - dt:
                if t > genmeta.t_max - 2 * dt:
                    print("The last")
                theta = genmeta.theta_change_per_sec * t + genmeta.theta

                if exp_lanecount > lanecount_homogenous:
                    print(
                        f"{genmeta.id}: SYNTH gen  (isotropic): Time t:{t:.2f} for lambda: {lamb:.2f}, theta {theta:.2f} Flux {flux_per_sec}/s. Lanecount: {exp_lanecount:.2f}"
                    )
                    hits = generate_synthetic_hits_isotropic(t, dt, ls)
                    if hits.size > 0:
                        events.extend(hits)
                else:
                    # Compute the change in theta over time

                    # Compute the new velocity vector
                    velvec = np.array([np.cos(theta), np.sin(theta)]) * ls.velocity
                    pos += dt * velvec

                    hits, pos, avg_lanes = generate_synthetic_hits_vectorized_3(
                        pos=pos,
                        theta=theta,
                        exp_lanecount=exp_lanecount,
                        t0=t,
                        dt=dt,
                        ls=ls,
                        lanesheet_tree=tree,
                        tile_width=g,
                        tile_height=g_y * 2,
                    )

                    total_added += len(hits)
                    if hits.size > 0:
                        events.extend(hits)

                    print(
                        f"{genmeta.id}: SYNTH gen (lightlane): Time t:{t:.2f} for lambda: {lamb:.2f}, theta {theta:.2f} Flux {flux_per_sec:.2f}/s. Lanecount: {exp_lanecount:.2f} real-lanecount : {avg_lanes:.2f} Diff {((avg_lanes / exp_lanecount)-1)* 100:.2f}%"
                    )

                t += dt

            l_idx += 1
    except Exception as e:
        raise e
    return events


def generate_synthetic_hits_isotropic_old(meta: LanesheetMetadata):
    """
    Generate isotropic (non-lightlane) synthetic photon hits for comparison.

    Parameters:
    - meta: contains camera metadata (start pos, dt, offset, theta)
    - lamb: wavelength
    - exp_count_per_dt: total expected photon count for the interval
    - N: time resolution of interpolation

    Returns:
    - camera_hits: array of synthetic hits with timestamps and positions
    """
    idt = meta.dt / meta.interpolation
    dt = meta.dt
    camera_hits = []

    pos = np.copy(meta.offset)
    velvec = np.array([np.cos(meta.theta), np.sin(meta.theta)])
    t = 0
    expected_per_idt = meta.exp_flux_per_sec / meta.interpolation
    halfwidth = meta.lambda_width / 2
    while t < dt:
        pos += velvec * idt

        # Generate photons uniformly per idt
        n_photons = np.random.poisson(expected_per_idt)
        for i in range(n_photons):
            camera_hits.append(
                np.array(
                    [
                        t + np.random.uniform(0, idt),
                        pos[0],
                        pos[1],
                        meta.lambda_center + np.random.uniform(-halfwidth, halfwidth),
                    ]
                )
            )
        t += idt

    return np.array(camera_hits)


def generate_dataset_for_experiment(
    theta_0: float,
    theta_change_er_sec: float,
    ls: LanesheetMetadata,
    dt: int = 1,
    t_max=10000,
    velocity: float = 1.0,
):
    photon_hits = []
    t0 = 0.0
    if ls.empirical:
        lc = estimate_lanecount(ls)
    else:
        lc = estimate_lanecount_fast(ls)
    g = ls.get_g()
    # Compute the initial unit vector angle
    velvec = np.array([np.cos(theta_0), np.sin(theta_0)])
    # A diophantine burst appears near phase = 0.0. Phase contains the number of seconds after such a diophantine burst
    # For example, a negative value here can be interesting for recreating diophantine bands
    pos = (ls.phase * velocity) * velvec
    perp_dir = np.array([-velvec[1], velvec[0]])
    pos += (ls.perp * velocity) * perp_dir
    t = 0
    while t < t_max:
        print(f"Generating for {ls.lambda_center} at time {t}")
        theta = theta_0 + theta_change_er_sec * t
        if lc > 100:
            more_hits = generate_synthetic_hits_isotropic(t0=t0, dt=dt, meta=ls)

            photon_hits.extend(more_hits)
        else:
            if ls.empirical:
                lanes, lanesheet_tree = generate_triangular_grid_along_path_empirical(
                    pos=pos, dt=dt, r_e=ls.r_e, g=g
                )
            else:
                lanes, lanesheet_tree = generate_triangular_grid_along_path(
                    start=pos, theta=theta, dt=dt, r_e=ls.r_e, g=g
                )

            more_hits, pos = generate_synthetic_hits_vectorized_3(
                pos=pos,
                t0=t,
                velocity=velocity,
                theta=theta,
                dt=dt,
                ls=ls,
                lanesheet_tree=lanesheet_tree,
            )
            photon_hits.extend(more_hits)
        t += dt

    photon_hits = np.array(photon_hits)
    photon_hits = photon_hits[np.argsort(photon_hits[:, 0])]
    return photon_hits


def generate_triangular_grid_along_path_empirical(pos, velvec, dt, g, r_e):
    """
    Generate 2D triangular (hexagonal) lattice points around a moving position.
    The lattice is aligned so that [0,0] lies on a grid node.
    """
    # local movement vector

    path_vec = velvec * dt
    path_length = np.linalg.norm(path_vec)
    if not np.isfinite(path_length) or path_length <= 0:
        raise ValueError("Path length must be positive and finite.")

    # nearest grid anchor offset

    offset = np.array([pos[0] % g, pos[1] % g])

    # how many columns/rows of grid points are needed to cover a radius r_e
    n_cols = int(np.ceil(r_e / g))
    n_rows = int(np.ceil((2 * r_e) / (np.sqrt(3) * g)))

    # pad to include motion
    pad_x = int(np.ceil(abs(path_vec[0]) / g)) + 2
    pad_y = int(np.ceil(abs(path_vec[1]) / g / (np.sqrt(3) / 2))) + 2

    dy = np.sqrt(3) / 2 * g
    lanes = []
    skipped = 0
    for row in range(-n_rows - pad_y, n_rows + pad_y):
        for col in range(-n_cols - pad_x, n_cols + pad_x):
            x = col * g + (0.5 * g if row % 2 else 0.0)
            y = row * dy
            point = pos + np.array([x, y]) - offset
            lanes.append(point)
            """
            if np.linalg.norm(point-pos) < r_e - path_vec:
                
                lanes.append(point)
            else:
                skipped +=1
            """

    # print(f"Included {len(lanes)}, skipped {skipped}")
    lanes = np.array(lanes)
    return cKDTree(lanes)


def generate_triangular_grid_along_path(start, theta, dt, g, r_e, velocity):
    """
    Generate a triangular grid aligned to [0,0]
    """
    # Compute snapped grid origin (lower-left anchor)
    anchor = (np.floor(start / g)) * g
    offset = start - anchor  # relative offset into grid cell
    velvec = np.array([np.cos(theta), np.sin(theta)]) * velocity
    # Shift the path by the offset so the grid is aligned at [0,0]
    rel_start = start - offset
    rel_end = rel_start + velvec * dt
    path_vec = rel_end - rel_start
    path_length = np.linalg.norm(path_vec)
    assert path_length > 0, "Path length cannot be None or nan"

    # Path directions
    path_dir = path_vec / path_length
    perp_dir = np.array([-path_dir[1], path_dir[0]])
    margin = r_e + g

    corners = [
        rel_start + perp_dir * margin,
        rel_start - perp_dir * margin,
        rel_end + perp_dir * margin,
        rel_end - perp_dir * margin,
    ]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    dy = (np.sqrt(3) / 2) * g
    points = []
    row = 0
    y = ymin
    while y <= ymax:
        x_offset = 0.5 * g if row % 2 else 0
        x = xmin
        while x <= xmax:
            px = x + x_offset
            py = y
            point = np.array([px, py])
            rel = point - rel_start
            proj_len = np.dot(rel, path_dir)
            if -margin <= proj_len <= path_length + margin:
                perp_dist = np.abs(np.dot(rel, perp_dir))
                if perp_dist <= margin:
                    points.append(point)
            x += g
        y += dy
        row += 1

    # Translate points back to the original anchor frame
    points = np.array(points) + offset
    return points, cKDTree(points)


def generate_synthetic_hits_isotropic(t0: float, dt: float, meta: LanesheetMetadata):
    halfwidth = meta.lambda_width / 2

    hits = []

    n_photons = np.random.poisson(meta.exp_flux_per_sec * meta.lambda_width * dt)
    if n_photons == 0:
        return np.empty((0, 4))
    # offset = np.array([np.cos(meta.phase), np.sin(meta.perp)])
    ccd_pos = np.random.uniform(-1.0, 1.0, (n_photons, 2))
    random_offsets = np.random.uniform(0, dt, size=n_photons)
    # pos_offsets = np.outer(random_offsets, velvec)
    # photon_positions = offset + pos_offsets
    random_wavelengths = meta.lambda_center + np.random.uniform(
        -halfwidth, halfwidth, size=n_photons
    )

    photons = np.column_stack(
        [
            t0 + random_offsets,
            ccd_pos[:, 0],
            ccd_pos[:, 1],
            random_wavelengths,
        ]
    )

    hits.append(photons)

    if len(hits) == 0:
        return np.empty((0, 4))  # important: must match 6 columns always!

    hits = np.vstack(hits)
    return hits


def generate_synthetic_hits_vectorized_3(
    pos: np.array,
    t0: float,
    dt: float,
    theta: float,
    exp_lanecount: float,
    ls: LanesheetMetadata,
    lanesheet_tree,
    tile_width: float,
    tile_height: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    try:
        # Compute the velocity vector used within this dt chunk
        velvec = np.array([np.cos(theta), np.sin(theta)]) * ls.velocity

        # Adaptive resolution of the simulation. If the r_e is small, it is more important to have fine resolution
        # If r_e is large enough, then it might be enough with a single subdivision within each dt.
        # But if r_e is significantly smaller than the range we cover within a single dt, then we must subdivide.
        subdivisions = max(1.0, np.ceil(ls.velocity / (ls.r_e / 100.0)))
        idt = (
            dt / subdivisions
        )  # The step-length in seconds for the subdivided generation

        exp_count_per_lane_per_sec = (
            ls.exp_flux_per_sec * ls.lambda_width
        ) / exp_lanecount
        # How many hits we expect to see per idt (second)
        exp_count_per_lane_per_idt = exp_count_per_lane_per_sec * idt

        t = 0.0
        halfwidth = ls.lambda_width / 2.0
        lanetime = []

        while t < dt:
            # Compute the position relative to the corner of an repeating lane tile (which is what the tree requires)
            # This hack means no need to reinitialize the ckdtree more than once per wavelength bin
            pos_rel_to_nearest_tile = np.array(
                [pos[0] % tile_width, pos[1] % tile_height]
            )
            # Obtain the list of indices that we are receiving from
            nearby_idxs = lanesheet_tree.query_ball_point(
                pos_rel_to_nearest_tile, r=ls.r_e
            )

            # Record time, number of nearby lanes, and the absolute position at this sub-step
            # Lanetime columns :
            # Time, instantatneous lanecount, lanesheet position x, lanesheet posistion y, wavelength, ccd_x, ccd_y
            lanetime.append(
                np.array(
                    [t0 + t, float(len(nearby_idxs)), pos[0], pos[1], 0.0, 0.0, 0.0]
                )
            )

            pos += velvec * idt
            t += idt

        lanetime = np.array(lanetime)
        # Guard against no samples
        if lanetime.size == 0:
            return np.empty((0, 4)), pos, 0.0

        sum_of_lanes = np.sum(lanetime[:, 1])
        if sum_of_lanes <= 0 or not np.isfinite(sum_of_lanes):
            return np.empty((0, 4)), pos, 0.0

        prob_of_idt = (
            lanetime[:, 1] / sum_of_lanes
        )  # Change lanes to probability that can be cumulated

        photon_hits = np.random.poisson(sum_of_lanes * exp_count_per_lane_per_idt)
        if photon_hits == 0:
            return np.empty((0, 4)), pos, 0.0

        # Precompute cumulative probabilities
        cumulative_prob = np.cumsum(prob_of_idt)

        # Uniform random numbers [0,1)
        random_uniform = np.random.uniform(0.0, 1.0, photon_hits)

        # Find indices corresponding to the cumulative distribution
        photon_time_index = np.searchsorted(
            cumulative_prob, random_uniform, side="right"
        )
        photon_time_index = np.clip(photon_time_index, 0, len(lanetime) - 1)

        # Generate time-jitter within the idt width
        photon_time_jitter = np.random.uniform(-idt / 2.0, idt / 2.0, photon_hits)

        # positions on the lanes (absolute coordinates) and apply motion jitter
        photon_positions = lanetime[photon_time_index].copy()
        photon_positions[:, [2, 3]] += photon_time_jitter[:, None] * velvec
        # Compute the avg_lane before pruning the empty idts
        avg_lanes = float(np.mean(lanetime[:, 1])) if lanetime.shape[0] > 0 else 0.0
        lanetime = lanetime[photon_time_index]
        # Appy time jitter to avoid the dt of the generator leaving a residual beat
        lanetime[:, 0] += photon_time_jitter
        # Apply wavelength jitter to avoid generated data clustering at wavelength bin centers
        lanetime[:, 4] = ls.lambda_center + np.random.uniform(
            -halfwidth, halfwidth, photon_hits
        )
        # Assign random coordinates on the CCD
        lanetime[:, 5] = np.random.uniform(-1, 1)
        lanetime[:, 6] = np.random.uniform(-1, 1)

        return lanetime, pos, avg_lanes
    except Exception as e:
        print(f"Error generating vectorized hits: {e}")
        raise e
