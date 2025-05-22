import numpy as np
import matplotlib.pyplot as plt
from common.powerdensityspectrum import PowerDensitySpectrum, exp_count_per_sec
from common.fitsmetadata import FitsMetadata, GenerationParameters
from common.fitsread import (
    fits_file_exists,
    fits_read,
    fits_save_from_generated,
    pi_channel_to_wavelength_and_width,
)

from scipy.spatial import cKDTree
from common.lanesheetMetadata import LanesheetMetadata

import imageio


from common.helper import (
    estimate_lanecount_fast,
)


def generate_synth_if_need_be(
    meta: FitsMetadata, pds: PowerDensitySpectrum, lamb=None, lwidth=None
):
    if fits_file_exists(meta.synthetic_twin_event_file):
        events = fits_read(meta.synthetic_twin_event_file)
    else:
        events = generate_synthetic_telescope_data(
            meta, single_wavelength=lamb, lwidth=lwidth
        )
        fits_save_from_generated(events, meta.synthetic_twin_event_file)

    return True


def generate_synthetic_telescope_data(
    genmeta: GenerationParameters, wavelength_bins=100, halt_at_count=np.inf
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
    sheet_size = (
        np.max(np.array([np.cos(genmeta.theta), np.sin(genmeta.theta)])) * genmeta.t_max
    )
    velvec = np.array([np.cos(genmeta.theta), np.sin(genmeta.theta)])

    lambda_nm = np.flip(
        np.linspace(genmeta.max_wavelength, genmeta.min_wavelength, wavelength_bins)
    )
    lambda_bin_width = np.ones(wavelength_bins) * np.abs((lambda_nm[0] - lambda_nm[1]))

    lambdas = np.stack((lambda_nm, lambda_bin_width), axis=1)
    l_idx = 0
    total_added = 0
    for lamb, lambda_bin_width in lambdas:
        if lamb > 0.12 and lamb < 0.15:
            print("HEre")
        t = 0.0
        flux_per_sec = exp_count_per_sec(
            lamb,
            genmeta.spectrum.A,
            genmeta.spectrum.lambda_0,
            genmeta.spectrum.sigma,
            genmeta.spectrum.C,
        )
        assert lambda_bin_width > 0, f"Lambda bin must be > 0, was {lambda_bin_width}"
        print(f"Flux per {lamb}:  {flux_per_sec} per sec from empirical lambda ")

        lanesheet_meta = LanesheetMetadata(
            id=f"{genmeta.id}_bin_{l_idx}",
            truth=False,
            lambda_center=lamb,
            lambda_width=lambda_bin_width,
            exp_flux_per_sec=flux_per_sec,
            start_t=0,
            g=genmeta.get_g(lamb),
            theta=genmeta.theta,
            r_e=genmeta.r_e,
            perp=genmeta.perp,
            phase=genmeta.phase,
        )

        dt = 100
        pos = (genmeta.phase + t) * velvec
        perp_dir = np.array([-velvec[1], velvec[0]])
        pos += perp_dir * genmeta.perp
        # Use the same x-ray distribution as the chandra dataset, to obtain an estimate for the number of photons to expect on average for this time bin length
        exp_lanecount = estimate_lanecount_fast(lanesheet_meta)
        while t < genmeta.t_max:
            if exp_lanecount > 4:
                print(
                    f"{genmeta.id}: Generating synthetic data  (isotropic): Time t:{t} for lambda: {lamb} : Flux {flux_per_sec}/s. Lanecount: {exp_lanecount}"
                )
                hits = generate_synthetic_hits_isotropic(t, dt, lanesheet_meta)
                if hits.size > 0:
                    events.extend(hits)
            else:
                print(
                    f"{genmeta.id}: Generating synthetic data  (lightlane): Time t:{t} for lambda: {lamb} : Flux {flux_per_sec}/s. Lanecount: {exp_lanecount}"
                )

                if True:
                    try:
                        points, tree = generate_triangular_grid_along_path(
                            pos,
                            lanesheet_meta.theta,
                            dt,
                            lanesheet_meta.g,
                            lanesheet_meta.r_e,
                        )
                    except Exception as e:
                        raise e
                else:
                    points, tree = generate_triangular_neighbourhood(
                        pos,
                        lanesheet_meta.theta,
                        meta.time_bin_seconds,
                        lanesheet_meta.g,
                        lanesheet_meta.r_e,
                    )
                hits = generate_synthetic_hits_vectorized_3(t, dt, lanesheet_meta, tree)
                total_added += len(hits)
                if hits.size > 0:
                    events.extend(hits)

                if total_added > halt_at_count:
                    print(
                        f"We reached the target at of {halt_at_count} events at {lamb}"
                    )
                    raise Exception(
                        f"We reached the target at of {halt_at_count} events at {lamb}"
                    )

            t += dt

        l_idx += 1
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


def generate_dataset_for_experiment(ls: LanesheetMetadata, dt: int = 100, t_max=10000):
    photon_hits = []
    t0 = 0.0
    lc = estimate_lanecount_fast(ls)
    g = ls.get_g()
    velvec = np.array([np.cos(ls.theta), np.sin(ls.theta)])
    offset = ls.phase * velvec
    perp_dir = np.array([-velvec[1], velvec[0]])
    offset += ls.perp * perp_dir
    t = 0
    while t < t_max:
        print(f"Generating for {ls.lambda_center} at time {t}")
        if lc > 5:
            more_hits = generate_synthetic_hits_isotropic(t0=t0, dt=dt, meta=ls)

            photon_hits.extend(more_hits)
        else:
            lanes, lanesheet_tree = generate_triangular_grid_along_path(
                start=offset, theta=ls.theta, dt=dt, r_e=ls.r_e, g=g
            )

            more_hits = generate_synthetic_hits_vectorized_3(
                t0=t, dt=dt, ls=ls, lanesheet_tree=lanesheet_tree
            )
            photon_hits.extend(more_hits)
        t += dt

    photon_hits = np.array(photon_hits)
    photon_hits = photon_hits[np.argsort(photon_hits[:, 0])]
    return photon_hits


def generate_triangular_neighbourhood(start, theta, dt, g, r_e):
    """
    Generate a triangular grid aligned to [0,0]
    """
    dy = (np.sqrt(3) / 2) * g

    # Align to base grid
    x_pos = start[0] - (start[0] % g)
    y_pos = start[1] - (start[1] % dy)

    # Maximum expected travel
    x_max = np.abs(np.cos(theta)) * dt
    y_max = np.abs(np.sin(theta)) * dt

    # Number of steps needed in each direction (+ margin)
    margin = r_e + g
    grid_x = int(np.ceil((x_max + margin) / g)) + 1
    grid_y = int(np.ceil((y_max + margin) / dy)) + 1

    points = []
    for row in range(-grid_y, grid_y + 1):
        for col in range(-grid_x, grid_x + 1):
            x_shift = 0.5 * g if (row % 2) else 0.0
            x = col * g + x_shift + x_pos
            y = row * dy + y_pos
            points.append([x, y])

    points = np.array(points)
    points.sort(axis=0)
    return points, cKDTree(points)


def generate_triangular_grid_along_path(start, theta, dt, g, r_e):
    """
    Generate a triangular grid aligned to [0,0]
    """
    # Compute snapped grid origin (lower-left anchor)
    anchor = (np.floor(start / g)) * g
    offset = start - anchor  # relative offset into grid cell
    velvec = np.array([np.cos(theta), np.sin(theta)])
    # Shift the path by the offset so the grid is aligned at [0,0]
    rel_start = start - offset
    rel_end = rel_start + velvec * dt
    path_vec = rel_end - rel_start
    path_length = np.linalg.norm(path_vec)

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


def generate_synthetic_hits_isotropic(t0: float, dt, meta: LanesheetMetadata):
    velvec = np.array([np.cos(meta.theta), np.sin(meta.theta)])
    perp_vec = np.array(
        [-np.sin(meta.theta), np.cos(meta.theta)]
    )  # perpendicular direction

    offset = meta.phase * velvec + meta.perp * perp_vec
    halfwidth = meta.lambda_width / 2

    hits = []

    n_photons = np.random.poisson(meta.exp_flux_per_sec * meta.lambda_width * dt)
    if n_photons == 0:
        return np.empty((0, 6))
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
        return np.empty((0, 6))  # important: must match 6 columns always!

    hits = np.vstack(hits)  # stack properly
    return hits


def generate_synthetic_hits_vectorized_3(
    t0: float, dt: float, ls: LanesheetMetadata, lanesheet_tree
):
    try:
        velvec = np.array([np.cos(ls.theta), np.sin(ls.theta)])
        interpolation = (ls.r_e / np.linalg.norm(velvec)) * 100
        idt = dt / interpolation
        exp_lanecount = estimate_lanecount_fast(ls)
        exp_count_per_lane_per_sec = (
            ls.exp_flux_per_sec * ls.lambda_width
        ) / exp_lanecount
        exp_count_per_lane_per_idt = exp_count_per_lane_per_sec * idt
        pos = ls.phase * velvec

        perp_dir = np.array([-velvec[1], velvec[0]])
        pos += ls.perp * perp_dir
        hits = []
        t = 0
        halfwidth = ls.lambda_width / 2
        times = []
        lanes = []
        positions = []
        while t < dt:
            # Find nearby lightlanes (using expanded radius)
            nearby_idxs = lanesheet_tree.query_ball_point(pos, r=ls.r_e)
            if nearby_idxs:
                times.append(t0 + t)
                lanes.append(len(nearby_idxs))
                positions.append(pos)

            pos += velvec * idt
            t += idt

        times = np.array(times)  # The points in time that were close enough
        lanes = np.array(lanes)
        if len(positions) == 0:
            positions = np.empty((0, 6))
        else:
            positions = np.array(positions)

        n_idt = idt * len(times)  # Number of idt that hit lanes

        sum_of_lanes = np.sum(lanes)
        p_planes = (
            lanes / sum_of_lanes
        )  # Change lanes to probability that can be cumulated

        photon_hits = np.random.poisson(sum_of_lanes * exp_count_per_lane_per_idt)
        if photon_hits == 0:
            return np.empty((0, 6))
        # Precompute cumulative probabilities
        cumulative_prob = np.cumsum(p_planes)

        # Uniform random numbers [0,1)
        random_uniform = np.random.uniform(0, 1, photon_hits)

        # Find indices corresponding to the cumulative distribution
        photon_time_index = np.searchsorted(cumulative_prob, random_uniform)

        # Generate time-jitter within the idt width

        photon_time_jitter = np.random.uniform(-idt / 2, idt / 2, photon_hits)
        ccd_pos = np.random.uniform(-1.0, 1.0, (photon_hits, 2))

        t_samples = times[photon_time_index] + photon_time_jitter
        photon_positions = positions[photon_time_index].copy()
        photon_positions += photon_time_jitter[:, None] * velvec
        photon_lambdas = ls.lambda_center + np.random.uniform(
            -halfwidth, halfwidth, photon_hits
        )
        hits = np.column_stack(
            [
                t_samples,
                ccd_pos[:, 0],
                ccd_pos[:, 1],
                photon_lambdas,
            ]
        )
        return hits
    except Exception as e:
        print(f"Error ", e)

        return np.empty((0, 5))


def generate_synthetic_hits_vectorized_2(
    t0: float, meta: LanesheetMetadata, lanesheet_tree
):
    dt = meta.dt

    velvec = np.array([np.cos(meta.theta), np.sin(meta.theta)])
    meta.interpolation = (meta.r_e / np.linalg.norm(velvec)) * 10

    idt = meta.dt / meta.interpolation
    exp_lanecount = estimate_lanecount_fast(meta)
    exp_count_per_lane = meta.exp_flux_per_sec / exp_lanecount
    exp_count_per_lane_per_idt = exp_count_per_lane / meta.interpolation

    camera_hits = []
    pos = np.copy(meta.offset)
    t = 0
    halfwidth = meta.lambda_width / 2
    while t < dt:
        # Find nearby lightlanes (using expanded radius)
        nearby_idxs = lanesheet_tree.query_ball_point(pos, r=meta.r_e)
        if nearby_idxs:
            n_lightlanes = len(nearby_idxs)
            n_photons = np.random.poisson(
                exp_count_per_lane_per_idt * n_lightlanes * meta.lambda_width
            )
            if n_photons == 0:
                continue
            # Vector of random time offsets within [0, idt]
            t_offsets = np.random.uniform(0, idt, n_photons)
            ccd_pos = np.random.uniform(-1.0, 1.0, (n_photons, 2))
            t_samples = t0 + t + t_offsets

            # Displacement along the camera path
            displacements = np.outer(t_offsets, velvec)
            photon_positions = pos + displacements
            photon_lambdas = meta.lambda_center + np.random.uniform(
                -halfwidth, halfwidth, n_photons
            )

            hits = np.column_stack(
                [
                    t_samples,
                    ccd_pos,
                    photon_positions,
                    photon_lambdas,
                ]
            )
            camera_hits.append(hits)
        print(
            f"Sampled {len(hits)} for lambda {meta.lambda_center} (+m {meta.lambda_width/2})"
        )
        pos += velvec * idt
        t += idt

    return np.vstack(camera_hits) if camera_hits else np.empty((0, 4))


def generate_synthetic_hits_vectorized(
    t0: float, meta: LanesheetMetadata, lanesheet_tree
):
    idt = meta.dt / meta.interpolation
    dt = meta.dt
    exp_lanecount = estimate_lanecount_fast(meta)
    velvec = np.array([np.cos(meta.theta), np.sin(meta.theta)])
    motion_margin = idt * np.linalg.norm(velvec)
    r_query = meta.r_e + motion_margin

    exp_count_per_lane = meta.exp_flux_per_sec / exp_lanecount
    exp_count_per_lane_per_idt = exp_count_per_lane / meta.interpolation
    oversample_ratio = (np.pi * r_query**2) / (np.pi * meta.r_e**2)
    adjusted_production = exp_count_per_lane_per_idt * oversample_ratio

    camera_hits = []
    pos = np.copy(meta.offset)
    t = 0
    halfwidth = meta.lambda_width / 2
    while t < dt:
        # Find nearby lightlanes (using expanded radius)
        nearby_idxs = lanesheet_tree.query_ball_point(pos, r=r_query)

        if nearby_idxs:
            n_lightlanes = len(nearby_idxs)
            n_photons = np.random.poisson(adjusted_production * n_lightlanes)
            if n_photons == 0:
                continue
            # Vector of random time offsets within [0, idt]
            t_offsets = np.random.uniform(0, idt, n_photons)
            ccd_pos = np.random.uniform(-1.0, 1.0, (n_photons, 2))
            t_samples = t0 + t + t_offsets

            # Displacement along the camera path
            displacements = np.outer(t_offsets, velvec)
            photon_positions = pos + displacements
            photon_lambdas = meta.lambda_center + np.random.uniform(
                -halfwidth, halfwidth, n_photons
            )

            # Query all points at once
            accepted_mask = np.array(
                [
                    len(lanesheet_tree.query_ball_point(p, r=meta.r_e)) > 0
                    for p in photon_positions
                ]
            )

            accepted_sheet_positions = photon_positions[accepted_mask]
            accepted_ccd_pos = ccd_pos[accepted_mask]
            accepted_times = t_samples[accepted_mask]
            accepted_lambdas = photon_lambdas[accepted_mask]
            hits = np.column_stack(
                [
                    accepted_times,
                    accepted_ccd_pos,
                    accepted_sheet_positions,
                    accepted_lambdas,
                ]
            )
            camera_hits.append(hits)

        pos += velvec * idt
        t += idt

    return np.vstack(camera_hits) if camera_hits else np.empty((0, 4))
