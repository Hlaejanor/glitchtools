import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio


from scipy.spatial import cKDTree
from common.helper import make_lambda_range
from common.generate_data import generate_triangular_grid_along_path
from common.fitsmetadata import FitsMetadata


class LanesheetForVideo:
    def __init__(
        self,
        lambda_center,
        lambda_bin_width,
        sheet_size,
        g,
        offset,
        theta,
        t_max,
        r_e,
        max_grid,
        duration,
        fps,
    ):
        self.lambda_center = lambda_center
        self.g = g
        self.r_e = r_e  # maybe set to something like 1.0 or 2.0
        self.lambda_bin_width = lambda_bin_width
        self.sheet_size = sheet_size
        self.offset = offset
        self.num_cols = int(self.sheet_size / self.g) * 2
        self.num_rows = int((2 * self.sheet_size / self.g) / np.sqrt(3)) * 2
        self.xmax = (self.num_cols) * self.g
        self.ymax = (self.num_rows) * (np.sqrt(3) / 2) * self.g
        self.r_e = r_e
        self.max_grid = 20
        self.fps = fps
        self.duration = duration

        self.exp_lanecount = self.estimate_lanecount_fast()
        if self.exp_lanecount > 10:
            self.ignore_gating_effects = True
            self.points = np.empty((0, 2))
        else:
            self.ignore_gating_effects = False
            max_distance = (
                self.max_grid * self.g
            )  # The distance we are intersted in probing
            self.velocity = max_distance / duration

            # self.points = self._generate_grid()
            self.points = self.generate_fixed_grid()
            if self.points.size == 0:
                print(f"Generated zero lightlanes for wavelength {self.lambda_center}")
                raise Exception(
                    f"Generated zero lightlanes for wavelength {self.lambda_center}"
                )
                self.points = np.empty((0, 2))

        # self.tree = cKDTree(self.points)

    def estimate_lanecount_fast(self):
        area_lane = np.pi * self.r_e**2
        area_cell = (np.sqrt(3) / 2) * self.g**2
        return area_lane / area_cell

    def compute_expected_lanecount(self):
        """By computing the total area of the overlapping lanes,
        divided by the total area of the grid, then we should get the total expected lanecount
        """

        grid_coverage_area = self.xmax * self.ymax

        total_lightlane_area = (
            (len(self.points) - (1 / 2 * self.num_cols * 2) - (1 / 2 * self.num_rows))
            * np.pi
            * self.r_e**2
        )
        if grid_coverage_area == 0:
            exp_lc = 0
        else:
            exp_lc = total_lightlane_area / grid_coverage_area
        print(f"Expected lanecount {exp_lc}")

        assert not np.isinf(exp_lc), "Exp_lc needs to be less than infinity"
        return exp_lc

    def visualize(self):
        """Visualize the lanesheet with circular lightlanes."""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect("equal")

        for x, y in self.points:
            circle = patches.Circle(
                (x, y), radius=self.r_e, edgecolor="black", facecolor="grey", alpha=0.5
            )
            ax.add_patch(circle)
        gridarea = patches.Rectangle(
            [0, 0], self.xmax, self.ymax, edgecolor="blue", facecolor="None"
        )
        ax.add_patch(gridarea)

        ax.set_xlim(-1, self.xmax)
        ax.set_ylim(-1, self.ymax)
        ax.set_title(f"Lanesheet Radius r_e = {self.r_e}, L^ : {self.exp_lanecount}")
        plt.axis("off")
        plt.show()

    def filter_points_along_path(self, points, start, end, r_e):
        """Keep only points within r_e of the path from start to end."""
        x0, y0 = start
        x1, y1 = end
        dx, dy = x1 - x0, y1 - y0
        norm = np.sqrt(dx**2 + dy**2)

        filtered = []
        for x, y in points:
            dist = abs((x - x0) * (y1 - y0) - (y - y0) * (x1 - x0)) / norm
            if dist <= r_e:
                filtered.append((x, y))
        return filtered

    def generate_fixed_grid(self):
        """
        Generate a large static triangular lattice (fixed in space).
        """
        dy = (np.sqrt(3) / 2) * self.g
        points = []
        row = 0

        x0 = self.offset[0] - self.offset[0] % self.g
        y0 = self.offset[1] - self.offset[1] % (self.g)

        y = -self.sheet_size / 2
        while y <= self.sheet_size / 2:
            x_offset = 0.5 * self.g if row % 2 else 0
            x = -self.sheet_size / 2
            while x <= self.sheet_size / 2:
                points.append(np.array([x + x0 + x_offset, y + y0]))
                x += self.g
            y += dy
            row += 1
        return np.array(points)  # No need for tree

    def generate_grid_within_range_wrapper(self, pos, theta, dt, g, r_e):
        lanes, lanesheet_tree = generate_triangular_grid_along_path(
            pos, theta, dt, g, r_e
        )

        return lanes

    def generate_grid_within_range(self, pos, theta, dt, grid_spacing, r_e):
        velvec = np.array([np.cos(theta), np.sin(theta)])

        end = (
            pos + velvec * dt * 2
        )  # Extending to ensure that there are lanes. To avoid zero counts in final time-bines

        # 1. Get bounding box
        pad = r_e + grid_spacing
        min_x = min(pos[0], end[0]) - pad
        max_x = max(pos[0], end[0]) + pad
        min_y = min(pos[1], end[1]) - pad
        max_y = max(pos[1], end[1]) + pad

        # 2. Generate triangle grid within bounding box
        points = []
        dy = (np.sqrt(3) / 2) * grid_spacing
        row = 0
        y = min_y
        #  The position at time t might be between lightlane sheets
        offset = np.array(
            [pos[0] % grid_spacing, pos[1] % ((np.sqrt(3) / 2) * grid_spacing)]
        )
        base_row = int(np.floor(min_y / dy))
        while y < max_y:
            x_offset = 0.5 * grid_spacing if base_row + row % 2 else 0
            x = min_x
            while x < max_x:
                px = x + x_offset
                if px <= max_x:
                    points.append(np.array([px, y]) - offset)
                x += grid_spacing
            y += dy
            row += 1

        return np.array(self.filter_points_along_path(points, pos, end, r_e))

    def sample(self, pos, resolution, v, t, t0, dt, expected_photons):
        """
        Simulate sampling from this wavelength-specific lanesheet.

        Parameters:
        - pos: Tuple (x, y), initial camera position
        - res: Tuple (resx, resy), CCD resolution (unused for now)
        - v: Tuple (vx, vy), velocity vector
        - dt: Time step duration
        - rho: Mean photon rate per lane (photons/s)

        Returns:
        - new_pos: Updated position
        - counts: Number of detected photon events (Poisson sampled)
        """
        bin_events = []

        # Interpolate the time bin
        N = 10
        if not self.ignore_gating_effects:
            print(
                f"t: {t:.2f}. Wavelength {self.lambda_center} using Lightlane sampling. Exp:count: {expected_photons:.2f}"
            )
            interpolated_dt = np.linspace(0, dt, N)
            for idt in interpolated_dt:
                new_pos = np.array(
                    [pos[0] + v[0] * (t0 + idt), pos[1] + v[1] * (t0 + idt)]
                )

                idxs = self.tree.query_ball_point(new_pos, r=self.r_e)
                lanecount = len(idxs)

                # Generate photons for the total interval
                expected_total_photons = (
                    (expected_photons / self.exp_lanecount) * lanecount / N
                )
                # Simulate received counts with shot noise for the entire dt
                photon_count = np.random.poisson(expected_total_photons)
                for j in range(0, photon_count):
                    # Generate random locations for the hits. This can later be used to generate position on the CCD for intra-CCD anisotropy
                    sampled_lambda = np.random.uniform(
                        self.lambda_center - self.lambda_bin_width / 2,
                        self.lambda_center + self.lambda_bin_width / 2,
                    )

                    bin_events.append(
                        np.array(
                            [
                                t + idt,
                                np.random.randint(0, resolution),
                                np.random.randint(0, resolution),
                                sampled_lambda,
                            ]
                        )
                    )
        else:  # If it makes no sense to interpolate because the number of photons is so small, we can just sample directly
            print(
                f"t: {t:.2f}. Wavelength {self.lambda_center} using isotropic sampling for {expected_photons:.2f}"
            )
            new_pos = np.array([pos[0] + v[0] * (t0 + dt), pos[1] + v[1] * (t0 + dt)])
            photon_count = np.random.poisson(expected_photons)
            for j in range(0, photon_count):
                # Generate random locations for the hits. This can later be used to generate position on the CCD for intra-CCD anisotropy

                sampled_lambda = np.random.uniform(
                    self.lambda_center - self.lambda_bin_width / 2,
                    self.lambda_center + self.lambda_bin_width / 2,
                )

                bin_events.append(
                    np.array(
                        [
                            t + np.random.uniform(0, dt),
                            np.random.randint(0, resolution),
                            np.random.randint(0, resolution),
                            sampled_lambda,
                        ]
                    )
                )

        return new_pos, bin_events


# Store event data
events = []

# For animation
generate_video = False
