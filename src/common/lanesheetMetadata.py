from dataclasses import dataclass, asdict
import json
import numpy as np
import os


@dataclass
class LanesheetMetadata:
    id: int
    truth: bool  # True if the data is used to generate a dataset, false if it summarises an estimate
    lambda_center: float
    lambda_width: float
    exp_flux_per_sec: float
    alpha: float
    lucretius: float
    r_e: float
    theta: float
    perp: float
    phase: float
    alpha_tolernace: float = None
    lucretius_tolerance: float = None
    r_e_tolerance: float = None
    theta_tolerance: float = None
    perp_tolerance: float = None
    phase_tolerance: float = None

    def dict(self):
        return {k: v for k, v in asdict(self).items()}

    def make_estimate_meta(self, new_id):
        est = LanesheetMetadata(
            id=new_id,
            truth=False,
            data_file=self.data_file,
            lambda_center=self.lambda_center,
            lambda_width=self.lambda_width,
            exp_flux_per_sec=self.exp_flux_per_sec,
            alpha=self.alpha,
            lucretius=self.lucretius,
            r_e=self.r_e,
            theta=self.theta,
            pert=self.perp,
            phase=self.phase,
            theta_tolerance=self.theta_tolerance,
            g_tolerance=self.g_tolerance,
            offset_tolerance=self.offset_tolerance,
        )
        return est

    def get_g(self):
        return self.alpha * np.exp(self.lucretius * self.lambda_center)

    def get_projected_g(self):
        return self.get_g() / np.cos(self.theta)


def save_lanesheet_metadata(metadata: LanesheetMetadata) -> None:
    """
    Serialize the LanesheetMetadata object to JSON and write it to a file.
    """
    # Convert the dataclass to a dictionary
    csv_path = f"meta/ls_{metadata.id}.json"
    with open(csv_path, "a") as f:
        # Write the dictionary in a human-readable (pretty-printed) way
        json.dump(metadata.dict(), f, indent=4)


def load_lanesheet_metadata(filename: str) -> LanesheetMetadata:
    """
    Read JSON data from a file and reconstruct a LanesheetMetadata object.
    """
    print(f"Loading experiment metadata {filename}")
    with open(filename, "r") as f:
        data = json.load(f)

    if "t_max" not in data:
        data["t_max"] = data["dt"]

    metadata = LanesheetMetadata(
        id=data["id"],
        truth=data["truth"],
        data_file=data["data_file"],
        lambda_center=data["lambda_center"],
        lambda_width=data["lambda_width"],
        exp_flux_per_sec=data["exp_flux_per_dt"],
        alpha=data["alpha"],
        lucretius=data["lucretius"],
        r_e=data["r_e"],
        velocity=data["velocity"],
        theta=data["theta"],
        dt=data["dt"],
        perp=data["perp"],
        phase=data["phase"],
        r_e_tolerance=data["r_e_tolerance"],
        alpha_tolerance=data["alpha_tolerance"],
        lucretius_tolerance=data["lucretius_tolerance"],
        theta_tolerance=data["theta_tolerance"],
        phase_tolerance=data["phase_tolerance"],
        perp_tolerance=data["perp_tolerance"],
    )
    if metadata.offset is None:
        metadata.offset = np.array([0, 0])
    return metadata
