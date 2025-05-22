from dataclasses import dataclass, asdict, field
import json
import numpy as np
import pandas as pd
import os


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


@dataclass
class Spectrum:
    A: float
    lambda_0: float
    sigma: float
    C: float

    def __init__(self, A: float, lambda_0: float, sigma: float, C: float):
        self.A = A
        self.lambda_0 = lambda_0
        self.sigma = sigma
        self.C = C

    def clone(self):
        return Spectrum(self.A, self.lambda_0, self.sigma, self.C)

    def dict(self):
        return {k: v for k, v in asdict(self).items()}

    def to_string(self):
        return f"""Spectrum : \n 
        A :  {self.A} \n
        Lambda_0 : {self.lambda_0} \n 
        Sigma : {self.sigma} \n 
        C : {self.C} \n 
        """


@dataclass
class ComparePipeline:
    id: str
    A_fits_id: str
    B_fits_id: str
    pp_id: str
    gen_id: str
    downsample_N: int

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


@dataclass
class ProcessingParameters:
    id: str
    resolution: int
    max_wavelength: float
    min_wavelength: float
    wavelength_bins: float
    time_bin_seconds: float
    take_time_seconds: int
    anullus_radius_inner: float
    anullus_radius_outer: float
    processed_filename: str
    source_radius: float

    def to_string(self):
        return f"""Processing params : \n 
        Take first seconds :  {self.take_time_seconds} \n
        Virtual CCD resolution : {self.resolution} \n 
        Max lambda : {self.max_wavelength} \n 
        Min lambda : {self.min_wavelength} \n 
        Lambda bins : {self.wavelength_bins} \n 
        Time bin sec : {self.time_bin_seconds} \n 
        Processed filename: {self.processed_filename} \n
        Annullus radius inner : {self.anullus_radius_inner} \n
        Annullus radius outer: {self.anullus_radius_outer} \n
        Source radius : {self.source_radius}
        """

    def dict(self):
        return {k: v for k, v in asdict(self).items()}


@dataclass
class GenerationParameters:
    id: str
    alpha: float  # The scalar strength of the grid falloff
    lucretius: float  # The exponent for the grid/wavelength relation
    r_e: float  # Size of the r_e emitter
    theta: float  # Angle across the lanesheet
    t_max: int  # Length of observatin
    perp: float  # The position perpendicular to [0,0] through theta
    phase: float  # The position paralell to the camera path from [0, 0]
    max_wavelength: float
    min_wavelength: float
    star: str
    raw_event_file: str = None
    spectrum: Spectrum = field(default_factory=lambda: Spectrum(0, 0, 0, 0))

    def dict(self):
        return {k: v for k, v in asdict(self).items()}

    def get_g(self, lamb):
        return self.alpha * np.exp(self.lucretius * lamb)

    def get_maximum_energy(self):
        return 1.239841984 / self.min_wavelength

    def get_minimum_energh(self):
        return 1.239841984 / self.max_wavelength


@dataclass
class FitsMetadata:
    id: int
    raw_event_file: str
    synthetic: bool
    source_pos_x: float
    source_pos_y: float
    max_energy: float
    min_energy: float
    source_count: int
    star: str
    t_max: int
    gen_id: str

    apparent_spectrum: Spectrum = field(default_factory=lambda: Spectrum(0, 0, 0, 0))
    # generated_spectrum: Spectrum = field(default_factory=lambda: Spectrum(0, 0, 0, 0))

    def dict(self):
        return {k: v for k, v in asdict(self).items()}

    def get_area(self):
        return np.pi * self.source_radius**2

    def get_anullus_area(self):
        return (np.pi * self.anullus_radius_outer**2) - (
            np.pi * self.anullus_radius_inner**2
        )

    def get_kept_percent(self, cropped_count):
        assert cropped_count is not None, "Cropped count not a value"
        assert self.source_count is not None, "Source count not a value"
        # assert self.cropped_count != 0, "Cropped count not > 0"
        # assert self.source_count != 0, "Source count not > 0"
        if self.source_count == 0:
            return 0.0
        return (1 - cropped_count / self.source_count) * 100

    def get_min_wavelength(self):
        wavelength = 1.239841984 / self.max_energy  # self.min_energy in keV
        # print(f"Min wavelength: {wavelength} nm")
        return wavelength

    def get_max_wavelength(self):
        wavelength = 1.239841984 / self.min_energy  # self.max_energy in keV
        # print(f"Max wavelength: {wavelength} nm")
        return wavelength
