from dataclasses import dataclass, asdict, field
import json
import numpy as np
import pandas as pd
import os
import hashlib
from typing import Literal


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
    time_bins_from: float
    time_bins_to: float
    time_bin_widths_count: float
    time_bin_chunk_length: int
    take_time_seconds: int
    anullus_radius_inner: float
    anullus_radius_outer: float
    processed_filename: str
    phase_bins: int
    take_top_variability_count: int | None
    padding_strategy: str
    source_radius: float
    percentile: float
    chunk_counts: int
    downsample_strategy: str
    downsample_target_count: int

    variability_type: Literal[
        "Excess Variability",
        "Variability Excess Adjacent",
        "Excess Variability Smoothed",
        "Fano Excess Local Variability",
        "Fano Excess Global Variability",
        "Odd even contrast",
        "Variability Excess Smoothed Adjacent",
    ]

    def to_string(self):
        return f"""Processing params : \n 
        Take first seconds :  {self.take_time_seconds} \n
        Virtual CCD resolution : {self.resolution} \n 
        Max lambda : {self.max_wavelength} \n 
        Min lambda : {self.min_wavelength} \n 
        Lambda bins : {self.wavelength_bins} \n 
        Time bin sec : {self.time_bin_seconds} \n 
        Time bins from : {self.time_bins_from} \n 
        Time bin widths count : {self.time_bin_widths_count} \n 
        Time bins to : {self.time_bins_to} \n 
        Time bin chunk count :  {self.time_bin_chunk_length} \n 
        Processed filename: {self.processed_filename} \n
        Annullus radius inner : {self.anullus_radius_inner} \n
        Annullus radius outer: {self.anullus_radius_outer} \n
        Source radius : {self.source_radius} \n
        Take top variability count : {self.take_top_variability_count} \n
        Wavelength bin padding strategy :  {self.padding_strategy}
        Wavelength bin downsampling strategy :  {self.downsample_strategy}
        Downsample target count :  {self.downsample_target_count}
        Percentile test :  {self.percentile} \n
        Chunk counts :  {self.chunk_counts} \n
        Variability type :  {self.variability_type}
        """

    # This is same as to_string, except some parameters are removed such as take_top_variability_count.
    # Including these would lead to cahche invalidation too early
    def to_string_cachebreaker(self):
        return f"""Processing params : \n 
        Take first seconds :  {self.take_time_seconds} \n
        Virtual CCD resolution : {self.resolution} \n 
        Max lambda : {self.max_wavelength} \n 
        Min lambda : {self.min_wavelength} \n 
        Lambda bins : {self.wavelength_bins} \n 
        Time bin sec : {self.time_bin_seconds} \n 
        Time bins from : {self.time_bins_from} \n 
        Time bin widths count : {self.time_bin_widths_count} \n 
        Time bins to : {self.time_bins_to} \n 
        Processed filename: {self.processed_filename} \n
        Annullus radius inner : {self.anullus_radius_inner} \n
        Annullus radius outer: {self.anullus_radius_outer} \n
        Source radius : {self.source_radius} \n
        Wavelength bin padding strategy :  {self.padding_strategy} \n
        Wavelength bin downsampling strategy :  {self.downsample_strategy} \n
        Downsample target count :  {self.downsample_target_count} \n
        Percentile test :  {self.percentile} \n
        Chunk counts :  {self.chunk_counts} \n
        Variability type :  {self.variability_type} \n
        """

    def dict(self):
        return {k: v for k, v in asdict(self).items()}

    def get_maximum_energy(self):
        return 1.239841984 / self.min_wavelength

    def get_minimum_energh(self):
        return 1.239841984 / self.max_wavelength

    def get_hash(self):
        # return self.to_string()
        return hashlib.sha256(self.to_string_cachebreaker().encode("utf-8")).hexdigest()


@dataclass
class GenerationParameters:
    id: str
    alpha: float  # The scalar strength of the grid falloff
    lucretius: float  # The exponent for the grid/wavelength relation
    r_e: float  # Size of the r_e emitter
    theta: float  # Start-angle across the lanesheet (radians)
    theta_change_per_sec: float  # Change in theta per second (radians)
    t_min: float  # Lowest time in the dataset
    t_max: int  # Length of observatin
    perp: float  # The position perpendicular to [0,0] through theta
    phase: float  # The position paralell to the camera path from [0, 0]
    max_wavelength: float
    min_wavelength: float
    star: str
    raw_event_file: str = None
    spectrum: Spectrum = field(default_factory=lambda: Spectrum(0, 0, 0, 0))

    def to_string(self):
        return f"""Generation params : \n 
        Id :  {self.id} \n
        Alpha : {self.alpha} \n 
        Lucretius : {self.lucretius} \n 
        Curvature change per sec : {self.theta_change_per_sec} \n,
        r_e : {self.r_e} \n 
        Theta : {self.theta} \n 
        t_min : {self.t_min} \n 
        t_max : {self.t_max} \n 
        Perp : {self.perp} \n 
        Phase : {self.phase} \n 
        Curvatre change per sec : {self.theta_change_per_sec} \n
        Max lambda : {self.max_wavelength} \n 
        Min lambda : {self.min_wavelength} \n 
        Star : {self.star} \n
        Raw event file: {self.raw_event_file} \n
        Spectrum : {self.spectrum.to_string()}
        """

    def dict(self):
        return {k: v for k, v in asdict(self).items()}

    def get_g(self, lamb):
        return self.alpha * np.exp(self.lucretius * lamb)

    def get_maximum_energy(self):
        return 1.239841984 / self.min_wavelength

    def get_minimum_energh(self):
        return 1.239841984 / self.max_wavelength

    def get_hash(self):
        # return self.to_string()
        return hashlib.sha256(self.to_string().encode("utf-8")).hexdigest()


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
    t_min: int
    t_max: int
    gen_id: str
    ascore: float
    apparent_spectrum: Spectrum = field(default_factory=lambda: Spectrum(0, 0, 0, 0))

    # generated_spectrum: Spectrum = field(default_factory=lambda: Spectrum(0, 0, 0, 0))
    def to_string(self):
        return f"""Id : {self.id}\n 
        raw_event_file :  {self.raw_event_file} \n
        Lsyntheticambda_0 : {self.synthetic} \n 
        source_pos_x : {self.source_pos_x} \n 
        source_pos_y : {self.source_pos_y} \n 
        max_energy : {self.max_energy} \n 
        min_energy : {self.min_energy} \n 
        source_count : {self.source_count} \n 
        star : {self.star} \n
        t_min : {self.t_min} \n
        t_max : {self.t_max} \n
        gen_id : {self.gen_id} \n
        """

    def get_apparent_spectrum_str(self):
        return self.apparent_spectrum.to_string()

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

    def get_hash(self):
        return hashlib.sha256(self.to_string().encode("utf-8")).hexdigest()
        # return self.to_string()


@dataclass
class ChunkVariabilityMetadata:
    id: int
    source_meta_id: str
    pp_id: str
    fits_meta_hash: str
    pp_meta_hash: str

    # Speedy way to check if the metadata from which the existing ChunkVariability file is composed has changed. Without change to the parameters or source file,
    # we can reuse the cached ChunkVariabiliy dataset

    def is_cache(self, fits_meta: FitsMetadata, pp: ProcessingParameters):
        fitshash = fits_meta.get_hash()
        if fitshash != self.fits_meta_hash:
            return False
        pphash = pp.get_hash()
        if pphash != self.pp_meta_hash:
            return False
        return True

    def to_string(self):
        return f"""ChunkVariabilityMetadata : \n 
        Source meta id :  {self.source_meta_id} \n
        Processing param id : {self.pp_id} \n 
        Fits meta hash : {self.fits_meta_hash} \n 
        Pp meta hash : {self.pp_meta_hash} \n 
        """

    def get_metafile_path(self):
        return f"meta_files/chunk_variability/{self.get_id()}.json"

    def get_fits_path(self):
        return f"chunk_variability/{self.get_id()}.fits"

    def get_id(self):
        return f"{self.source_meta_id}_{self.pp_id}"

    def dict(self):
        return {k: v for k, v in asdict(self).items()}

    def get_hash(self):
        # return self.to_string()
        return hashlib.sha256(self.to_string().encode("utf-8")).hexdigest()


def get_cache(stage: str, meta: FitsMetadata, pp: ProcessingParameters):
    return hashlib.sha256(f"{stage}_{meta.get_hash()}_{pp.get_hash()}")
