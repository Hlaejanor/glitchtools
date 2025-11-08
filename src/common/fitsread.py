import numpy as np
from astropy.io import fits
import os
from astropy.table import Table
import pandas as pd
import matplotlib.pyplot as plt
from common.fitsmetadata import (
    FitsMetadata,
    ChunkVariabilityMetadata,
    ProcessingParameters,
    GenerationParameters,
)
from datetime import datetime
from common.metadatahandler import (
    load_fits_metadata,
    save_fits_metadata,
    save_chunk_metadata,
    load_processing_param,
)
from pandas import DataFrame
import uuid


def generate_metadata_from_fits(
    fits_path: str, identifier: str, synthetic: bool = False
) -> FitsMetadata:
    with fits.open(fits_path) as hdul:
        header = hdul[1].header
        data = hdul[1].data

        # Estimate timing
        times = data["time"]
        t_max = float(np.max(times) - np.min(times))

        # Position estimates (placeholder - you may want to refine)
        x = np.mean(data["x"])
        y = np.mean(data["y"])

        # Energy bounds — or calculate from PI channel if needed
        try:
            energy = data["energy"]  # if energy is in eV or keV directly
        except KeyError:
            # convert from PI if energy column doesn't exist
            pi = data["pi"]
            energy = pi * 14.6  # example: 1 PI ≈ 14.6 eV

        max_energy = float(np.max(energy) / 1000)
        min_energy = float(np.min(energy) / 1000)

        source_count = len(data)

        return FitsMetadata(
            id=identifier,
            raw_event_file=fits_path,
            synthetic=synthetic,
            source_pos_x=None,
            source_pos_y=None,
            max_energy=max_energy,
            min_energy=min_energy,
            source_count=source_count,
            star=identifier,
            t_max=int(t_max),
        )


def get_cached_filename(
    stage: str, meta: FitsMetadata, pp: ProcessingParameters
) -> tuple[bool, str]:
    cache_filename = f"cache/{stage}_{meta.id}_{pp.id}_h{meta.get_hash()[0:8]}{pp.get_hash()[0:8]}.fits"

    if os.path.exists(f"fits/{cache_filename}"):
        exists = True
    else:
        exists = False
    return exists, cache_filename


def fits_to_dataframe(fits_file: str) -> DataFrame:
    assert fits_file_exists(fits_file), f"FITS file {fits_file} does not exist!"
    # Path to the FITS file (update this with the correct path)
    print(f"Reading file {fits_file}")
    try:
        # Silly but efficient recover from stupid path error
        if fits_file[0:5] == "fits/":
            fits_file = fits_file[5:]
        with fits.open(f"fits/{fits_file}") as hdul:
            hdul.info()

            # Access the data table
            event_data = hdul[1].data  # Events are typically in the second HDU

            # Print column names and the first few rows
            # print("Columns in the event file:")
            # print(event_data.names)

        # Convert the FITS data to an Astropy Table
        table = Table(event_data)
        names = [name for name in table.colnames if len(table[name].shape) <= 1]
        if "time" in table.colnames:
            table.sort("time")
        df = table[names].to_pandas()

    except Exception as e:
        print(f"Error reading FITS file {fits_file}: {e}")
        raise e

    return df


def fits_read_cache_if_exists(cache_filename_path) -> DataFrame:
    # Path to the FITS file (update this with the correct path)
    print(f"Reading cached file {cache_filename_path}")

    # Silly but efficient recover from stupid path error
    if cache_filename_path[0:5] == "fits/":
        cache_filename_path = cache_filename_path[5:]
    with fits.open(f"fits/{cache_filename_path}") as hdul:
        hdul.info()

        # Access the data table
        event_data = hdul[1].data  # Events are typically in the second HDU

        # Print column names and the first few rows
        # print("Columns in the event file:")
        # print(event_data.names)
        # print("\nFirst 20 rows:")
        # print(event_data[:20])

    # Convert the FITS data to an Astropy Table
    table = Table(event_data)
    names = [name for name in table.colnames if len(table[name].shape) <= 1]
    if "time" in table.colnames:
        table.sort("time")
    df = table[names].to_pandas()

    # Print the first 20 row
    return df


def fits_file_exists(filename):
    filename = str.replace(filename, "//", "/")

    file_exists = os.path.isfile(filename)

    return file_exists


def fits_read(fits_file) -> DataFrame:
    # Path to the FITS file (update this with the correct path)
    print(f"Attempting fits file {fits_file}")
    try:
        # Silly but efficient recover from stupid path error
        if fits_file[0:5] == "fits/":
            fits_file = fits_file[5:]
        with fits.open(f"fits/{fits_file}") as hdul:
            hdul.info()

            # Access the data table
            event_data = hdul[1].data  # Events are typically in the second HDU

            # Print column names and the first few rows
            # print("Columns in the event file:")
            # print(event_data.names)
            # print("\nFirst 20 rows:")
            # print(event_data[:20])

        # Convert the FITS data to an Astropy Table
        table = Table(event_data)
        names = [name for name in table.colnames if len(table[name].shape) <= 1]
        if "time" in table.colnames:
            table.sort("time")
        df = table[names].to_pandas()

    except Exception as e:
        return None
    # Print the first 20 row
    return df


def crop_chandra_data(data, pos, radius):
    cropped_idx = np.abs(data["CCD X"] - pos[0]) < radius
    cropped_idy = np.abs(data["CCD Y"] - pos[1]) < radius

    combined = np.logical_and(cropped_idx, cropped_idy)
    croppped_data = data[combined].copy()

    return croppped_data, np.pi * radius**2


def mask_event_data(data, bright_pixel_array, radius):
    print("Removing hits close to existing sources")
    combined = data["Hit"] == 1
    print(f"Removing {len(bright_pixel_array)} pixels with radius {radius}")
    for pix in bright_pixel_array:
        outside_x = np.abs(data["CCD X"] - pix[0]) > radius
        combined = np.logical_and(combined, outside_x)
        outside_y = np.abs(data["CCD Y"] - pix[1]) > radius
        combined = np.logical_or(combined, outside_y)

    masked_data = data[combined].copy()

    return masked_data


def find_bright_pixels(data, n=5):
    """
    Find the N brightest pixels in a CCD image based on photon count.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain columns "CCD X", "CCD Y", and "Hit" (1 for photon).
    n : int
        Number of brightest pixels to return.

    Returns
    -------
    pd.DataFrame
        With columns ["CCD X", "CCD Y", "Counts"], sorted by descending Counts,
        limited to the top N.
    """
    # 1. Filter to actual photon hits (if your DataFrame has "Hit"=1 for hits)
    hits = data[data["Hit"] == 1]

    # 2. Group by (CCD X, CCD Y) and count
    pixel_counts = hits.groupby(["CCD X", "CCD Y"]).size().reset_index(name="Counts")
    print(f"Lenth of groups {len(pixel_counts)}")
    # 3. Sort in descending order by "Counts"
    pixel_counts_sorted = pixel_counts.sort_values(by="Counts", ascending=False)

    print("Inspecting the top 10 pixels counts")
    print(pixel_counts_sorted[0:10])
    if len(pixel_counts_sorted) < n:
        print(
            f"ERROR : Cannot return that many bright pixels, only have {len(pixel_counts_sorted)} groups"
        )
    u = []
    for i in range(0, min(n, len(pixel_counts_sorted))):
        u.append(
            [
                pixel_counts_sorted.iloc[i]["CCD X"],
                pixel_counts_sorted.iloc[i]["CCD Y"],
                pixel_counts_sorted.iloc[i]["Counts"],
            ]
        )

    return u


def quick_plot(event_data: Table):
    # Extract X and Y detector positions
    x = event_data["rawx"]
    y = event_data["rawy"]

    # Create a scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=1, color="blue")
    plt.title("Photon Distribution on the Detector")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.grid(True)
    plt.show()


def test_time_resolution(event_data: Table):
    # Extract the fractional part of each timestamp
    fractional_parts = event_data["time"] - np.floor(event_data["time"])

    # Check if there are any non-zero fractional parts
    has_fractional = np.any(fractional_parts > 0)

    print(f"Are there fractional parts? {'Yes' if has_fractional else 'No'}")
    if has_fractional:
        print(
            f"Example fractional parts: {fractional_parts[fractional_parts > 0][:10]}"
        )


def chandra_time(chandra_timestamp):
    chandra_epoch = 883609200  # 1998 Jan 1

    if chandra_timestamp < chandra_epoch:
        raise Exception("This cannot be chandra timestamp, before the epoch")


def pi_channel_to_wavelength_and_width(pi_channel):
    # Energy of the center
    energy_eV = 14.6 * pi_channel
    # Center wavelength
    wavelength_nm = 1239.84193 / energy_eV
    # Width
    delta_lambda_nm = (1239.84193 / energy_eV**2) * 14.6

    return wavelength_nm, delta_lambda_nm, pi_channel


def chandra_pi_center(wavelength_nm):
    """Deterministic mapping: wavelength (nm) → PI channel (int)"""
    energy_eV = 1239.84193 / wavelength_nm
    pi_channel = int(round(energy_eV / 14.6))
    return pi_channel


def chandra_like_pi_mapping(wavelength_nm, fwhm_eV=100.0):
    """
    Convert wavelength (in nm) to a Chandra-like PI channel value,
    mimicking instrumental energy smearing.

    Parameters:
    - wavelength_nm: array-like of wavelengths in nanometers.
    - fwhm_eV: float, full-width-half-maximum for energy smearing (default ~100 eV).

    Returns:
    - pi_channels: np.ndarray of integer PI channel values.
    """
    # Planck relation: E (eV) = 1239.84193 / λ (nm)
    energy_eV = 1239.84193 / wavelength_nm

    # Convert FWHM to standard deviation for Gaussian smearing
    sigma_eV = fwhm_eV / 2.355

    # Apply Gaussian noise to simulate energy resolution
    energy_smeared = np.random.normal(energy_eV, sigma_eV)

    # Convert to PI channel (Chandra ~14.6 eV per PI channel)
    pi_channels = np.round(energy_smeared / 14.6).astype(int)

    return pi_channels


def fits_save_cache(filename: str, df: DataFrame) -> bool:
    """
    Save FITS file from a pandas DataFrame by converting it to an Astropy Table.

    Parameters:
    - events: DataFrame with columns: [time, x, y, wavelength_nm]
    - meta: FitsMetadata

    Returns:
    - Updated meta with file path and source count.
    """
    print(f"Caching to {filename}")
    if not os.path.isdir("fits/cache"):
        os.mkdir("fits/cache")

    assert len(df) > 0, "Events DataFrame is empty"
    if filename[0:5] == "fits/":
        filename = filename[5:]
    filepath = os.path.join("fits/", filename)

    try:
        # Convert to Astropy Table
        table = Table.from_pandas(df)

        # Save as FITS
        table.write(filepath, format="fits", overwrite=True)

        return True

    except Exception as e:
        raise Exception(f"Error saving FITS metadata: {e}")


def fits_save(events: DataFrame, meta: FitsMetadata) -> FitsMetadata:
    """
    Save FITS file from a pandas DataFrame by converting it to an Astropy Table.

    Parameters:
    - events: DataFrame with columns: [time, x, y, wavelength_nm]
    - meta: FitsMetadata

    Returns:
    - Updated meta with file path and source count.
    """
    print("These are the column names")
    print(events.columns)
    assert len(events) > 0, "Events DataFrame is empty"
    if meta.synthetic:
        filename = f"./fits/synthetic_lightlane/{meta.id}.fits"
    else:
        raise Exception(
            f"Saving non-synthetic fits files not currently supported. Tried to save {meta.id}"
        )

    filepath = os.path.join("fits", filename)

    print(f"Saving dataset {meta.id} ({len(events)} observations) to {filepath}")

    try:
        # Convert to Astropy Table
        table = Table.from_pandas(events)

        # Save as FITS
        table.write(filepath, format="fits", overwrite=True)

        # Update metadata
        meta.raw_event_file = filename
        meta.source_count = len(events)

        return meta

    except Exception as e:
        raise Exception(f"Error saving FITS metadata: {e}")


def fits_save_chunk_analysis(
    variability_chunks: DataFrame, source_meta: FitsMetadata, pp: ProcessingParameters
) -> ChunkVariabilityMetadata:
    """
    Save chunk analysis dataset

    Parameters:
    - events: list or array of shape (N, 4) with columns:
        [time, x, y, wavelength_nm]
    - filename: output FITS filename (will be placed in ./fits/)
    """
    assert len(variability_chunks) > 0, "Events array was empty"

    chunk_meta_id = f"{source_meta.id}_{pp.id}"

    filename = f"fits/chunk_variability/{chunk_meta_id}.fits"
    print(f"Saving dataset {chunk_meta_id} ({len(variability_chunks)}) obs)")
    try:
        # bec1777ec69438f8fe65ba3adf01ab704f20477e4949250cdbc502ee9da08e6f
        #
        # Create FITS table from pandas DataFrame (use classmethod to preserve columns)
        if isinstance(variability_chunks, pd.DataFrame):
            event_table = Table.from_pandas(variability_chunks)
        else:
            # fallback: construct directly if it's already in a suitable format
            event_table = Table(variability_chunks)

        chunkmetadata = ChunkVariabilityMetadata(
            id=chunk_meta_id,
            source_meta_id=source_meta.id,
            pp_id=pp.id,
            fits_meta_hash=source_meta.get_hash(),
            pp_meta_hash=pp.get_hash(),
        )
        save_chunk_metadata(chunkmetadata)
        # Save events to fits file
        print(f"Writing {filename} with {len(event_table)} events")
        event_table.write(filename, format="fits", overwrite=True)

        return chunkmetadata
    except Exception as e:
        raise Exception("Error saving fitsmetadata ", e)
        print(f" Exception : {repr(e)}")


def fits_save_event_file(event_data: DataFrame, meta: FitsMetadata):
    print(
        f"Saving fits file {meta.id} into file {meta.raw_event_file} with {event_data.shape[0]} obs and {event_data.shape[1]} columns"
    )
    event_table = Table.from_pandas(event_data)

    event_table.write(meta.raw_event_file, format="fits", overwrite=True)

    return meta


def fits_save_events_with_pi_channel(
    events: list, genmeta: GenerationParameters, use_this_id: str = None
) -> tuple[FitsMetadata, DataFrame]:
    """
    Save synthetic or modified photon events to a FITS file with chandra like pi-mappintg to mitigate quantization artifacts in the low-energy regime.

    Parameters:
    - events: list or array of shape (N, 4) with columns:
        [time, x, y, wavelength_nm]
    - filename: output FITS filename (will be placed in ./fits/)
    """
    assert len(events) > 0, "Events array was empty"
    if use_this_id is None:
        meta_id = str(uuid.uuid4())[:8]  # unique short id
    else:
        meta_id = use_this_id
    filename = f"fits/{meta_id}.fits"
    print(
        f"Saving dataset {meta_id} ({len(events)}) obs. Used Generation Params {genmeta.id})"
    )
    try:
        events_array = np.array(events)

        # Create FITS table
        event_table = Table()
        event_table["time"] = events_array[:, 0]
        # event_table["lanecount"] = events_array[:, 1]
        # event_table["ll_x"] = events_array[:, 2]
        # event_table["ll_y"] = events_array[:, 3]
        event_table["pi"] = chandra_like_pi_mapping(events_array[:, 4])
        event_table["x"] = events_array[:, 5]
        event_table["y"] = events_array[:, 6]

        fitsmeta = FitsMetadata(
            id=meta_id,
            raw_event_file=filename,
            synthetic=True,
            source_pos_x=None,
            source_pos_y=None,
            max_energy=genmeta.get_maximum_energy(),
            min_energy=genmeta.get_minimum_energh(),
            source_count=len(events),
            star=genmeta.star,
            t_min=genmeta.t_min,
            t_max=genmeta.t_max,
            gen_id=genmeta.id,
            apparent_spectrum=genmeta.spectrum,
            ascore=None,
        )
        save_fits_metadata(fitsmeta)

        # Save to FITS
        print(f"Writing {filename} with {len(event_table)} events")
        event_table.write(filename, format="fits", overwrite=True)
        # Verify by reading back
        events = fits_read(fitsmeta.raw_event_file)
        return events, fitsmeta
    except Exception as e:
        raise Exception("Error saving fits file ", e)
        print(f" Exception : {repr(e)}")


def process_dataset(
    event_data,
    ccd_resolution,
    max_wavelength,
    min_wavelength,
) -> DataFrame:
    """
    Convert Chandra event data to a virtual CCD dataset.

    Parameters:
        event_data (pd.DataFrame): Original event data from the Chandra FITS file.
        ccd_resolution (int): Resolution of the virtual CCD (e.g., 1024 for 1024x1024).
        max_energy_keV (float): Maximum allowable photon energy to filter pile-up events.

    Returns:
        pd.DataFrame: Simplified dataset with relative time, CCD X, CCD Y, and wavelength.
    """
    event_data = event_data.copy()
    assert "x" in event_data.columns, "Need to have 'x' column in dataset"
    assert "y" in event_data.columns, "Need to have 'y' column in dataset"
    # Sort by time to ensure first event is earliest
    event_data = event_data.sort_values("time").reset_index(drop=True)
    print(f"Length 1.0 {len(event_data)}")
    # Set t = 0 for the first observation
    t0 = event_data["time"][0]

    print(f"Length after time reduction: {len(event_data)}")
    event_data = assign_sampled_wavelengths(event_data)
    before_filtering = len(event_data)
    # print(event_data.head(200))
    print(f"Removing entries with wavelength > {max_wavelength} and < {min_wavelength}")
    wavelength_mask = np.logical_and(
        event_data["Wavelength (nm)"] > min_wavelength,
        event_data["Wavelength (nm)"] < max_wavelength,
    )
    event_data = event_data[wavelength_mask]
    after_filtering = len(event_data)
    print(
        f"Filtered on wavelenth kept N {after_filtering} out of {before_filtering} ({(after_filtering/before_filtering*100)}%"
    )
    if len(event_data) == 0:
        raise Exception(
            f"Wavelength filtering kept N {after_filtering} out of {before_filtering} ({(after_filtering/before_filtering*100)}%)"
        )
    # Scale X, Y to CCD resolution
    x_min, x_max = event_data["x"].min(), event_data["x"].max()
    y_min, y_max = event_data["y"].min(), event_data["y"].max()

    assert x_max - x_min > 0, "If x_max - x_min is 0, then we're in trouble!"
    assert y_max - y_min > 0, "If y_max - y_min is 0, then we're in trouble!"
    event_data = event_data[np.isfinite(event_data["x"]) & np.isfinite(event_data["y"])]
    # If CCD resolution is a single integer (square grid)
    event_data["CCD X"] = (
        (event_data["x"] - x_min) / (x_max - x_min) * (ccd_resolution - 1)
    ).astype(int)

    event_data["CCD Y"] = (
        (event_data["y"] - y_min) / (y_max - y_min) * (ccd_resolution - 1)
    ).astype(int)

    event_data["Hit"] = True
    print(f"Length 4.0 {len(event_data)}")

    print(f"Length 5.0 {len(event_data)}")
    # Create a smaller dataset with the required columns
    simplified_data = event_data[
        ["time", "CCD X", "CCD Y", "Wavelength (nm)", "pi", "Hit"]
    ]

    if not isinstance(simplified_data, DataFrame):
        raise Exception("Expecting to return a DataFrame")
    return simplified_data


def assign_sampled_wavelengths(event_table: pd.DataFrame) -> pd.DataFrame:
    """Assign continuous wavelength values by sampling within each PI range."""
    # Convert PI to energy bounds
    pi = event_table["pi"]

    # PI channels correspond to energy bins 14.6 eV wide
    energy_low = 14.6 * (pi - 0.5)  # lower edge of PI bin
    energy_high = 14.6 * (pi + 0.5)  # upper edge of PI bin

    # Sample energy uniformly within each PI channel bin
    energy_sampled = np.random.uniform(energy_low, energy_high)

    # Convert energy to wavelength
    wavelength_sampled = 1239.84193 / energy_sampled

    event_table["Wavelength (nm)"] = wavelength_sampled
    event_table["Energy (eV)"] = energy_sampled

    return event_table


def crop_and_project_to_CCD(
    fitsmeta: FitsMetadata, pparams: ProcessingParameters, event_table: DataFrame
) -> tuple[bool, FitsMetadata, pd.DataFrame]:
    assert fitsmeta is not None, "Fitsmetadata params cannot be empty"
    assert pparams is not None, "Processing params cannot be empty"
    filtered_table = event_table[["time", "x", "y", "pi"]]

    # if fitsmeta.lowPI is not None and fitsmeta.highPI is not None:
    #     print(
    #        f"Filtering the ({'synthetic' if fitsmeta.synthetic else 'empiric'} dataset {fitsmeta.star} to only include PI channel in range [{fitsmeta.lowPI}, {fitsmeta.highPI}]"
    #    )
    #    mask = (event_table["pi"] >= fitsmeta.lowPI) & (
    #        event_table["pi"] < fitsmeta.highPI
    #    )
    #    filtered_table = event_table[mask]

    filtered_table.sort_values("time")

    first_time = filtered_table["time"].iloc[0]
    last_time = filtered_table["time"].iloc[-1]
    duration = last_time - first_time
    print(
        f"From time: {datetime.fromtimestamp(first_time).strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(f"To time: {datetime.fromtimestamp(last_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration} seconds")
    fitsmeta.t_max = duration

    # Write the filtered table to CSV
    # maxradius = (int(np.max([pparams.anullus_radius_outer, pparams.source_radius])),)
    if len(filtered_table) == 0:
        return False, fitsmeta, None
        raise Exception("Table got filtered empty")

    # Process the event data
    virtual_ccd_data = process_dataset(
        event_data=filtered_table,
        ccd_resolution=pparams.resolution,
        max_wavelength=pparams.max_wavelength,
        min_wavelength=pparams.min_wavelength,
    )
    if len(virtual_ccd_data) == 0:
        return False, fitsmeta, virtual_ccd_data

    # fitsmeta.source_count = len(virtual_ccd_data)
    # processed_filename = f"temp/m{fitsmeta.id}_pp{pparams.id}.csv"

    # virtual_ccd_data.to_csv(f"temp/raw_{meta.source_filename}", index=False)
    # print(f"Brightest pixel found at location [{brightest_pixels[0][0]:.0f}, {brightest_pixels[0][1]:.0f}]")
    if fitsmeta.source_pos_x and fitsmeta.source_pos_y:
        center = [int(fitsmeta.source_pos_x), int(fitsmeta.source_pos_y)]
    else:
        print("WARNING : Assuming that the radius cropping is from the center position")
        center = np.array([pparams.resolution / 2, pparams.resolution / 2])
    cropped_ccd_data, source_area = crop_chandra_data(
        virtual_ccd_data,
        center,
        pparams.source_radius,
    )
    if len(cropped_ccd_data) == 0:
        raise Exception("Cropped ccd_data was empty!")

    print(f"Size before removing bright sources {len(cropped_ccd_data)}. ")
    brightest_pixels = find_bright_pixels(cropped_ccd_data, 1000)
    masked_ccd_data = mask_event_data(cropped_ccd_data, brightest_pixels, 3)
    if len(masked_ccd_data) == 0:
        raise Exception("Masked ccd_data was empty!")
    print(f"Size after removing bright sources {len(masked_ccd_data)}. ")

    # print(f"Expects at least {take_target[2]} for pixel at x:{meta.source_pos_x}, y:{meta.source_pos_y}")
    # fitsmeta.cropped_count = len(cropped_ccd_data)

    print(
        f"Masking and cropping {fitsmeta.star} data set with {len(masked_ccd_data)} observations, reduction {fitsmeta.get_kept_percent(len(cropped_ccd_data)):.2f}%"
    )

    return (True, fitsmeta, masked_ccd_data)


def read_process_twin_file(meta: FitsMetadata) -> tuple[bool, pd.DataFrame]:
    print(f"Processing synthetic twin file {meta.synthetic_twin_event_file}")
    event_table = fits_read(meta.synthetic_twin_event_file)

    save_fits_metadata(meta)
    if len(event_table) == 0:
        raise Exception("Table was empty!")

    return crop_and_project_to_CCD(fitsmeta=meta, event_table=event_table)


def read_event_data_crop_and_project_to_ccd(
    fits_id: str, processing_param_id: str
) -> tuple[bool, FitsMetadata, ProcessingParameters, DataFrame]:
    fits_meta = load_fits_metadata(fits_id)
    assert fits_meta is not None, "Error"
    pp = load_processing_param(processing_param_id)
    assert pp is not None, "Error"

    print(
        f"Processing raw event file for : {fits_meta.star}, file {fits_meta.raw_event_file}."
    )
    print(pp.to_string())

    event_table = fits_read(fits_meta.raw_event_file)
    print(f"  -  Length: {len(event_table)}")
    fits_meta.source_count = event_table.shape[0]
    if len(event_table) == 0:
        raise Exception("Table was empty!")

    success, meta, events = crop_and_project_to_CCD(
        fitsmeta=fits_meta, pparams=pp, event_table=event_table
    )

    save_fits_metadata(meta=meta)

    return success, fits_meta, pp, events
