import numpy as np
from scipy.ndimage import gaussian_filter1d
from astropy.stats import bayesian_blocks
from scipy.signal import find_peaks
from astropy.timeseries import LombScargle
from scipy.stats import chi2
from skimage import feature, transform
from common.fitsmetadata import (
    FitsMetadata,
    ChunkVariabilityMetadata,
    ProcessingParameters,
    ComparePipeline,
)

from event_processing.binning import get_binned_datasets
from common.fitsread import (
    fits_save_events_with_pi_channel,
    read_event_data_crop_and_project_to_ccd,
    fits_save_chunk_analysis,
    fits_read,
)
from pandas import DataFrame
import pandas as pd
from event_processing.vartemporal_plotting import Arrow


def chandrashift(source_data: DataFrame) -> DataFrame:
    """
    Perform Chandra-like time bin shifting on the input event data.

    Parameters:
    - source_data: DataFrame containing event data with 'Wavelength Bin' and 'Time' columns.

    Returns:
    The dataframe with shifted time bins

    Note:
    A wiring error in the HRC causes the time of an event to be associated with the following event,
    which may or may not be telemetered.
    The result is an error in HRC event timing that degrades accuracy from about 16 microsec to roughly the mean time between events.
    For example, if the trigger rate is 250 events/sec, then the average uncertainty in any time tag is less than 4 millisec.
    Reference: https://cxc.harvard.edu/proposer/POG/html/chap7.html#sec:hrc_anom

    This assigns the time of each event to the previous event. This is a problem for variability analysis that depend on wavelength, because it means that the
    time associated with a given event is actually the time of the next event, which may be in a different wavelength bin.
    This degrades the ability to do wavelength-dependent timing analysis.
    However, since the camera telemeters as many events as it can, the probability of losing an event is independent of wavelength, so the overall variability profile should be preserved.
    Therefore, with a probability equal to the fraction of events lost due to the telemetry, we can assume that the time of the next event is assigned to the current event.
    """

    print(
        f"Applying Chandra-like time bin shifting to event data, for details see code and https://cxc.harvard.edu/proposer/POG/html/chap7.html#sec:hrc_anom"
    )
    # Sort the data by time to ensure proper temporal order
    source_data.sort_values(by="time", inplace=True)
    # Extract time values excluding the first entry

    first_time_value = source_data["time"].iloc[0]
    second_time_value = source_data["time"].iloc[1]

    time_values = source_data["time"].iloc[1 : source_data.shape[0]].values
    # Extract all data except the last row
    data = source_data.loc[1:]
    # Assign the extracted time values back to the 'time' column, so that the timestam from index 1 is applied to index 0, etc.
    data["time"] = time_values

    assert (
        data["time"].iloc[0] == second_time_value
    ), "First time value should be the second original time value"
    assert (
        data["time"].iloc[0] != first_time_value
    ), "First time value should be different from the original first time value"

    return data
