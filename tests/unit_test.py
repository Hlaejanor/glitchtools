from common.fitsmetadata import FitsMetadata, load_fits_metadata
from common.generate_data import generate_synthetic_telescope_data
from common.fitsread import (
    fits_save_events_with_pi_channel,
    read_event_data_crop_and_project_to_ccd,
)
from event_processing.binning import add_wavelength_bin_columns, get_binned_datasets
from event_processing.var_analysis_plots import binning_process, make_standard_plots
import os


def chandra_data_pipeline():
    id = "default"
    meta = load_fits_metadata(id)

    events = generate_synthetic_telescope_data(
        meta, single_wavelength=None, lwidth=None
    )

    fits_save_events_with_pi_channel(events=events, filename=meta.raw_event_file)

    success, meta, pp, source_data = read_event_data_crop_and_project_to_ccd(meta)

    source_data = add_wavelength_bin_columns(meta, pp, source_data)
    binned_datasets = get_binned_datasets(source_data=source_data, meta=meta, pp=pp)

    for binned_df in binned_datasets:
        make_standard_plots(meta, binned_df, source_data)
