from common.fitsmetadata import FitsMetadata, load_fits_metadata
from common.generate_data import generate_synthetic_telescope_data
from common.fitsread import fits_save_from_generated, read_crop_and_project_to_ccd
from var_analysis.readandplotchandra import binning_process, var_analysis_plot
import os


def chandra_data_pipeline():
    id = "default"
    meta = load_fits_metadata(id)

    events = generate_synthetic_telescope_data(
        meta, single_wavelength=None, lwidth=None
    )

    fits_save_from_generated(events=events, filename=meta.raw_event_file)

    success, meta, pp, source_data = read_crop_and_project_to_ccd(meta)
    variability = binning_process(source_data, meta)

    var_analysis_plot(meta, source_data, variability)
