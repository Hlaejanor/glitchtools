from common.fitsread import (
    chandra_like_pi_mapping,
    ProcessingParameters,
    load_fits_metadata,
    load_processing_param,
)

from common.metadatahandler import (
    load_fits_metadata,
    save_fits_metadata,
    load_gen_param,
    save_gen_param,
    load_pipeline,
    save_pipeline,
    save_processing_metadata,
)

if __name__ == "__main__":
    pp = load_processing_param("default")
    max_pi = chandra_like_pi_mapping(pp.min_wavelength)
    min_pi = chandra_like_pi_mapping(pp.max_wavelength)

    print(f"Max pi in dataset {max_pi}, min pi in dataset {min_pi}")
    print(
        f"Recommended number of wavelength bins to avoid quantization of pi channel is = {max_pi-min_pi} "
    )
