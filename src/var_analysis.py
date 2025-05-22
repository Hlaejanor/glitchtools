import sys
import os
from var_analysis.plotting import compute_time_variability
from common.fitsread import fits_save_from_generated, read_crop_and_project_to_ccd
from common.helper import randomly_sample_from, get_duration
from common.fitsmetadata import FitsMetadata, ProcessingParameters
from common.metadatahandler import (
    load_fits_metadata,
    load_best_metadata,
    load_processing_param,
    load_pipeline,
    load_gen_param,
)
from var_analysis.readandplotchandra import binning_process, var_analysis_plot
from common.generate_data import generate_synthetic_telescope_data
import argparse

version = 1.0


def regenerate_synth_file(meta: FitsMetadata):
    print(f"Regenerating synthetic file for meta {meta.id}")
    log = []
    events = generate_synthetic_telescope_data(
        meta, single_wavelength=None, lwidth=None
    )
    log.append(f"Saving synthetic event file to fits file :{meta.raw_event_file}")
    fits_save_from_generated(events=events, filename=meta.raw_event_file)


def load_fits_and_process(
    meta: FitsMetadata, pp: ProcessingParameters, handle: str, N: int
):
    log = []
    log.append(
        f"{handle} : Load and process fits file - id:{meta.id} {meta.raw_event_file}"
    )

    duration = get_duration(meta, pp)

    synth_event_file_exists = os.path.isfile(meta.raw_event_file)

    if not synth_event_file_exists:
        log.append(
            f"Could not fin pre-generated data-file. Regenerate using parameters from meta :{meta.id}"
        )
        regenerate_synth_file()

    success, meta, pp, source_data = read_crop_and_project_to_ccd(meta.id, pp.id)
    if not success:
        log.append(f"Dataset rejected, see ")
        return log
    log.append(
        f"{handle} : Downsampling  to : {N} ({(N / len(source_data))*100:.0f}%). Length of real_data file {len(source_data)}"
    )
    # print(source_reduced.head(2000))
    # source_reduced = randomly_sample_from(source_data, N)
    log.append(f"{handle} : Process dataset - id:{meta.id} {meta.raw_event_file}")
    binned, meta = binning_process(source_data, meta, pp)

    variability = compute_time_variability(source_data=binned, duration=duration)
    var_analysis_plot(meta, pp, source_data, variability)
    log.append(f"{handle} : Finished processing dataset - id:{meta.id}")
    return log


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare two metadata files and optionally set downsampling."
    )
    parser.add_argument(
        "-p",
        "--pipe",
        type=str,
        default=None,
        help="Select the experiment pipe name ",
    )

    parser.add_argument(
        "-a",
        "--meta_a",
        type=str,
        default=None,
        help="Path to first metadata JSON file (default: meta_1.json)",
    )
    parser.add_argument(
        "-b",
        "--meta_b",
        type=str,
        default=None,
        help="Path to B fits-file JSON file (default: best synthetic metadata)",
    )
    parser.add_argument(
        "-pp",
        "--pp",
        type=str,
        default=None,
        help="Processing Params Processing (default)",
    )

    parser.add_argument(
        "-g",
        "--gen",
        type=str,
        default=None,
        help="Generation Parameters (default to best)",
    )

    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        default=100000,
        help="Number of samples for downsampling (default: 100000)",
    )

    args = parser.parse_args()
    return args


def main():
    print(f"Chandra Lightlane Variability analysis v{version}")
    args = parse_arguments()
    N = args.n_samples
    log = []
    log.append(f"Both : Using N = {N}")

    if args.pipe is not None:
        pipeline_meta = load_pipeline(args.pipe)
        assert pipeline_meta is not None, "Pipeline metadata was 0"
        if args.meta_a is not None:
            raise Exception(
                f"You cannot both provide a pipe and a specific meta id, try modifying the /meta_files/pipelines/{pipeline_meta.id}_pipe.json"
            )
        if args.pp is not None:
            raise Exception(
                f"You cannot use a different ProcessingParameter id {args.pp} than what is specified in the pipeline file. Try modifying /meta_files/pipelines/{pipeline_meta.id}_pipe.json instead"
            )

        metaA = load_fits_metadata(pipeline_meta.A_fits_id)
        assert metaA.t_max > 0, "T_max has not been set. "
        pp = load_processing_param(pipeline_meta.pp_id)

        logA = load_fits_and_process(metaA, pp, "A", N=N)
        log.extend(logA)
        # B is set:
        filename = f"meta_files/fits/meta_{pipeline_meta.B_fits_id}.json"
        if os.path.isfile(filename):
            metaB = load_fits_metadata(pipeline_meta.B_fits_id)
        else:
            metaB, best_gen = load_best_metadata(A_meta_id=metaA.id)
            log.append(
                f"B : Loaded best generated dataset: {metaB.id}, which was built using gen params {best_gen.id}"
            )

        logB = load_fits_and_process(metaB, pp, "B", N=N)
        log.extend(logB)
        # B_gen = load_gen_param(pipeline_meta.gen_id)
    else:
        assert (
            args.meta_a is not None
        ), "You must provide -a (meta-file param) or a pipeline. Yry -a default or -p default"
        meta_id = args.meta_a
        metaA = load_fits_metadata(meta_id)

        assert (
            args.pp is not None
        ), "You must provide -pp (processing param), try -pp default"
        pp_id = args.pp
        pp = load_processing_param(pp_id)

        if args.meta_b:
            meta_id = f"meta_files/meta_{args.meta_b}.json"
            metaB = load_fits_metadata(meta_id)
        else:
            log.append(
                "B : User did not provide B file, try to find best synthetic dataset"
            )
            using_best = True
            metaB, best_gen = load_best_metadata(A_meta_id=metaA.id)
            log.append(
                f"Loaded best generated dataset: {metaB.id}, which was built using gen params {best_gen.id}"
            )

        try:
            logA = load_fits_and_process(metaA, pp, "A", N=N)
            log.extend(logA)
            log.append(f"A : Loaded {metaA.id}")
            log.append(
                f"Plots have been generated for A and can be found in plots folder. A : /plots/{metaA.id}, B"
            )
        except Exception as e:
            log.append(f"A : Exception occured when processing A . Full error: {e}")
        try:
            logB = load_fits_and_process(metaB, pp, "B", N=N)
            log.extend(logB)
            log.append(f"B : Loaded {metaB.id}")
            log.append(
                f"Plots have been generated for B and can be found in plots folder. A : /plots/{metaA.id}, B"
            )
        except Exception as e:
            log.append(f"B : Exception occured when processing B. Full error: {e}")

        if using_best:
            log.append(
                f"B : Selecting with best scored synth-dataset set B - id:{metaB.id}"
            )
        else:
            log.append(f"B : User selected dataset B id :{metaB.id}")

    print("LOG : ")
    for logline in log:
        print(logline)

    print("\n".join(log))


if __name__ == "__main__":
    main()
