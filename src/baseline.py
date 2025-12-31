import argparse
import numpy as np

from create_pipeline import run_pipeline

from var_analysis import compare_two_sets
from common.helper import (
    write_result_to_csv,
)
from common.metadatahandler import (
    load_fits_metadata,
    load_processing_param,
    load_pipeline,
)
from event_processing.binning import (
    load_source_data,
)

version = 1.0
"""
TL;DR : Script builds statistical_tests.csv - the dataset that determines the empirical q value for the p1a prediciton

Explanation:
The purpose of this script is to run multiple (specified by --runs) independent versions of the same pipeline,
applying all transformations from the source and storing the results in the statistical_tests.csv file
The reason for this is prediction 1A, the tail exceedance test.
The exceedance test needs to determine where the 0.999th percentile begins, the point in the control dataset where 0.001 of the observations are higher
However, due to monte-carlo sampling, this depends on randomness and so it can vary quite a lot between runs. 
To ensure that this q value represents the mean, we need to compute the empirical_q as the mean result of many runs
To complicate things, variability types have different sensitivities so this empirical q varies with variability type
The simplest way to do this is to compute the q threshold for each run, write to a table and then use that table as the data
for deciding on the empirical q. This script builds that baseline
"""


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Perform variability analysis on event data using chunk phased sampling."
    )
    parser.add_argument(
        "--pipe",
        type=str,
        default="antares_vs_rnd_antares",
        help="Pipeline ID to run. Use 'all' to run all pipelines.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default="3",
        help="How many runs",
    )

    args = parser.parse_args()
    return args


def main():
    log = []
    print(
        "Baseline computation script. Will run CPV analysis many times and write to files/baseline_<pipe_id>.csv. To establieh threshold take the average of the 0.999 (or other cutoff) value"
    )
    args = parse_arguments()
    assert args.runs > 0, "Number of runs must be positive"

    print(f"Running baseline computation {version}")
    if args.pipe is not None:
        for i in range(0, args.runs):
            print(f" --- RUN {i+1} / {args.runs} ---")
            # Load and run a single pipeline
            pipe = load_pipeline(args.pipe)
            # Run the tasks in the pipeline, includes the randomization step.
            run_pipeline(pipe)
            meta_B = load_fits_metadata(pipe.B_fits_id)
            meta_A = load_fits_metadata(pipe.A_fits_id)

            pp = load_processing_param(pipe.pp_id)

            log_A, meta_A, source_A = load_source_data(meta_A, pp)
            log_B, meta_B, source_B = load_source_data(meta_B, pp)

            log, results = compare_two_sets(pipe, metaA=meta_A, metaB=meta_B, pp=pp)
            print(f"Write for pipeline {pipe.id} : {results}")
            write_result_to_csv(results, f"files/baseliner.csv")


if __name__ == "__main__":
    main()
