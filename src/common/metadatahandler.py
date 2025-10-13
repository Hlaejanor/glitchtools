import json
from common.fitsmetadata import (
    FitsMetadata,
    ChunkVariabilityMetadata,
    ComparePipeline,
    ProcessingParameters,
    GenerationParameters,
    Spectrum,
)
import pandas as pd
import os
import numpy as np
from pandas import DataFrame


def load_fits_metadata(id: str) -> FitsMetadata:
    """
    Read JSON data from a file and reconstruct a FitsMetadata object.
    """
    assert id is not None, "Cannot load Fits metadata for id None"
    filename = f"meta_files/fits/{id}.json"
    file_exists = os.path.isfile(filename)
    if not file_exists:
        return None
    print(f"Loading Fits file {filename}")
    with open(filename, "r") as f:
        data = json.load(f)

    # if "lucretius" not in data:
    #    data["lucretius"] = -1

    # if "offset" not in data:
    #    data["offset"] = None

    # if "g" not in data:
    #    data["g"] = None

    if "synthetic" not in data:
        data["synthetic"] = True

    # if "highPI" not in data:
    #    data["highPI"] = None

    # if "lowPI" not in data:
    #    data["lowPI"] = None

    if "alpha" not in data:
        data["alpha"] = 1.0

    if "ascore" not in data:
        data["ascore"] = None

    # if "synthetic_twin_event_file" not in data:
    #    data["synthetic_twin_event_file"] = ""
    if data["apparent_spectrum"] is not None:
        spec = data["apparent_spectrum"]
        spectrum = Spectrum(spec["A"], spec["lambda_0"], spec["sigma"], spec["C"])
    else:
        spectrum = None
    if "gen_id" not in data:
        data["gen_id"] = None

    metadata = FitsMetadata(
        id=data["id"],
        raw_event_file=data["raw_event_file"],
        # synthetic_twin_event_file=data["synthetic_twin_event_file"],
        synthetic=data["synthetic"],
        source_pos_x=data["source_pos_x"],
        source_pos_y=data["source_pos_y"],
        # resolution=data["resolution"],
        max_energy=data["max_energy"],
        min_energy=data["min_energy"],
        t_min=data["t_min"],
        gen_id=data["gen_id"],
        # wavelength_bins=data["wavelength_bins"],
        # time_bin_seconds=data["time_bin_seconds"],
        # cropped_count=data["cropped_count"],
        # source_radius=data["source_radius"],
        # source_filename=data["source_filename"],
        # background_filename=data["background_filename"],
        source_count=data["source_count"],
        # anullus_radius_inner=data["anullus_radius_inner"],
        # anullus_radius_outer=data["anullus_radius_outer"],
        ascore=data["ascore"],
        star=data["star"],
        t_max=data["t_max"],
        apparent_spectrum=spectrum,
    )

    return metadata


def load_chunk_metadata(id: str) -> ChunkVariabilityMetadata:
    """
    Read JSON data from a file and reconstruct a ChunkVariabilityMetadata object.
    """
    assert id is not None, "Cannot load ChunkVariability metadata for id None"
    filename = f"meta_files/chunk_variability/{id}.json"
    print(f"Loading ChunkVariability file {filename}")

    if not os.path.isfile(filename):
        print(f"File {filename} was not on disk!")
        return False
    with open(filename, "r") as f:
        data = json.load(f)

    metadata = ChunkVariabilityMetadata(
        id=data["id"],
        source_meta_id=data["source_meta_id"],
        pp_id=data["pp_id"],
        fits_meta_hash=data["fits_meta_hash"],
        pp_meta_hash=data["pp_meta_hash"],
    )

    return metadata


def save_pipeline(metadata: ComparePipeline) -> None:
    """
    Serialize the ComparePipeline object to JSON and write it to a file.
    """
    # Convert the dataclass to a dictionary
    path = f"meta_files/pipeline/{metadata.id}.json"
    # file_exists = os.path.isfile(csv_path)

    with open(path, "w") as f:
        # Write the dictionary in a human-readable (pretty-printed) way
        json.dump(metadata.dict(), f, indent=4)

    print(f"Saved Pipeline file: {path}")


def save_gen_param(metadata: GenerationParameters) -> None:
    """
    Serialize the GenerationParameters object to JSON and write it to a file.
    """
    # Convert the dataclass to a dictionary
    path = f"meta_files/generation/{metadata.id}.json"

    with open(path, "w") as f:
        # Write the dictionary in a human-readable (pretty-printed) way
        json.dump(metadata.dict(), f, indent=4)

    print(f"Saved genereation parameters meta : {path}")


def save_processing_metadata(metadata: ProcessingParameters) -> None:
    """
    Serialize the ProcessingParameters object to JSON and write it to a file.
    """
    # Convert the dataclass to a dictionary
    path = f"meta_files/processing/{metadata.id}.json"

    with open(path, "w") as f:
        # Write the dictionary in a human-readable (pretty-printed) way
        json.dump(metadata.dict(), f, indent=4)

    print(f"Saved Processing Parameters metadata : {path}")


def save_chunk_metadata(metadata: ChunkVariabilityMetadata) -> None:
    """
    Serialize the FitsMetadata object to JSON and write it to a file.
    """
    # Convert the dataclass to a dictionary

    # file_exists = os.path.isfile(csv_path)

    with open(metadata.get_metafile_path(), "w") as f:
        # Write the dictionary in a human-readable (pretty-printed) way
        json.dump(metadata.dict(), f, indent=4)
    print(f"Saved chunk-variability-metadata : {metadata.get_metafile_path()}")
    return True


def save_fits_metadata(meta: FitsMetadata) -> None:
    """
    Serialize the FitsMetadata object to JSON and write it to a file.
    """
    # Convert the dataclass to a dictionary
    path = f"meta_files/fits/{meta.id}.json"
    # file_exists = os.path.isfile(csv_path)
    if not meta.t_max:
        meta.t_max = 0
    else:
        meta.t_max = int(meta.t_max)

    with open(path, "w") as f:
        # Write the dictionary in a human-readable (pretty-printed) way
        json.dump(meta.dict(), f, indent=4)
    print(f"Saved fits-metadata : {path}")
    return True


def load_processing_param(id) -> ProcessingParameters:
    path = f"meta_files/processing/{id}.json"
    is_file = os.path.isfile(path)
    if is_file:
        with open(path, "r") as f:
            data = json.load(f)

            if "padding_strategy" not in data:
                data["padding_strategy"] = False
            if "downsample_strategy" not in data:
                data["downsample_strategy"] = None
            if "downsample_target_count" not in data:
                data["downsample_target_count"] = None

            if "time_bins_from" not in data:
                data["time_bins_from"] = None
            if "time_bins_to" not in data:
                data["time_bins_to"] = None
            if "time_bin_widths_count" not in data:
                data["time_bin_widths_count"] = 1
            if "time_bin_chunk_length" not in data:
                data["time_bin_chunk_length"] = 12
            if "take_top_variability_count" not in data:
                data["take_top_variability_count"] = 10
            if "phase_bins" not in data:
                data["phase_bins"] = 12

            metadata = ProcessingParameters(
                id=data["id"],
                source_radius=data["source_radius"],
                processed_filename=data["processed_filename"],
                wavelength_bins=data["wavelength_bins"],
                resolution=data["resolution"],
                time_bin_chunk_length=data["time_bin_chunk_length"],
                time_bin_widths_count=data["time_bin_widths_count"],
                time_bins_from=data["time_bins_from"],
                time_bins_to=data["time_bins_to"],
                max_wavelength=data["max_wavelength"],
                min_wavelength=data["min_wavelength"],
                time_bin_seconds=data["time_bin_seconds"],
                anullus_radius_inner=data["anullus_radius_inner"],
                anullus_radius_outer=data["anullus_radius_outer"],
                take_time_seconds=data["take_time_seconds"],
                padding_strategy=data["padding_strategy"],
                downsample_strategy=data["downsample_strategy"],
                downsample_target_count=data["downsample_target_count"],
                variability_type=data.get("variability_type", "neighbour"),
                percentile=data.get("percentile", None),
                chunk_counts=data.get("chunk_counts", 100000),
                take_top_variability_count=data["take_top_variability_count"],
                phase_bins=data["phase_bins"],
            )
        assert metadata.id == id, "What - the file and internal id has diverged"
        return metadata
    else:
        return None


def load_all_in_pipeline(id: str):
    filename = f"meta_files/pipeline/{id}.json"
    print(f"Loading Pipene file {filename}")
    with open(filename, "r") as f:
        data = json.load(f)
        pipe = ComparePipeline(
            id=id, A_fits_id=data["fits_id"], pp_id=data["pp_id"], gen_id=["gen_id"]
        )

    fitsmeta = load_fits_metadata(f"meta_files/fits/{pipe.A_fits_id}.json")
    processing_meta = load_processing_param("meta_files/processing/{id}.json")
    gen_meta = load_gen_param("meta_files/generation/{id}.json")

    return pipe, fitsmeta, processing_meta, gen_meta


def load_best_metadata(A_meta_id: str) -> tuple[FitsMetadata, GenerationParameters]:
    try:
        best_gen_param = None
        print("Loading experiment scores:")
        print("-- Try to find the experiment that most closely matched the data")
        summaries = load_summaries("temp/parameter_search.csv", A_meta_id)

        if len(summaries) == 0:
            print(
                "Oops. We must stop here, there are no valid synthetic datasets to explore for a match. Try running parameter_search.py"
            )
            exit(100)

        best = find_best_match(summaries)

        for i in range(0, min(100, len(best))):
            row = best.iloc[i]
            b_gen_id = row["B_gen_id"]
            genmeta_file = f"meta_files/generation/{b_gen_id}.json"
            print(genmeta_file)
            exists = os.path.isfile(genmeta_file)
            if exists:
                print(f"\n Best genemeta experiment found: {b_gen_id}")
                best_gen_param = load_gen_param(id=b_gen_id)
            else:
                print(
                    f"Wanted to fetch meta {b_gen_id} but the file {genmeta_file} was not found, moving on to next"
                )
                continue

            meta_file = f"meta_files/fits/{row['B_id']}.json"
            exists = os.path.isfile(meta_file)
            if exists:
                print(f"\n Best generated data found: {row['B_id']}")
                best_meta = load_fits_metadata(id=row["B_id"])
            else:
                print(
                    f"Wanted to fetch meta {row['B_id']} but the file {meta_file} was not found, moving on to next"
                )
                continue

            if best_gen_param is None:
                raise Exception(
                    "Could not find any best generation params to compare with. Have you deleted al the generation/[].json files?"
                )
            if best_meta is None:
                raise Exception(
                    "Could not find any best metadata to compare with. Have you deleted al the fits/meta_[].json files?"
                )

            return best_meta, best_gen_param

    except Exception as e:
        print(
            f"Could not find any best experiment. Try running parametersearch to generate more",
            e,
        )


def load_pipeline(id) -> ComparePipeline:
    print("Current working directory:", os.getcwd())

    filename = f"{os.getcwd()}/meta_files/pipeline/{id}.json"
    is_file = os.path.isfile(filename)

    if is_file:
        with open(filename, "r") as f:
            data = json.load(f)
            if "downsample_str" not in data:
                data["downsample_str"] = None

            metadata = ComparePipeline(
                id=data["id"],
                A_fits_id=data["A_fits_id"],
                B_fits_id=data["B_fits_id"],
                pp_id=data["pp_id"],
                gen_id=data["gen_id"],
            )
        return metadata
    else:
        filename = f"../meta_files/pipeline/{id}.json"
        return None


def load_gen_param(id) -> GenerationParameters:
    path = f"meta_files/generation/{id}.json"
    is_file = os.path.isfile(path)
    if is_file:
        with open(path, "r") as f:
            data = json.load(f)

            if "spectrum" in data and data["spectrum"] is not None:
                spec = data["spectrum"]
                spectrum = Spectrum(
                    spec["A"], spec["lambda_0"], spec["sigma"], spec["C"]
                )
            else:
                spectrum = Spectrum(0, 0, 0, 0)

            # TODO A hack to deal with some older files, can be removed
            if "raw_event_tile" in data:
                data["raw_event_file"] = data["raw_event_tile"]
            genmeta = GenerationParameters(
                id=data["id"],
                alpha=data["alpha"],
                lucretius=data["lucretius"],
                theta_change_per_sec=data.get("theta_change_per_sec", 0.0),
                r_e=data["r_e"],
                theta=data["theta"],
                t_min=data["t_min"],
                t_max=data["t_max"],
                perp=data["perp"],
                phase=data["phase"],
                raw_event_file=data["raw_event_file"],
                star=data["star"],
                max_wavelength=data["max_wavelength"],
                min_wavelength=data["min_wavelength"],
                spectrum=spectrum,
            )
        return genmeta
    return None


def load_summaries(path, target_meta_id: str = None):
    try:
        print(f"Loading summaries filtered on meta_id {target_meta_id}")
        if not os.path.isfile(path):
            return pd.DataFrame()
        df = pd.read_csv(path)
        if target_meta_id is not None:
            mask = df["A_id"] == target_meta_id
            return df[mask]
        else:
            return df

    except FileNotFoundError:
        print("Summary file not found.")
        return pd.DataFrame()


def find_best_match(df, metric="Combined MSE") -> DataFrame:
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in summaries.")
    return df.sort_values(by=metric)


def theta_error_modulo_symmetry(theta1, theta2):
    """
    Computes the minimal angular difference between theta1 and theta2
    modulo hexagonal symmetry (60 degrees or pi/3 radians).
    """
    diff = abs(theta1 - theta2) % (np.pi / 3)
    return min(diff, (np.pi / 3) - diff)


def compare_generation_params(g1, g2, label1="Reference", label2="Recovered"):
    print(f"{'Parameter':<15} | {label1:<12} | {label2:<12} | Δ (Abs Diff)")
    print("-" * 58)

    def row(name, val1, val2):
        diff = abs(val1 - val2)
        print(f"{name:<15} | {val1:<12.5f} | {val2:<12.5f} | {diff:<10.5f}")

    row("alpha", g1.alpha, g2.alpha)
    row("lucretius", g1.lucretius, g2.lucretius)
    row("r_e", g1.r_e, g2.r_e)
    row("theta", g1.theta, g2.theta)
    row("perp", g1.perp, g2.perp)
    row("phase", g1.phase, g2.phase)
    row("min_lambda", g1.min_wavelength, g2.min_wavelength)
    row("max_lambda", g1.max_wavelength, g2.max_wavelength)

    print(f"Theta mod error | {theta_error_modulo_symmetry(g1.theta, g2.theta)}")
    print("-" * 58)
    print(f"{'star':<15} | {g1.star:<12} | {g2.star:<12}")
    print(f"{'t_max':<15} | {g1.t_max:<12} | {g2.t_max:<12}")
