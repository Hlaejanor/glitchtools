import os
import sys
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import tarfile
from common.fitsmetadata import FitsMetadata
from pathlib import Path
from astropy.io import fits
from common.metadatahandler import (
    save_fits_metadata,
)


def build_metadata(obs_dir: str, starname: str = ""):
    """Generate metadata for a Chandra observation directory."""
    primary = Path(obs_dir) / "primary"
    evt_files = list(primary.glob("*_evt2.fits"))
    if not evt_files:
        raise FileNotFoundError(f"No *_evt2.fits file found in {primary}")

    evt_file = evt_files[0]

    with fits.open(evt_file) as hdul:
        hdr = hdul[0].header
        aik = hdr.get("OBJECT", "Unknown")
        star = f"{starname} {aik}"
        events = hdul[1].data
        # Extract key quantities safely
        obs_id = int(hdr.get("OBS_ID", -1))
        t_min = float(events["TIME"].min())
        t_max = float(events["TIME"].max())
        ra, dec = hdr.get("RA_TARG", 0.0), hdr.get("DEC_TARG", 0.0)
        e_min, e_max = float(events["ENERGY"].min()), float(events["ENERGY"].max())
        count = len(events)

    meta = FitsMetadata(
        id=obs_id,
        raw_event_file=str(evt_file.relative_to(obs_dir)),
        synthetic=True,
        source_pos_x=ra,
        source_pos_y=dec,
        max_energy=e_max,
        min_energy=e_min,
        source_count=count,
        star=star,
        t_min=t_min,
        t_max=t_max,
    )

    return meta


def download_and_extract(
    url, observation_id: str, dest_folder="./fits", starname: str = ""
):
    os.makedirs(dest_folder, exist_ok=True)

    print(f"Fetching file list from {url} ...")
    response = requests.get(url)
    if not response.ok:
        raise Exception(f"Failed to fetch URL: {url}")

    soup = BeautifulSoup(response.text, "html.parser")
    links = [a["href"] for a in soup.find_all("a") if a["href"] not in ("../",)]

    for link in links:
        file_url = urljoin(url, link)
        dest_path = os.path.join(dest_folder, os.path.basename(link))

        print(f"Downloading {file_url} ...")
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        if dest_path.endswith((".tar", ".tar.gz", ".tgz")):
            print(f"Extracting {dest_path} ...")
            try:
                with tarfile.open(dest_path) as tar:
                    tar.extractall(path=dest_folder)
            except tarfile.TarError as e:
                print(f"Warning: Failed to extract {dest_path}: {e}")

    meta = build_metadata(dest_folder, starname=starname)
    meta = save_fits_metadata(meta)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python download_chandra_data.py <URL>")
        sys.exit(1)

    input_url = sys.argv[1]
    download_and_extract(input_url)
