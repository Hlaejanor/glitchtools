import os
import sys
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import tarfile


def download_and_extract(url, dest_folder="./fits"):
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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python download_chandra_data.py <URL>")
        sys.exit(1)

    input_url = sys.argv[1]
    download_and_extract(input_url)
