"""Step 0 - download the Montgomery and Shenzhen chest X-ray sets from the NLM.

Only needed if data/raw/ is empty (for example on a new computer).

Usage:
    python step0_download_data.py                  # both sets
    python step0_download_data.py --only montgomery

The images end up in:
    data/raw/MontgomerySet/CXR_png/        MCUCXR_0001_0.png ...  (138 images)
    data/raw/ChinaSet_AllFiles/CXR_png/    CHNCXR_0001_0.png ...  (662 images)

The last digit of each filename is the label: _0 = normal, _1 = TB.
Files that are already downloaded are skipped, so you can simply re-run the
script if the download stops part way. Total size is about 4 GB.
"""
import argparse
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets"
HERE = Path(__file__).parent

# (folder on the NLM server, local folder)
SOURCES = {
    "montgomery": [
        ("Montgomery-County-CXR-Set/MontgomerySet/CXR_png", "MontgomerySet/CXR_png"),
        ("Montgomery-County-CXR-Set/MontgomerySet/ClinicalReadings", "MontgomerySet/ClinicalReadings"),
    ],
    "shenzhen": [
        ("Shenzhen-Hospital-CXR-Set/CXR_png", "ChinaSet_AllFiles/CXR_png"),
        ("Shenzhen-Hospital-CXR-Set/ClinicalReadings", "ChinaSet_AllFiles/ClinicalReadings"),
    ],
}

# one row of the server's folder listing: file name and size in bytes
_ROW = re.compile(r"<a href='([^']+)'>[^<]*</a></td><td class='size'>(\d+|-)</td>")
_HEADERS = {"User-Agent": "python-urllib dataset downloader"}


def fetch(url, timeout=120):
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def list_files(remote_dir):
    """Return [(filename, size)] for the files in one server folder."""
    html = fetch(f"{BASE}/{remote_dir}/index.html").decode("utf-8", errors="replace")
    return [(name, int(size)) for name, size in _ROW.findall(html)
            if not name.endswith("/") and not name.endswith("index.html") and size != "-"]


def download(url, dest, size, retries=3):
    if dest.exists() and dest.stat().st_size == size:
        return "skip"
    for attempt in range(retries):
        try:
            data = fetch(url)
            if size and len(data) != size:
                raise IOError(f"size mismatch ({len(data)} != {size})")
            dest.write_bytes(data)
            return "ok"
        except Exception as exc:
            if attempt == retries - 1:
                return f"FAILED: {exc}"
    return "FAILED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(HERE / "data" / "raw"), help="where to put the datasets")
    ap.add_argument("--only", choices=sorted(SOURCES), help="download just one set")
    args = ap.parse_args()

    raw = Path(args.raw)
    failures = []
    for dataset in ([args.only] if args.only else list(SOURCES)):
        for remote, local in SOURCES[dataset]:
            dest_dir = raw / local
            dest_dir.mkdir(parents=True, exist_ok=True)
            files = list_files(remote)
            print(f"{dataset}: {local} - {len(files)} files")
            with ThreadPoolExecutor(max_workers=4) as pool:
                jobs = {pool.submit(download, f"{BASE}/{remote}/{name}", dest_dir / name, size): name
                        for name, size in files}
                for i, job in enumerate(as_completed(jobs), 1):
                    status = job.result()
                    if status.startswith("FAILED"):
                        failures.append(f"{local}/{jobs[job]}: {status}")
                    if i % 100 == 0 or i == len(files):
                        print(f"  {i}/{len(files)} done")

    if failures:
        print(f"\n{len(failures)} files failed - run the same command again to retry.")
        sys.exit(1)
    print("\nDone. Next: python step1_extract_features.py")


if __name__ == "__main__":
    main()
