"""Phase 2 — download the NLM Montgomery and Shenzhen sets into data/raw.

Usage:
    python scripts/00_download_data.py                 # both sets
    python scripts/00_download_data.py --only montgomery
    python scripts/00_download_data.py --raw D:/tb_data/raw

The NLM server publishes the sets as browsable folders (no zip), so this walks
the folder listings and fetches the subfolders the pipeline needs, into the
layout the README describes:

    data/raw/MontgomerySet/{CXR_png, ClinicalReadings, ManualMask/leftMask, ManualMask/rightMask}
    data/raw/ChinaSet_AllFiles/{CXR_png, ClinicalReadings}

Files already on disk with the right size are skipped, so an interrupted
download can simply be re-run.
"""
import argparse
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets"

# (remote folder, local folder) per dataset
SOURCES = {
    "montgomery": [
        ("Montgomery-County-CXR-Set/MontgomerySet/CXR_png", "MontgomerySet/CXR_png"),
        ("Montgomery-County-CXR-Set/MontgomerySet/ClinicalReadings", "MontgomerySet/ClinicalReadings"),
        ("Montgomery-County-CXR-Set/MontgomerySet/ManualMask/leftMask", "MontgomerySet/ManualMask/leftMask"),
        ("Montgomery-County-CXR-Set/MontgomerySet/ManualMask/rightMask", "MontgomerySet/ManualMask/rightMask"),
    ],
    "shenzhen": [
        ("Shenzhen-Hospital-CXR-Set/CXR_png", "ChinaSet_AllFiles/CXR_png"),
        ("Shenzhen-Hospital-CXR-Set/ClinicalReadings", "ChinaSet_AllFiles/ClinicalReadings"),
    ],
}

_ROW = re.compile(r"<a href='([^']+)'>[^<]*</a></td><td class='size'>(\d+|-)</td>")
_HEADERS = {"User-Agent": "tbx-dataset-downloader/0.1 (python urllib)"}


def _get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def list_files(remote_dir: str) -> list[tuple[str, int]]:
    """Return [(filename, size_bytes)] for the files (not subfolders) in a listing."""
    html = _get(f"{BASE}/{remote_dir}/index.html").decode("utf-8", errors="replace")
    out = []
    for href, size in _ROW.findall(html):
        if href.endswith("/") or href.endswith("index.html") or size == "-":
            continue
        out.append((href, int(size)))
    return out


def _download(url: str, dest: Path, size: int, retries: int = 3) -> str:
    if dest.exists() and dest.stat().st_size == size:
        return "skip"
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            data = _get(url)
            if size and len(data) != size:
                raise IOError(f"size mismatch ({len(data)} != {size})")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return "ok"
        except Exception as exc:  # network hiccup: retry
            if attempt == retries:
                return f"FAILED: {exc}"
    return "FAILED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw", help="where to put the datasets")
    ap.add_argument("--only", choices=sorted(SOURCES), help="download just one set")
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads")
    args = ap.parse_args()

    raw = Path(args.raw)
    datasets = [args.only] if args.only else list(SOURCES)
    failures = []

    for ds in datasets:
        for remote, local in SOURCES[ds]:
            dest_dir = raw / local
            dest_dir.mkdir(parents=True, exist_ok=True)
            files = list_files(remote)
            total_mb = sum(s for _, s in files) / 1e6
            print(f"[download] {ds}: {remote.split('/')[-1]} — {len(files)} files, {total_mb:.0f} MB -> {dest_dir}")

            counts = {"ok": 0, "skip": 0}
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futs = {
                    pool.submit(_download, f"{BASE}/{remote}/{name}", dest_dir / name, size): name
                    for name, size in files
                }
                for i, fut in enumerate(as_completed(futs), 1):
                    status = fut.result()
                    if status in counts:
                        counts[status] += 1
                    else:
                        failures.append(f"{remote}/{futs[fut]}: {status}")
                    if i % 50 == 0 or i == len(files):
                        print(f"           {i}/{len(files)} done", flush=True)
            print(f"           downloaded {counts['ok']}, already had {counts['skip']}")

    if failures:
        print(f"\n[download] {len(failures)} files failed — re-run the same command to retry:")
        for f in failures[:20]:
            print("   ", f)
        sys.exit(1)
    print("\n[download] all done. Next: python scripts/01_prepare_data.py --config configs/default.yaml")


if __name__ == "__main__":
    main()
