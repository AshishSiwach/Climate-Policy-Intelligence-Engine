"""
Download the CPIE corpus PDFs to data/raw/.

Reads canonical URLs from configs/corpus_sources.yaml. Skips files that
already exist. Verifies SHA-256 checksums when recorded in the YAML.

Usage:
    uv run python scripts/download_data.py               # download missing files
    uv run python scripts/download_data.py --force       # re-download all
    uv run python scripts/download_data.py --check       # verify checksums only

    make data                                             # shorthand

Notes:
  - The IEA WEO 2025 requires free registration and cannot be auto-downloaded.
    The script will print instructions and skip it.
  - PDFs are gitignored. Never commit them.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
SOURCES_YAML = REPO_ROOT / "configs" / "corpus_sources.yaml"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    """Download url → dest with a basic progress indicator."""
    print(f"  Downloading {dest.name} ...", end=" ", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "cpie-data-downloader/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
        while chunk := resp.read(65536):
            out.write(chunk)
    size_mb = dest.stat().st_size / 1_048_576
    print(f"done ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CPIE corpus PDFs")
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist")
    parser.add_argument("--check", action="store_true", help="Verify checksums only, no downloading")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with open(SOURCES_YAML) as f:
        config = yaml.safe_load(f)

    docs = config.get("documents", [])
    skipped_manual = []
    errors = []

    for doc in docs:
        filename = doc["filename"]
        url = doc.get("url")
        expected_sha = doc.get("sha256")
        dest = RAW_DIR / filename

        # Documents that require manual download (e.g. IEA registration wall)
        if url is None:
            skipped_manual.append(doc)
            if not dest.exists():
                print(f"  [MANUAL] {filename}")
                print(f"           {doc.get('doc_id')} requires manual download.")
                print("           Obtain from the institution's website and place in data/raw/")
            else:
                print(f"  [OK]     {filename} (manual — already present)")
            continue

        if args.check:
            if not dest.exists():
                print(f"  [MISSING] {filename}")
                errors.append(filename)
            elif expected_sha and _sha256(dest) != expected_sha:
                print(f"  [MISMATCH] {filename} — checksum does not match")
                errors.append(filename)
            else:
                print(f"  [OK]      {filename}")
            continue

        if dest.exists() and not args.force:
            # Verify checksum if available, warn on mismatch but don't re-download
            if expected_sha:
                actual = _sha256(dest)
                if actual != expected_sha:
                    print(f"  [WARN] {filename} exists but checksum mismatch — rerun with --force to re-download")
                else:
                    print(f"  [OK]   {filename} (already present, checksum verified)")
            else:
                print(f"  [OK]   {filename} (already present)")
            continue

        try:
            _download(url, dest)
            actual_sha = _sha256(dest)
            if expected_sha and actual_sha != expected_sha:
                print(f"  [WARN] Checksum mismatch for {filename}")
                print(f"         expected: {expected_sha}")
                print(f"         got:      {actual_sha}")
            elif not expected_sha:
                print(f"  SHA-256: {actual_sha}  (add to corpus_sources.yaml to enable future verification)")
        except urllib.error.URLError as e:
            print(f"  [FAIL] {filename}: {e}")
            errors.append(filename)
            if dest.exists():
                dest.unlink()  # remove partial download

    print()
    if skipped_manual:
        print(f"  {len(skipped_manual)} document(s) require manual download (see above).")
    if errors:
        print(f"  {len(errors)} error(s). Check URLs in configs/corpus_sources.yaml.")
        sys.exit(1)
    else:
        present = sum(1 for doc in docs if (RAW_DIR / doc["filename"]).exists())
        print(f"  {present}/{len(docs)} PDFs present in data/raw/")
        if present < len(docs):
            print("  Some files are missing — check the output above for details.")


if __name__ == "__main__":
    main()
