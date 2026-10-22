"""Download NHTSA complaint flat files, skipping any the server reports unchanged.

NHTSA regenerates the files daily, so a rerun on a later day usually fetches
everything again. The manifest records which snapshot is on disk.
"""

import json
import shutil
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from nhtsa.paths import RAW_DIR

BASE_URL = "https://static.nhtsa.gov/odi/ffdd/cmpl"
FILES = ("FLAT_CMPL.zip", "CMPL.txt")
DEFAULT_DATA_DIR = RAW_DIR
MANIFEST_NAME = "manifest.json"


def download_complaints(
    data_dir: Path = DEFAULT_DATA_DIR,
    force: bool = False,
    base_url: str = BASE_URL,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / MANIFEST_NAME
    manifest = _load_manifest(manifest_path)

    for name in FILES:
        dest = data_dir / name
        cached = manifest.get(name) if dest.exists() and not force else None
        last_modified = cached["last_modified"] if cached else None
        entry = _fetch(f"{base_url}/{name}", dest, last_modified)
        if entry is None:
            print(
                f"{name}: unchanged (last modified {manifest[name]['last_modified']})"
            )
        else:
            manifest[name] = entry
            _save_manifest(manifest_path, manifest)
            size_mb = dest.stat().st_size / 1e6
            print(
                f"{name}: downloaded {size_mb:.1f} MB (last modified {entry['last_modified']})"
            )
        if dest.suffix == ".zip":
            _extract(dest, data_dir, overwrite=entry is not None)


def _fetch(url: str, dest: Path, last_modified: str | None) -> dict | None:
    """Download url to dest, or return None if unchanged since last_modified."""
    request = urllib.request.Request(url)
    # NHTSA's server ignores If-None-Match (ETags) but honors If-Modified-Since.
    if last_modified:
        request.add_header("If-Modified-Since", last_modified)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code == 304:
            return None
        raise

    # Write to a temp file first so an interrupted download never replaces a good file.
    part = dest.with_name(dest.name + ".part")
    try:
        with response, part.open("wb") as f:
            shutil.copyfileobj(response, f, length=1 << 20)
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)

    return {
        "url": url,
        "etag": response.headers.get("ETag"),
        "last_modified": response.headers.get("Last-Modified"),
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _extract(archive: Path, data_dir: Path, overwrite: bool) -> None:
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            if overwrite or not (data_dir / member).exists():
                zf.extract(member, data_dir)
                print(f"  extracted {member}")


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _save_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n")
