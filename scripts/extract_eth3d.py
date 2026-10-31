"""Validate ETH3D archive members and record libarchive extraction/CRC status."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import locale
from pathlib import Path, PurePosixPath
import subprocess
import time


DOWNLOADS = Path("F:/NSI_GARW_Data/downloads/ETH3D_openssl")
OUTPUT = Path("F:/NSI_GARW_Data/datasets/ETH3D/training")
AUDIT = Path("F:/NSI_GARW_Data/manifests/eth3d/archive_checks")
ENCODING = locale.getpreferredencoding(False)


def extract(path: Path) -> None:
    marker = AUDIT / (path.name + ".json")
    if marker.exists():
        return
    root = OUTPUT.resolve()
    listing = subprocess.run(["tar", "-tf", str(path)], capture_output=True,
                             text=True, encoding=ENCODING, check=True)
    members = listing.stdout.splitlines()
    if not members:
        raise ValueError(f"Empty archive: {path}")
    for name in members:
        relative = PurePosixPath(name.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts or ":" in name:
            raise ValueError(f"Unsafe archive member: {name!r}")
        (root / Path(*relative.parts)).resolve().relative_to(root)
    verbose = subprocess.run(["tar", "-tvf", str(path)], capture_output=True,
                             text=True, encoding=ENCODING, check=True)
    if any(line and line[0] not in {"-", "d"} for line in verbose.stdout.splitlines()):
        raise ValueError(f"Archive includes links or special files: {path}")
    root.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {path.name}: {len(members)} safe members", flush=True)
    # libarchive verifies compressed streams and file CRCs during extraction.
    completed = subprocess.run(["tar", "-xf", str(path), "-C", str(root)],
                               capture_output=True, text=True, encoding=ENCODING)
    if completed.returncode or completed.stderr.strip():
        raise RuntimeError(f"Archive extraction did not pass cleanly: {path}: {completed.stderr}")
    missing = [name for name in members if not (root / name).exists()]
    if missing:
        raise ValueError(f"Missing extracted members: {missing[:5]}")
    download_record = path.with_name(path.name + ".download.json")
    digest = json.loads(download_record.read_text(encoding="utf-8"))["sha256"] if download_record.exists() else None
    if digest is None:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
    AUDIT.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"archive": str(path), "sha256": digest,
        "bytes": path.stat().st_size, "member_count": len(members), "output_root": str(root),
        "integrity": "libarchive extraction returned 0 with empty stderr; all member paths exist",
        "checked_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    print(f"Verified extraction: {path.name}", flush=True)


def main():
    global OUTPUT, AUDIT
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    OUTPUT = args.output.resolve()
    AUDIT = AUDIT / hashlib.sha256(str(OUTPUT).encode()).hexdigest()[:12]
    while True:
        paths = list(DOWNLOADS.glob("*.7z"))
        paths += list(Path("F:/NSI_GARW_Data/downloads/ETH3D").glob("pipes_dslr_depth.7z"))
        for path in sorted(paths, key=lambda p: p.stat().st_size):
            extract(path)
        if not args.watch:
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
