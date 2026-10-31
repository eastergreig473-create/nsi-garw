"""Create verified temporary SSD copies; preserve every canonical F: cache."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("F:/NSI_GARW_Data/cache/attributes/hpatches_ncc")
DESTINATION = ROOT / ".cache/attribute_hot/hpatches_ncc"


def main():
    records = []
    files = sorted(SOURCE.rglob("*.npz"))
    for index, source in enumerate(files):
        destination = DESTINATION / source.relative_to(SOURCE)
        source_stat = source.stat()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".copying")
        digest = hashlib.sha256()
        with source.open("rb") as first, temporary.open("wb") as second:
            for chunk in iter(lambda: first.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
                second.write(chunk)
        copied = hashlib.sha256()
        with temporary.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                copied.update(chunk)
        if digest.digest() != copied.digest():
            raise ValueError(f"SSD cache copy mismatch: {source}")
        if source.stat().st_mtime_ns != source_stat.st_mtime_ns:
            raise ValueError(f"Source cache changed during copy: {source}")
        shutil.copystat(source, temporary)
        temporary.replace(destination)
        records.append({"source": str(source), "ssd_copy": str(destination), "sha256": digest.hexdigest(), "bytes": source_stat.st_size})
        if (index + 1) % 50 == 0:
            print(f"Verified SSD cache {index+1}/{len(files)}", flush=True)
    (DESTINATION / "copy_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"All {len(records)} attribute caches copied and byte-verified", flush=True)


if __name__ == "__main__":
    main()
