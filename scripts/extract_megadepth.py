"""Safely extract the official archive in one pass, hashing archive and members."""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time
import tarfile

DATA = Path("F:/NSI_GARW_Data")
ARCHIVE = DATA / "downloads/MegaDepth_v1.tar.gz"
OUTPUT = DATA / "datasets/MegaDepth/MegaDepth_v1"
AUDIT = DATA / "manifests/megadepth"
EXPECTED_SIZE = 213461598257


class HashingReader:
    def __init__(self, source):
        self.source, self.digest, self.count = source, hashlib.sha256(), 0

    def read(self, size=-1):
        data = self.source.read(size)
        self.digest.update(data)
        self.count += len(data)
        return data


def main():
    if ARCHIVE.stat().st_size != EXPECTED_SIZE:
        raise ValueError("The official archive length is incomplete")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    root = OUTPUT.resolve()
    started, last_report = time.monotonic(), 0.
    file_count, expanded_bytes, scenes = 0, 0, set()
    index_path = AUDIT / "archive_members.jsonl"
    with ARCHIVE.open("rb") as raw, index_path.open("w", encoding="utf-8") as index:
        reader = HashingReader(raw)
        with gzip.GzipFile(fileobj=reader, mode="rb") as decoded:
            with tarfile.open(fileobj=decoded, mode="r|") as archive:
                for member in archive:
                    parts = PurePosixPath(member.name).parts
                    if PurePosixPath(member.name).is_absolute() or ".." in parts or "\\" in member.name:
                        raise ValueError(f"Unsafe member: {member.name}")
                    if parts[:4] != ("phoenix", "S6", "zl548", "MegaDepth_v1"):
                        raise ValueError(f"Unexpected archive prefix: {member.name}")
                    relative = parts[4:]
                    target = root.joinpath(*relative).resolve()
                    if not target.is_relative_to(root):
                        raise ValueError(f"Escaping member: {member.name}")
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    if not member.isfile() or not relative:
                        raise ValueError(f"Unsupported link or special member: {member.name}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + f".writing-{os.getpid()}")
                    digest, length = hashlib.sha256(), 0
                    source = archive.extractfile(member)
                    if source is None:
                        raise ValueError(f"Cannot read member: {member.name}")
                    with source, temporary.open("wb") as destination:
                        while block := source.read(1024*1024):
                            digest.update(block)
                            length += len(block)
                            destination.write(block)
                    if length != member.size:
                        raise ValueError(f"Truncated member: {member.name}")
                    if target.exists():
                        # Preserve a pre-existing file unless byte equality is established.
                        existing = hashlib.sha256()
                        with target.open("rb") as original:
                            while block := original.read(1024*1024):
                                existing.update(block)
                        if existing.hexdigest() != digest.hexdigest():
                            raise ValueError(f"Refusing to replace a differing existing file: {target}")
                        temporary.unlink()
                    else:
                        temporary.replace(target)
                    file_count += 1
                    expanded_bytes += length
                    scenes.add(relative[0])
                    index.write(json.dumps({"path": str(target), "archive_member": member.name,
                        "bytes": length, "sha256": digest.hexdigest()}) + "\n")
                    if time.monotonic() - last_report >= 30:
                        index.flush()
                        print(f"MegaDepth extraction: {reader.count:,}/{EXPECTED_SIZE:,} archive bytes; "
                              f"{file_count:,} files; {len(scenes)} scenes", flush=True)
                        last_report = time.monotonic()
            # Consume gzip EOF/footer to validate CRC and hash the entire source.
            while decoded.read(1024*1024):
                pass
        while reader.read(1024*1024):
            pass
        if reader.count != EXPECTED_SIZE:
            raise ValueError("Archive hashing did not consume the full source")
        report = {"status": "PASS", "archive": str(ARCHIVE), "bytes": reader.count,
            "archive_sha256": reader.digest.hexdigest(), "root": str(root), "file_count": file_count,
            "expanded_bytes": expanded_bytes, "scenes": sorted(scenes),
            "gzip_crc": "validated through EOF", "member_index": str(index_path),
            "wall_seconds": time.monotonic()-started, "completed_at_utc": datetime.now(timezone.utc).isoformat()}
    (AUDIT / "archive_complete.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
