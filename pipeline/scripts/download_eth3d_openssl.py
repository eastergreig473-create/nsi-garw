"""Official ETH3D range downloader using Python OpenSSL instead of Schannel."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

ROOT = Path("F:/NSI_GARW_Data/downloads/ETH3D_openssl")
FILES = {
    "multi_view_training_dslr_undistorted.7z": 5887406146,
    "multi_view_training_dslr_scan_eval.7z": 1920731473,
    "multi_view_training_dslr_occlusion.7z": 1267705580,
    "courtyard_dslr_depth.7z": 428289105, "delivery_area_dslr_depth.7z": 388144128,
    "electro_dslr_depth.7z": 655596830, "facade_dslr_depth.7z": 804421080,
    "kicker_dslr_depth.7z": 415014798, "meadow_dslr_depth.7z": 146736485,
    "office_dslr_depth.7z": 441352937, "playground_dslr_depth.7z": 283987129,
    "relief_dslr_depth.7z": 686900728, "relief_2_dslr_depth.7z": 646673138,
    "terrace_dslr_depth.7z": 376720366, "terrains_dslr_depth.7z": 529179696,
}


def _request(name: str, start: int, end: int):
    response = urlopen(Request("https://www.eth3d.net/data/" + name,
        headers={"Range": f"bytes={start}-{end}", "User-Agent": "NSI-GARW academic reproducibility downloader"}), timeout=30)
    expected = f"bytes {start}-{end}/{FILES[name]}"
    if response.status != 206 or response.headers.get("Content-Range") != expected:
        response.close()
        raise ValueError(f"Invalid range response: expected {expected}")
    return response


def seed_prefix(source: Path, length: int) -> None:
    """Copy only the verified sequential prefix, leaving the BITS file intact."""
    destination = ROOT / "multi_view_training_dslr_undistorted.7z.part"
    if destination.exists():
        return
    name = "multi_view_training_dslr_undistorted.7z"
    offsets = [0, 1048576, length // 4, length // 2, 3 * length // 4, length - 4096]
    with source.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            local = handle.read(4096)
            for attempt in range(10):
                try:
                    with _request(name, offset, offset + 4095) as response:
                        remote = response.read()
                    break
                except Exception:
                    if attempt == 9:
                        raise
                    time.sleep(2)
            if local != remote:
                raise ValueError(f"BITS prefix mismatch at {offset}; no prefix copied")
            print(f"BITS prefix verified at offset {offset}", flush=True)
    print(f"Verified BITS prefix at {len(offsets)} offsets; copying {length} bytes", flush=True)
    temporary = destination.with_suffix(".seeding")
    with source.open("rb") as source_handle, temporary.open("wb") as target_handle:
        remaining = length
        while remaining:
            chunk = source_handle.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise ValueError("Unexpected EOF in BITS prefix")
            target_handle.write(chunk)
            remaining -= len(chunk)
    temporary.replace(destination)


def download(name: str, total: int) -> dict:
    final, partial = ROOT / name, ROOT / (name + ".part")
    if final.is_file() and final.stat().st_size == total:
        return {"file": str(final), "state": "already_complete", "bytes": total}
    failures, started = 0, time.perf_counter()
    previous_report = 0.
    while not partial.exists() or partial.stat().st_size < total:
        start = partial.stat().st_size if partial.exists() else 0
        end = min(total - 1, start + 64 * 1024 * 1024 - 1)
        try:
            with _request(name, start, end) as response, partial.open("ab") as handle:
                received = 0
                while True:
                    chunk = response.read(min(1024 * 1024, end - start + 1 - received))
                    if not chunk:
                        break
                    handle.write(chunk)
                    received += len(chunk)
                    if received == end - start + 1:
                        break
                if received != end - start + 1:
                    raise IOError(f"Short range read {received}/{end-start+1}")
            failures = 0
        except Exception as error:
            failures += 1
            print(f"RETRY {name}: {type(error).__name__}: {error}", flush=True)
            if failures >= 30:
                raise RuntimeError(f"30 successive download failures for {name}") from error
            time.sleep(min(30, 2 * failures))
        if time.perf_counter() - previous_report >= 30:
            print(f"{name}: {partial.stat().st_size if partial.exists() else 0}/{total}", flush=True)
            previous_report = time.perf_counter()
    if partial.stat().st_size != total:
        raise ValueError(f"Unexpected final size for {name}")
    digest = hashlib.sha256()
    with partial.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    partial.replace(final)
    result = {"file": str(final), "source": "https://www.eth3d.net/data/" + name,
              "bytes": total, "sha256": digest.hexdigest(), "wall_seconds": time.perf_counter() - started,
              "state": "download_complete_archive_CRC_not_yet_checked"}
    final.with_suffix(final.suffix + ".download.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"COMPLETE {name}: SHA256 {digest.hexdigest()}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-bits-file", type=Path)
    parser.add_argument("--seed-prefix-bytes", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--distorted-calibration", action="store_true")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.distorted_calibration:
        name = "multi_view_training_dslr_jpg.7z"
        for attempt in range(30):
            try:
                with urlopen(Request("https://www.eth3d.net/data/" + name,
                    headers={"Range": "bytes=0-0"}), timeout=30) as response:
                    match = re.fullmatch(r"bytes 0-0/(\d+)", response.headers.get("Content-Range", ""))
                    if response.status != 206 or match is None:
                        raise ValueError("Official server did not confirm archive length")
                    total = int(match.group(1))
                break
            except Exception as error:
                if attempt == 29:
                    raise
                print(f"Retry calibration archive size probe: {error}", flush=True)
                time.sleep(3)
        FILES[name] = total
        print(f"Official distorted-image archive: {total} bytes", flush=True)
        download(name, total)
        return
    if args.seed_bits_file:
        if not args.seed_prefix_bytes:
            raise ValueError("--seed-prefix-bytes is required")
        seed_prefix(args.seed_bits_file, args.seed_prefix_bytes)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = [pool.submit(download, name, total) for name, total in FILES.items()]
        for task in as_completed(tasks):
            task.result()


if __name__ == "__main__":
    main()
