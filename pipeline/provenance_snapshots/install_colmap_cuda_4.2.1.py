"""Fetch the pinned official portable CUDA build; verify release digest and ZIP CRC."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

from nsi_garw.experiments import write_json
from nsi_garw.io import sha256_file


def main():
    base = Path("F:/NSI_GARW_Data")
    release_path = base / "manifests/colmap/release_4.2.1.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    asset = next(a for a in release["assets"] if a["name"] == "colmap-x64-windows-cuda.zip")
    expected_sha = asset["digest"].removeprefix("sha256:")
    archive = base / "downloads/COLMAP-4.2.1-windows-cuda.zip"
    if not archive.exists():
        temporary = archive.with_suffix(".downloading")
        request = urllib.request.Request(asset["browser_download_url"], headers={"User-Agent": "NSI-GARW-research"})
        digest, size, last = hashlib.sha256(), 0, 0
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            for chunk in iter(lambda: response.read(1024*1024), b""):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
                if size-last >= 20*1024*1024:
                    print(f"COLMAP download {size}/{asset['size']} bytes", flush=True)
                    last = size
        if size != asset["size"] or digest.hexdigest() != expected_sha:
            raise ValueError("Pinned COLMAP archive digest/size mismatch")
        temporary.replace(archive)
    if archive.stat().st_size != asset["size"] or sha256_file(archive) != expected_sha:
        raise ValueError("Existing COLMAP archive digest/size mismatch")
    destination = base / "tools/colmap-4.2.1-cuda"
    destination.mkdir(parents=True, exist_ok=True)
    entries = []
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("COLMAP ZIP CRC failed")
        for member in z.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Unsafe path in official ZIP")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            content = z.read(member)
            digest = hashlib.sha256(content).hexdigest()
            if target.exists():
                if sha256_file(target) != digest:
                    raise ValueError(f"Existing installed COLMAP member changed: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            entries.append({"path": str(target), "sha256": digest, "bytes": len(content)})
    write_json(base / "manifests/colmap/installation.json", {
        "status": "PASS", "release": release["tag_name"], "release_metadata_sha256": sha256_file(release_path),
        "source": asset["browser_download_url"], "archive": str(archive), "archive_sha256": expected_sha,
        "zip_crc": "PASS", "members": entries, "installer_sha256": sha256_file(Path(__file__))})
    print("Pinned COLMAP 4.2.1 CUDA archive/member/CRC verification PASS", flush=True)


if __name__ == "__main__":
    main()
