"""After complete archive verification, freeze scene splits and an outcome-blind pilot cohort."""
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import time

import h5py
import numpy as np
from PIL import Image

DATA = Path("F:/NSI_GARW_Data")
AUDIT = DATA / "manifests/megadepth"
SEED = 20261023


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-archive", action="store_true")
    args = parser.parse_args()
    AUDIT.mkdir(parents=True, exist_ok=True)
    protocol_path = AUDIT / "cohort_preparation_protocol_v2.json"
    protocol = {"status":"frozen before any MegaDepth feature/PCA/NSI outcome; exploratory reconstruction after HPatches/ETH3D outcomes",
        "seed":SEED, "scene_universe":"all195 numeric scene directories actually present in the completely verified official v1 archive; original project describes196, but no missing scene is invented",
        "amendment":"Before any image eligibility, features, PCA or NSI outcome: complete archive bytes213461598257/SHA11c556e68fab76639b01ca9e9f776aa77394451571b9528be1b85391b06b1ff5/CRC contain195 numeric directories, notthe196 assumed inv1. Preservev1; useactual complete archive universe andunchangedrounding/seed/exclusionrules.",
        "split":"SHA256(seed:scene:scene_id) order; round(0.8N) training, round(0.1N) validation, remainder test; fixed before eligibility exclusions",
        "sampling":"at most16 candidate depth/image pairs per scene ordered by SHA256(seed:image:relative_depth_path); no replacement after exclusion",
        "eligibility":"depth2D; no negative values (ordinal masks excluded); at least1% finitepositive depth; image/depth shape identical; image exists; no model scores involved",
        "geometry":"camera intrinsics absent; this cohort does not authorize calibrated-normal NSI or semantic labeling of PCA directions",
        "runner_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8"))["protocol"] != protocol:
            raise ValueError("A different frozen MegaDepth cohort protocol already exists")
    else:
        protocol_path.write_text(json.dumps({"frozen_at_utc":datetime.now(timezone.utc).isoformat(),"protocol":protocol},indent=2),encoding="utf-8")
    marker = AUDIT / "archive_complete.json"
    if not marker.exists() and not args.wait_for_archive:
        raise FileNotFoundError("Complete archive SHA/CRC marker absent")
    deadline = time.monotonic()+24*3600
    if not marker.exists():
        print("Waiting for complete MegaDepth archive SHA/CRC; no partial cohort will be analyzed",flush=True)
    while not marker.exists():
        if time.monotonic()>deadline:
            raise TimeoutError("MegaDepth verification did not finish within24h")
        time.sleep(30)
    complete = json.loads(marker.read_text(encoding="utf-8"))
    if complete["status"] != "PASS" or complete["bytes"] != 213461598257:
        raise ValueError("Archive verification marker is not valid")
    root = Path(complete["root"]).resolve()
    scenes = sorted(s for s in complete["scenes"] if re.fullmatch(r"\d{4}",s))
    if len(scenes) != 195:
        raise ValueError(f"Verified archive scene universe changed: expected195, found{len(scenes)}")
    ordered = sorted(scenes,key=lambda scene:digest(f"{SEED}:scene:{scene}"))
    n_train,n_val = round(.8*len(ordered)),round(.1*len(ordered))
    split = {scene:("train" if i<n_train else "validation" if i<n_train+n_val else "test") for i,scene in enumerate(ordered)}
    split_payload = {"seed":SEED,"scene_to_split":split,"counts":{"train":n_train,"validation":n_val,"test":len(ordered)-n_train-n_val},
        "archive_sha256":complete["archive_sha256"],"protocol_sha256":hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "created_at_utc":datetime.now(timezone.utc).isoformat(),"eligibility_exclusions_do_not_reshuffle":True}
    split_path = AUDIT / "frozen_scene_split.json"
    if split_path.exists():
        if json.loads(split_path.read_text(encoding="utf-8"))["scene_to_split"] != split:
            raise ValueError("Refusing to replace a different frozen scene split")
    else:
        split_path.write_text(json.dumps(split_payload,indent=2),encoding="utf-8")
    candidates = {scene:[] for scene in scenes}
    image_index = {}
    with Path(complete["member_index"]).open(encoding="utf-8") as stream:
        for line in stream:
            item=json.loads(line)
            path=Path(item["path"])
            relative=path.relative_to(root).as_posix()
            if relative.split("/")[0] not in candidates:
                continue
            if path.suffix.lower()==".h5" and path.parent.name=="depths":
                candidates[relative.split("/")[0]].append(item)
            elif path.parent.name=="imgs":
                image_index[(path.parent.parent.as_posix(),path.stem)]=item
    records, excluded = [],[]
    for scene in scenes:
        selected=sorted(candidates[scene],key=lambda item:digest(f"{SEED}:image:{Path(item['path']).relative_to(root).as_posix()}"))[:16]
        for item in selected:
            path=Path(item["path"])
            image=image_index.get((path.parent.parent.as_posix(),path.stem))
            record={"scene":scene,"split":split[scene],"depth":str(path),"depth_sha256":item["sha256"]}
            if image is None:
                excluded.append({**record,"reason":"missing_image"}); continue
            with h5py.File(path,"r") as handle:
                depth=np.asarray(handle["depth"],dtype=np.float32)
            reason=None
            if depth.ndim!=2:
                reason="non2D_depth"
            elif np.any(depth<0):
                reason="ordinal_mask_negative_values"
            elif float(np.mean(np.isfinite(depth)&(depth>0)))<.01:
                reason="less_than1percent_positive_finite_depth"
            else:
                with Image.open(image["path"]) as picture:
                    width,height=picture.size
                if depth.shape!=(height,width):
                    reason="image_depth_shape_mismatch"
            if reason:
                excluded.append({**record,"reason":reason}); continue
            records.append({**record,"image":image["path"],"image_sha256":image["sha256"],"height":height,"width":width})
        print(f"MegaDepth cohort {scene}: {sum(r['scene']==scene for r in records)}/{len(selected)} eligible",flush=True)
    if not records:
        raise ValueError("No eligible metric-depth images in the outcome-blind cohort")
    output=AUDIT / "pilot_image_cohort.csv"
    with output.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    payload={"status":"PASS","created_at_utc":datetime.now(timezone.utc).isoformat(),"eligible_images":len(records),
        "eligible_scenes":len(set(r["scene"] for r in records)),"eligible_by_split":{s:sum(r["split"]==s for r in records) for s in ("train","validation","test")},
        "excluded":excluded,"cohort_csv":str(output),"cohort_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
        "protocol":protocol,"archive_sha256":complete["archive_sha256"],"calibrated_normals_status":"not available; intrinsics unresolved"}
    (AUDIT / "cohort_complete.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in payload.items() if k not in ("excluded","protocol")},indent=2),flush=True)


if __name__=="__main__":
    main()
