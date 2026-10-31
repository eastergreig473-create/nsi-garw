"""Paired laser/COLMAP NSI on identical features and a common MVS pixel grid."""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.stats import kendalltau,spearmanr

from nsi_garw.config import load_config
from nsi_garw.eth3d import read_cameras,read_images
from nsi_garw.experiments import compute_nsi_artifacts,write_json,source_sha256
from nsi_garw.feature_builders import _load_local_features,_deterministic_subset,_atomic_savez
from nsi_garw.io import FeatureDataset,load_feature_npz,sha256_file
from nsi_garw.statistics import paired_group_bootstrap,benjamini_hochberg
from run_eth3d_suite import sampled_normals
from run_hpatches_suite import DATA,MODELS,RESULTS,ROOT

FOLDER=RESULTS/"eth3d_dense_sensitivity"
AUDIT=DATA/"manifests/eth3d/colmap_dense"
EXPORTS=DATA/"manifests/features/eth3d_dense_sensitivity"


def read_dense(path):
    with Path(path).open("rb") as stream:
        dimensions=[]
        for _ in range(3):
            token=bytearray()
            while True:
                value=stream.read(1)
                if not value:raise ValueError("Truncated COLMAP map header")
                if value==b"&":break
                token.extend(value)
                if len(token)>20:raise ValueError("Malformed COLMAP map header")
            dimensions.append(int(token))
        width,height,channels=dimensions
        if min(dimensions)<1:raise ValueError("Invalid COLMAP map dimensions")
        values=np.fromfile(stream,dtype="<f4")
    if len(values)!=width*height*channels:raise ValueError("COLMAP map byte count mismatch")
    return values.reshape((width,height,channels),order="F").transpose(1,0,2).squeeze()


def common_grid_laser(depth,shape):
    """Nearest inverse pixel-center resize; preserve unknown laser samples."""
    h,w=shape
    y=np.floor((np.arange(h)+.5)*depth.shape[0]/h).astype(np.int64)
    x=np.floor((np.arange(w)+.5)*depth.shape[1]/w).astype(np.int64)
    return np.asarray(depth[np.ix_(y,x)],dtype=np.float32)


def common_normals(laser,colmap,coordinates,params,original_hw):
    h,w=colmap.shape;oh,ow=original_hw
    sx,sy=w/ow,h/oh
    resized=np.asarray(coordinates,dtype=np.float64).copy()
    resized[:,0]=(resized[:,0]+.5)*sx-.5
    resized[:,1]=(resized[:,1]+.5)*sy-.5
    fx,fy,cx,cy=params
    intrinsics=[fx*sx,fy*sy,cx*sx,cy*sy]
    first,ka=sampled_normals(laser,resized,intrinsics)
    second,kb=sampled_normals(colmap,resized,intrinsics)
    return first,second,ka&kb


def freeze(split):
    protocol=DATA/"manifests/eth3d/dense_sensitivity_protocol.json"
    settings={"runner_sha256":sha256_file(Path(__file__)),"core_source_sha256":source_sha256(),
        "normal_helper_sha256":sha256_file(Path(__file__).with_name("run_eth3d_suite.py")),
        "reconstruction_protocol_sha256":sha256_file(DATA/"manifests/eth3d/colmap_dense_protocol_v2.json"),
        "split_sha256":sha256_file(DATA/"manifests/eth3d/frozen_scene_split.json"),
        "normal_operator":"Both sources use identical5point central differences of camera-frame3D points on each actual COLMAP output grid(max1600), exact resized fx/fy/cx/cy, positivez sign. Native laser map nearest-resized without filling holes; no source-agreement filter.",
        "features":"Up to128 SHA-seeded valid common native features/image; both sources use byte-identical features, appearance, sampleIDs and splits; original13scenes are not reassigned",
        "eligibility":"Before NSI outcomes, retain intersection of scenes with >=12 common samples for every one of10representations. Require >=3train,>=2validation,>=3test scenes; otherwise record ineligible and stop. No score-driven replacement.",
        "statistics":"Default ridge grid validation-tuned independently per annotation source; equal-weight common heldoutscene NSI differences,10000scene bootstraps,frozen probes.10exploratorypaired comparisons with BH10. Rank agreement both pooled NSI and scene-mean NSI, not rank invariance proof.",
        "scope":"Exploratory after defaultETH3D and failed sparse-support feasibility, frozen before these dense-source outcomes. Controls feature/appearance/operator/grid, conditional on common validity. Supplied calibration and source-view selection are shared, so not independent pose validation; cannot generalize to unobserved laser holes.",
        "format_reference":"https://github.com/colmap/colmap/blob/3.9.1/scripts/python/read_write_dense.py",
        "intrinsics_resize_reference":"https://github.com/colmap/colmap/blob/3.9.1/src/colmap/mvs/image.cc"}
    if protocol.exists():
        if json.loads(protocol.read_text(encoding="utf-8"))["settings"]!=settings:raise ValueError("Dense sensitivity protocol changed")
    else:write_json(protocol,{"created_at_utc":datetime.now(timezone.utc).isoformat(),"settings":settings})
    return protocol


def export(identifier,records,split,protocol):
    report_path=FOLDER/(identifier+".build.json")
    if report_path.exists():
        report=json.loads(report_path.read_text(encoding="utf-8"))
        if report["protocol_sha256"]!=sha256_file(protocol):raise ValueError("Dense sensitivity export protocol differs")
        for item in report["exports"].values():
            if sha256_file(item["path"])!=item["sha256"]:raise ValueError("Common feature export changed")
        return report
    parts={name:[] for name in ("features","geometry_laser","geometry_colmap","appearance","groups","split","sample_ids")}
    coverage=[]
    sources={}
    for label in ("train","validation","test"):
        for scene in split[label]:
            root=ROOT/".cache/eth3d_training"/scene
            cameras=read_cameras(root/"dslr_calibration_undistorted/cameras.txt")
            images=read_images(root/"dslr_calibration_undistorted/images.txt")
            completed=json.loads((AUDIT/(scene+"_complete.json")).read_text(encoding="utf-8"))
            file_hashes={item["path"]:item["sha256"] for item in completed["outputs"]}
            for record in records:
                if record["scene"]!=scene or record["state"]!="prepared":continue
                name=Path(record["image"]).name
                image=images[name];camera=cameras[image["camera_id"]]
                depth_path=DATA/"cache/eth3d_colmap_dense_1600"/scene/"stereo/depth_maps"/(image["name"]+".geometric.bin")
                depth_sha=sha256_file(depth_path)
                if depth_sha!=file_hashes[str(depth_path)]:raise ValueError("Dense COLMAP map SHA differs")
                colmap=read_dense(depth_path)
                colmap[~np.isfinite(colmap)|(colmap<=0)]=np.nan
                laser_original=np.load(record["output"],mmap_mode="r",allow_pickle=False)
                if sha256_file(Path(record["output"]))!=record["output_sha256"]:raise ValueError("Laser source changed")
                laser=common_grid_laser(laser_original,colmap.shape)
                local_path=DATA/"cache/features"/("eth3d_"+identifier)/scene/(name+".npz")
                local=_load_local_features(local_path)
                ng,ns,valid=common_normals(laser,colmap,local.coordinates_xy,camera.params,laser_original.shape)
                appearance=np.load(ROOT/".cache/eth3d_attributes"/scene/(name+".npy"),mmap_mode="r",allow_pickle=False)
                sampled=np.column_stack([map_coordinates(appearance[...,c],[local.coordinates_xy[:,1],local.coordinates_xy[:,0]],order=1,mode="constant",cval=np.nan) for c in range(3)])
                valid&=np.isfinite(sampled).all(axis=1)
                indices=np.flatnonzero(valid)
                chosen=indices[_deterministic_subset(len(indices),128,seed=20261023,pair_id=scene+":"+name)]
                coverage.append({"scene":scene,"image":name,"split":label,"native_features":len(local.descriptors),
                                 "common_valid_features":len(indices),"chosen":len(chosen),"grid_height":colmap.shape[0],"grid_width":colmap.shape[1]})
                if len(chosen):
                    for key,array in (("features",local.descriptors),("geometry_laser",ng),("geometry_colmap",ns),("appearance",sampled)):
                        parts[key].append(array[chosen].astype(np.float32))
                    parts["groups"].append(np.full(len(chosen),scene,dtype="U32"))
                    parts["split"].append(np.full(len(chosen),label,dtype="U10"))
                    parts["sample_ids"].append(np.asarray([f"{scene}:{name}:{i}" for i in chosen],dtype="U128"))
                sources[str(depth_path)]=depth_sha;sources[str(local_path)]=sha256_file(local_path)
                del laser_original,laser,colmap,local,appearance,sampled
            print(f"Dense source export {identifier}:{scene}",flush=True)
    if not parts["features"]:raise ValueError("No paired dense-source feature samples")
    arrays={key:np.concatenate(value) for key,value in parts.items()}
    metadata={"model":identifier,"dataset":"ETH3D","protocol_sha256":sha256_file(protocol),"annotation_source":"paired common-grid laser/COLMAP central-difference normals; sharedSobel3","source_sha256":source_sha256()}
    exports={}
    for source in ("laser","colmap"):
        path=EXPORTS/(identifier+"_"+source+"_unfiltered.npz")
        current={key:arrays[key] for key in ("features","appearance","groups","split","sample_ids")}
        current["geometry"]=arrays["geometry_"+source]
        _atomic_savez(path,compressed=True,**current,metadata_json=np.asarray(json.dumps({**metadata,"geometry_source":source},sort_keys=True)))
        exports[source]={"path":str(path),"sha256":sha256_file(path)}
    counts={scene:int(np.sum(arrays["groups"]==scene)) for label in ("train","validation","test") for scene in split[label]}
    dot=np.sum(arrays["geometry_laser"]*arrays["geometry_colmap"],axis=1)
    angles=np.degrees(np.arccos(np.clip(np.abs(dot),0,1)))
    report={"status":"EXPORTED_BEFORE_NSI","protocol_sha256":sha256_file(protocol),"exports":exports,"group_counts":counts,
            "coverage":coverage,"source_sha256":sources,"normal_axial_angle_median_degrees":float(np.median(angles)),"normal_axial_angle_p95_degrees":float(np.percentile(angles,95))}
    write_json(report_path,report)
    return report


def analyze(split,protocol,reports):
    eligible=set(scene for label in ("train","validation","test") for scene in split[label])
    for report in reports.values():eligible&={scene for scene,n in report["group_counts"].items() if n>=12}
    counts={label:sum(scene in eligible for scene in split[label]) for label in ("train","validation","test")}
    eligibility={"protocol_sha256":sha256_file(protocol),"retained_scenes":sorted(eligible),"split_counts":counts,
                 "excluded_scenes":[scene for label in ("train","validation","test") for scene in split[label] if scene not in eligible]}
    write_json(FOLDER/"common_scene_eligibility.json",eligibility)
    if counts["train"]<3 or counts["validation"]<2 or counts["test"]<3:
        raise ValueError("Insufficient paired dense coverage; NSI not computed")
    config=load_config(ROOT/"configs/default.toml")
    comparisons,summary,matrices=[],[],{"laser":[],"colmap":[]}
    for identifier,_ in MODELS:
        payloads={}
        datasets={}
        for source in ("laser","colmap"):
            raw=load_feature_npz(reports[identifier]["exports"][source]["path"])
            keep=np.isin(raw.groups,list(eligible))
            arrays={key:getattr(raw,key)[keep] for key in ("features","geometry","appearance","groups","split","sample_ids")}
            datasets[source]=arrays
            path=EXPORTS/(identifier+"_"+source+".npz")
            if not path.exists():
                _atomic_savez(path,compressed=True,**arrays,metadata_json=np.asarray(json.dumps({**raw.metadata,"eligible_scenes":sorted(eligible)},sort_keys=True)))
            result=FOLDER/(identifier+"_"+source+".json")
            if not result.exists():compute_nsi_artifacts(path,result,config)
            payloads[source]=json.loads(result.read_text(encoding="utf-8"))
            if payloads[source]["provenance"]["input_sha256"]!=sha256_file(path):raise ValueError("Paired NSI source export changed")
            row={"representation":identifier,"source":source,**{k:payloads[source]["result"][k] for k in ("NSI","geometry_r2","appearance_r2","A_accessibility","B_separation","C_information_purity")},
                 "group_NSI_mean":payloads[source]["groupwise_uncertainty"]["group_balanced_NSI_mean"],"samples":len(arrays["features"])}
            summary.append(row)
        for key in ("features","appearance","groups","split","sample_ids"):
            if not np.array_equal(datasets["laser"][key],datasets["colmap"][key]):raise ValueError("Paired sources lost exact common support")
        first=pd.DataFrame(payloads["colmap"]["groupwise_uncertainty"]["groups"]).set_index("group").sort_index()
        second=pd.DataFrame(payloads["laser"]["groupwise_uncertainty"]["groups"]).set_index("group").sort_index()
        p=paired_group_bootstrap(first.NSI,second.NSI,first.index,bootstrap_replicates=10000,seed=config.seed).to_dict()
        comparisons.append({"representation":identifier,"contrast":"COLMAP-minus-laser common-grid NSI",**p})
        for source,frame in (("colmap",first),("laser",second)):matrices[source].append(frame.NSI.to_numpy())
        del datasets,payloads,raw
        gc.collect()
    for row,q in zip(comparisons,benjamini_hochberg([r["p_value_wilcoxon"] for r in comparisons])):row["BH_q_10_source_contrasts"]=float(q)
    frame=pd.DataFrame(summary);pivot=frame.pivot(index="representation",columns="source",values="NSI")
    meanpivot=frame.pivot(index="representation",columns="source",values="group_NSI_mean")
    correlations={"pooled_NSI_spearman":float(spearmanr(pivot.laser,pivot.colmap).statistic),
                  "pooled_NSI_kendall":float(kendalltau(pivot.laser,pivot.colmap).statistic),
                  "scene_mean_NSI_spearman":float(spearmanr(meanpivot.laser,meanpivot.colmap).statistic)}
    frame.to_csv(FOLDER/"paired_source_summary.csv",index=False)
    pd.DataFrame(comparisons).to_csv(FOLDER/"paired_NSI_differences.csv",index=False)
    write_json(FOLDER/"analysis_complete.json",{"status":"PASS","protocol_sha256":sha256_file(protocol),"eligibility":eligibility,
        "comparisons":comparisons,"rank_agreement":correlations,"summary":summary,"scope":"Exploratory common-grid/common-feature annotation-source sensitivity; fixed probes for scene uncertainty; not dense coverage or source-independence proof"})
    print("Dense paired annotation sensitivity:20NSI fits and10paired contrasts complete",flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--wait",action="store_true");args=parser.parse_args()
    FOLDER.mkdir(parents=True,exist_ok=True);EXPORTS.mkdir(parents=True,exist_ok=True)
    split=json.loads((DATA/"manifests/eth3d/frozen_scene_split.json").read_text(encoding="utf-8"))
    protocol=freeze(split)
    scenes=[scene for label in ("train","validation","test") for scene in split[label]]
    deadline=time.monotonic()+36*3600
    while not all((AUDIT/(scene+"_complete.json")).exists() for scene in scenes):
        if not args.wait:raise FileNotFoundError("13complete MVS scene markers required")
        if time.monotonic()>deadline:raise TimeoutError("MVS did not complete within36h")
        time.sleep(20)
    records=json.loads((DATA/"manifests/eth3d/depth_preprocessing/all_images.json").read_text(encoding="utf-8"))
    reports={identifier:export(identifier,records,split,protocol) for identifier,_ in MODELS}
    analyze(split,protocol,reports)


if __name__=="__main__":main()
