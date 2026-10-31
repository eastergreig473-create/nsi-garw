"""Outcome-blind MegaDepth extension with explicit camera-unaware depth slopes.

CPU: fixed ResNet-50 stages plus SIFT and exploratory PCA.
GPU: nine learned representations and three extra DINO blocks after MVS.
Never infer semantic subspaces from PCA variance order or invent camera K.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import map_coordinates,sobel

from nsi_garw.adapters import ResNet50Adapter,DINOv2Adapter
from nsi_garw.attributes import rgb_to_gray
from nsi_garw.cli import _batch_adapter,build_parser
from nsi_garw.config import load_config
from nsi_garw.experiments import compute_nsi_artifacts,write_json,source_sha256
from nsi_garw.feature_builders import _atomic_savez,_deterministic_subset,_load_local_features,_save_local_features
from nsi_garw.io import FeatureDataset,load_feature_npz,sha256_file
from run_hpatches_suite import DATA,MODELS,RESULTS,ROOT

AUDIT=DATA/"manifests/megadepth"
FOLDER=RESULTS/"megadepth"
EXPORTS=DATA/"manifests/features/megadepth"
RESNET_WEIGHTS=ROOT/"weights/resnet50-11ad3fa6.pth"


def image_slope_attributes(image,depth):
    """Unit log-depth slope vector in normalized image coordinates, not a normal."""
    d=np.asarray(depth,dtype=np.float64)
    valid=np.isfinite(d)&(d>0)
    logd=np.zeros_like(d);logd[valid]=np.log(d[valid])
    keep=np.zeros_like(valid)
    keep[1:-1,1:-1]=(valid[1:-1,1:-1]&valid[1:-1,:-2]&valid[1:-1,2:]&valid[:-2,1:-1]&valid[2:,1:-1])
    gx=np.zeros_like(d);gy=np.zeros_like(d)
    gx[1:-1,1:-1]=.5*(logd[1:-1,2:]-logd[1:-1,:-2])*d.shape[1]
    gy[1:-1,1:-1]=.5*(logd[2:,1:-1]-logd[:-2,1:-1])*d.shape[0]
    geometry=np.stack([-gx,-gy,np.ones_like(d)],axis=-1)
    geometry/=np.maximum(np.linalg.norm(geometry,axis=-1,keepdims=True),1e-12)
    gray=rgb_to_gray(image)
    sx=sobel(gray,axis=1,mode="reflect");sy=sobel(gray,axis=0,mode="reflect");magnitude=np.hypot(sx,sy)
    scale=max(float(np.percentile(magnitude,99)),1e-12)
    appearance=np.stack([np.clip(sx/scale,-1,1),np.clip(sy/scale,-1,1),np.clip(magnitude/scale,0,1)],axis=-1)
    return geometry.astype(np.float32),appearance.astype(np.float32),keep


def wait_for(path,wait):
    deadline=time.monotonic()+36*3600
    while not path.exists():
        if not wait:raise FileNotFoundError(path)
        if time.monotonic()>deadline:raise TimeoutError(f"Dependency absent after36h:{path}")
        time.sleep(20)


def freeze_and_cohort(wait):
    protocol=AUDIT/"revision_feature_protocol_v3.json"
    settings={"runner_sha256":sha256_file(Path(__file__)),"core_source_sha256":source_sha256(),"seed":20261023,
        "selection":"From the prior complete-archive, outcome-blind <=16candidate/image cohort, keep up to4 eligible images per scene in SHA256(seed:revision:imagepath) order; no feature-score selection or scene reshuffling",
        "geometry":"3channel unit vector(-dlogz/du,-dlogz/dv,1)/norm, u=x/imagewidth,v=y/imageheight, centered finite difference, all5depth values finitepositive. Camera-unaware image-slope proxy, explicitly NOT a metric camera-frame surface normal. Invariant to a global positive depth scale.",
        "appearance":"3signed Sobelx/y/magnitude channels, image99percentile magnitude normalization and clipping as ETH3D",
        "sampling":"Up to128 SHA-seeded valid native feature cells/keypoints per image; no replacement for empty images; retain frozen scene labels; outcome-blind common scene intersection across18exports with >=12samples/export/scene before NSI; require>=10train,>=3validation,>=3testscenes",
        "models":"10cross-model representations as HPatches/ETH3D, plus ResNet50maxpool/layer1..4 and DINO blocks2/5/8/11; default DINO11 reused",
        "pca":"ResNet50layer3 fixed before MegaDepth outcomes; center raw training-aligned features; randomized8component PCA(seed20261023), max50000training samples by seeded choice; transform heldouttest; Pearson correlations with6attributes descriptive,10000scene bootstrap intervals with fixed PCA, no training/PCA refit; example image lowest seed hash in test cohort, fixed before component inspection",
        "interpretation":"Exploratory after HPatches/ETH3D outcomes; preregistration not claimed. Cross-dataset absolute scores not invariant. Pretrained models may have used MegaDepth; probe scene isolation is not a pretrained-training-data holdout claim.",
        "resnet_checkpoint_sha256":sha256_file(RESNET_WEIGHTS),
        "prior_cohort_protocol_sha256":sha256_file(AUDIT/"cohort_preparation_protocol.json")}
    if protocol.exists():
        if json.loads(protocol.read_text(encoding="utf-8"))["settings"]!=settings:raise ValueError("Frozen MegaDepth feature protocol changed")
    else:write_json(protocol,{"created_at_utc":datetime.now(timezone.utc).isoformat(),"settings":settings})
    wait_for(AUDIT/"cohort_complete.json",wait)
    completed=json.loads((AUDIT/"cohort_complete.json").read_text(encoding="utf-8"))
    if completed["status"]!="PASS":raise ValueError("MegaDepth cohort has not passed archive/eligibility checks")
    source=Path(completed["cohort_csv"])
    if sha256_file(source)!=completed["cohort_sha256"]:raise ValueError("MegaDepth cohort manifest changed")
    cohort_path=AUDIT/"revision_image_cohort.csv"
    frame=pd.read_csv(source,dtype={"scene":str})
    frame["selection_key"]=[hashlib.sha256(("20261023:revision:"+p).encode()).hexdigest() for p in frame.image]
    cohort=frame.sort_values("selection_key").groupby("scene",sort=True).head(4).drop(columns="selection_key").sort_values(["scene","image"])
    text=cohort.to_csv(index=False)
    if cohort_path.exists() and cohort_path.read_text(encoding="utf-8")!=text:raise ValueError("Refusing to replace frozen image cohort")
    if not cohort_path.exists():
        with tempfile.NamedTemporaryFile('w',dir=AUDIT,prefix='revision-cohort-',suffix='.tmp',encoding='utf-8',newline='',delete=False) as handle:
            handle.write(text);temporary=handle.name
        os.replace(temporary,cohort_path)
    return protocol,cohort


def image_key(depth_path):
    relative=Path(depth_path).relative_to(DATA/"datasets/MegaDepth/MegaDepth_v1").as_posix()
    return hashlib.sha256(relative.encode()).hexdigest()[:24]


def export(identifier,adapter,cohort,protocol):
    output=EXPORTS/(identifier+"_unfiltered.npz");report_path=FOLDER/(identifier+".build.json")
    if output.exists() and report_path.exists():
        record=json.loads(report_path.read_text(encoding="utf-8"))
        if record["protocol_sha256"]!=sha256_file(protocol) or record["output_sha256"]!=sha256_file(output):raise ValueError("MegaDepth export changed")
        return
    parts={key:[] for key in ("features","geometry","appearance","groups","split","sample_ids")}
    coverage=[];metadata=None
    for number,(_,row) in enumerate(cohort.iterrows()):
        image_path,depth_path=Path(row.image),Path(row.depth)
        if sha256_file(image_path)!=row.image_sha256 or sha256_file(depth_path)!=row.depth_sha256:raise ValueError("MegaDepth source bytes changed")
        image=np.asarray(Image.open(image_path).convert("RGB"))
        with h5py.File(depth_path,"r") as h:depth=np.asarray(h["depth"],dtype=np.float32)
        if depth.shape!=image.shape[:2] or np.any(depth<0):raise ValueError("Ineligible metric-depth image reached feature exporter")
        image_hash=image_key(depth_path)
        cache=DATA/"cache/features"/("megadepth_"+identifier)/row.scene/(image_hash+".npz")
        if cache.exists():local=_load_local_features(cache)
        else:local=adapter.extract(image);_save_local_features(cache,local)
        if metadata is None:metadata=dict(local.metadata)
        elif local.metadata!=metadata:raise ValueError("MegaDepth preprocessing changed within representation")
        geometry,appearance,valid=image_slope_attributes(image,depth)
        ij=np.floor(local.coordinates_xy+.5).astype(np.int64)
        inside=(ij[:,0]>=0)&(ij[:,0]<depth.shape[1])&(ij[:,1]>=0)&(ij[:,1]<depth.shape[0])
        good=np.flatnonzero(inside)
        good=good[valid[ij[good,1],ij[good,0]]]
        chosen=good[_deterministic_subset(len(good),128,seed=20261023,pair_id=row.scene+":"+image_hash)]
        xy=local.coordinates_xy[chosen]
        appearance_values=np.column_stack([map_coordinates(appearance[...,c],[xy[:,1],xy[:,0]],order=1,mode="nearest") for c in range(3)])
        geometry_values=geometry[ij[chosen,1],ij[chosen,0]]
        coverage.append({"scene":row.scene,"image":str(image_path),"split":row.split,"native_features":len(local.descriptors),"valid_features":len(good),"samples":len(chosen),"native_cache":str(cache),"native_cache_sha256":sha256_file(cache)})
        if len(chosen):
            for array_key,value in (("features",local.descriptors[chosen]),("geometry",geometry_values),("appearance",appearance_values)):
                parts[array_key].append(value.astype(np.float32))
            parts["groups"].append(np.full(len(chosen),row.scene,dtype="U8"));parts["split"].append(np.full(len(chosen),row.split,dtype="U10"))
            parts["sample_ids"].append(np.asarray([f"{row.scene}:{image_hash}:{i}" for i in chosen],dtype="U128"))
        if (number+1)%20==0:print(f"MegaDepth {identifier}:{number+1}/{len(cohort)}images",flush=True)
        del image,depth,geometry,appearance,local
    if not parts["features"]:raise ValueError("No valid MegaDepth native features")
    arrays={key:np.concatenate(value) for key,value in parts.items()}
    if len(np.unique(arrays['sample_ids']))!=len(arrays['sample_ids']):raise ValueError('MegaDepth sample identifiers are not unique across images')
    metadata.update({"dataset":"MegaDepth v1 fixed4-image/scene exploratory extension","protocol_sha256":sha256_file(protocol),
        "annotation_source":"camera-unaware normalized-image log-depth slope3, signed Sobel3; not surface normals","sampling":"up to128 valid native features/image","cohort_sha256":sha256_file(AUDIT/"revision_image_cohort.csv")})
    FeatureDataset(**arrays,metadata=metadata).validate()
    _atomic_savez(output,compressed=True,**arrays,metadata_json=np.asarray(json.dumps(metadata,sort_keys=True)))
    write_json(report_path,{"status":"EXPORTED_BEFORE_NSI","output":str(output),"output_sha256":sha256_file(output),
        "protocol_sha256":sha256_file(protocol),"metadata":metadata,"coverage":coverage,"samples":len(arrays["features"]),
        "group_counts":{scene:int(np.sum(arrays["groups"]==scene)) for scene in cohort.scene.unique()}})


def pca_pilot(cohort,protocol):
    from sklearn.decomposition import PCA
    source=EXPORTS/"resnet50_layer3_unfiltered.npz"
    d=load_feature_npz(source)
    train=d.split=="train";test=d.split=="test"
    values=d.features[train]
    if len(values)>50000:
        indices=np.sort(np.random.default_rng(20261023).choice(len(values),50000,replace=False));values=values[indices]
    pca=PCA(n_components=8,svd_solver="randomized",random_state=20261023).fit(values)
    scores=pca.transform(d.features[test]);attributes=np.column_stack([d.geometry[test],d.appearance[test]])
    columns=["image_slope_x","image_slope_y","image_slope_z","Sobel_x","Sobel_y","Sobel_magnitude"]
    groups=d.groups[test]
    unique=np.unique(groups)
    if len(unique)<3:raise ValueError("PCA pilot needs at least3heldoutscenes for cluster intervals")
    sums=[]
    for group in unique:
        x,y=scores[groups==group],attributes[groups==group]
        sums.append((len(x),x.sum(axis=0),y.sum(axis=0),(x*x).sum(axis=0),(y*y).sum(axis=0),x.T@y))
    n,sx,sy,sxx,syy,sxy=[np.asarray([item[i] for item in sums]) for i in range(6)]
    index=np.random.default_rng(20261023).integers(0,len(unique),size=(10000,len(unique)))
    nb=n[index].sum(axis=1);xb=sx[index].sum(axis=1);yb=sy[index].sum(axis=1)
    xx=sxx[index].sum(axis=1)-xb*xb/nb[:,None]
    yy=syy[index].sum(axis=1)-yb*yb/nb[:,None]
    numerator=sxy[index].sum(axis=1)-xb[:,:,None]*yb[:,None,:]/nb[:,None,None]
    denominator=np.sqrt(np.maximum(xx[:,:,None]*yy[:,None,:],0))
    bootstrap=np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>1e-12)
    lo,hi=np.nanpercentile(bootstrap,[2.5,97.5],axis=0)
    rows=[]
    for pc in range(8):
        for attribute,name in enumerate(columns):
            rows.append({"component":pc+1,"attribute":name,"test_Pearson_r":float(np.corrcoef(scores[:,pc],attributes[:,attribute])[0,1]),
                         "scene_bootstrap_CI_low":lo[pc,attribute],"scene_bootstrap_CI_high":hi[pc,attribute],
                         "valid_bootstrap_replicates":int(np.isfinite(bootstrap[:,pc,attribute]).sum())})
    table=pd.DataFrame(rows);table.to_csv(FOLDER/"pca_attribute_correlations.csv",index=False)
    _atomic_savez(FOLDER/"pca_heldout_scores.npz",compressed=True,scores=scores.astype(np.float32),attributes=attributes.astype(np.float32),groups=groups,sample_ids=d.sample_ids[test])
    test_images=cohort.loc[cohort.split=="test"].copy()
    test_images["key"]=[hashlib.sha256(("20261023:pca-example:"+s).encode()).hexdigest() for s in test_images.image]
    example=test_images.sort_values("key").iloc[0]
    cache=DATA/"cache/features/megadepth_resnet50_layer3"/example.scene/(image_key(example.depth)+".npz")
    local=_load_local_features(cache)
    transformed=pca.transform(local.descriptors)
    _atomic_savez(FOLDER/"pca_model_and_example.npz",compressed=True,mean=pca.mean_,components=pca.components_,
        explained_variance_ratio=pca.explained_variance_ratio_,training_score_scale=np.sqrt(pca.explained_variance_),
        example_scores=transformed,example_coordinates=local.coordinates_xy)
    write_json(FOLDER/"pca_complete.json",{"status":"PASS","protocol_sha256":sha256_file(protocol),"input_sha256":sha256_file(source),
        "training_samples":len(values),"test_samples":int(test.sum()),"test_scenes":len(np.unique(groups)),"example_image":example.image,
        "example_scene":example.scene,"example_cache_sha256":sha256_file(cache),"explained_variance_ratio":pca.explained_variance_ratio_.tolist(),
        "scope":"Exploratory8variance directions, no semantic geometry/appearance assignment; heldout correlations descriptive, not evidence for complementary semantic subspaces"})
    print("MegaDepth PCA8directions and48heldout attribute correlations complete",flush=True)


def analyze(protocol,cohort):
    identifiers=[i for i,_ in MODELS]+["resnet50_"+layer for layer in ("maxpool","layer1","layer2","layer3","layer4")]+["dinov2_vits14_block"+str(i)+"_518" for i in (2,5,8)]
    if not all((FOLDER/(i+".build.json")).exists() for i in identifiers):return
    eligible=set(cohort.scene)
    for identifier in identifiers:
        report=json.loads((FOLDER/(identifier+".build.json")).read_text(encoding="utf-8"));eligible&={scene for scene,n in report["group_counts"].items() if n>=12}
    counts={label:len(set(cohort.loc[cohort.split==label,"scene"])&eligible) for label in ("train","validation","test")}
    write_json(FOLDER/"common_scene_eligibility.json",{"protocol_sha256":sha256_file(protocol),"retained_scenes":sorted(eligible),"split_counts":counts,"candidate_scenes":int(cohort.scene.nunique())})
    if counts["train"]<10 or counts["validation"]<3 or counts["test"]<3:raise ValueError("Insufficient shared MegaDepth scene coverage for18exports")
    config=load_config(ROOT/"configs/default.toml");rows=[]
    for identifier in identifiers:
        d=load_feature_npz(EXPORTS/(identifier+"_unfiltered.npz"));keep=np.isin(d.groups,list(eligible))
        output=EXPORTS/(identifier+".npz")
        if not output.exists():
            _atomic_savez(output,compressed=True,**{key:getattr(d,key)[keep] for key in ("features","geometry","appearance","groups","split","sample_ids")},metadata_json=np.asarray(json.dumps({**d.metadata,"common_scenes":sorted(eligible)},sort_keys=True)))
        result=FOLDER/(identifier+".json")
        if not result.exists():compute_nsi_artifacts(output,result,config)
        p=json.loads(result.read_text(encoding="utf-8"));r,u=p["result"],p["groupwise_uncertainty"]
        rows.append({"representation":identifier,**{k:r[k] for k in ("A_accessibility","B_separation","C_information_purity","NSI","geometry_r2","appearance_r2")},
            "scene_NSI_mean":u["group_balanced_NSI_mean"],"scene_NSI_CI_low":u["confidence_interval"][0],"scene_NSI_CI_high":u["confidence_interval"][1],"test_scenes":u["group_count"],"input_sha256":p["provenance"]["input_sha256"]})
        print(f"MegaDepth NSI complete:{identifier}={r['NSI']:.6g}",flush=True)
        del d;gc.collect()
    pd.DataFrame(rows).to_csv(FOLDER/"nsi_summary.csv",index=False)
    write_json(FOLDER/"analysis_complete.json",{"status":"PASS","protocol_sha256":sha256_file(protocol),"rows":rows,"retained_scene_counts":counts,
        "scope":"18representations/stages with common within-dataset scene eligibility; image-space depth-slope proxy; not calibrated normal comparison or measurement-invariant cross-dataset scale"})


def main():
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=["cpu","gpu","analyze"]);parser.add_argument("--wait",action="store_true");args=parser.parse_args()
    FOLDER.mkdir(parents=True,exist_ok=True);EXPORTS.mkdir(parents=True,exist_ok=True)
    protocol,cohort=freeze_and_cohort(args.wait)
    if args.mode=="cpu":
        for layer in ("layer3","maxpool","layer1","layer2","layer4"):
            identifier="resnet50_"+layer
            adapter=ResNet50Adapter(RESNET_WEIGHTS,layer=layer,device="cpu")
            export(identifier,adapter,cohort,protocol)
            del adapter;gc.collect()
            if layer=="layer3":pca_pilot(cohort,protocol)
        from nsi_garw.adapters import SIFTAdapter
        export("sift_default",SIFTAdapter(4096),cohort,protocol)
    elif args.mode=="gpu":
        split=json.loads((DATA/"manifests/eth3d/frozen_scene_split.json").read_text(encoding="utf-8"))
        for label in ("train","validation","test"):
            for scene in split[label]:wait_for(DATA/"manifests/eth3d/colmap_dense"/(scene+"_complete.json"),args.wait)
        for identifier,arguments in MODELS:
            if identifier=="sift_default":continue
            parsed=build_parser().parse_args(["build-hpatches-model",*arguments,"--export-id",identifier,"--pairs-manifest",str(AUDIT/"revision_image_cohort.csv"),"--dataset-root",str(DATA/"datasets/MegaDepth/MegaDepth_v1"),"--cache-root",str(DATA/"cache"),"--output",str(EXPORTS/"unused.npz"),"--device","cuda"])
            adapter=_batch_adapter(parsed);export(identifier,adapter,cohort,protocol)
            del adapter;gc.collect()
            import torch
            torch.cuda.empty_cache()
        for layer in (2,5,8):
            identifier=f"dinov2_vits14_block{layer}_518"
            adapter=DINOv2Adapter(repository=ROOT/"external/dinov2",checkpoint=DATA/"models/dinov2_vits14_pretrain.pth",layer=layer,max_size=518,device="cuda")
            export(identifier,adapter,cohort,protocol)
            del adapter;gc.collect();torch.cuda.empty_cache()
    if args.mode=="analyze" and args.wait:
        expected=[i for i,_ in MODELS]+["resnet50_"+l for l in ("maxpool","layer1","layer2","layer3","layer4")]+[f"dinov2_vits14_block{i}_518" for i in (2,5,8)]
        for identifier in expected:wait_for(FOLDER/(identifier+".build.json"),True)
    if args.mode=="gpu":
        write_json(FOLDER/"gpu_exports_complete.json",{"status":"PASS","protocol_sha256":sha256_file(protocol),"scope":"GPU feature exports only; NSI analysis has a separate single owner"})
    if args.mode=="cpu":
        write_json(FOLDER/"cpu_exports_complete.json",{"status":"PASS","protocol_sha256":sha256_file(protocol)})
    if args.mode=="analyze":
        analyze(protocol,cohort)


if __name__=="__main__":main()
