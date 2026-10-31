"""Frozen real ETH3D encoder retrieval, non-oracle GARW and scene associations.

Uses native cached full descriptors, never injects positives, and never uses
laser labels in the observable gate. Laser depth defines evaluable queries.
"""
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nsi_garw.config import load_config
from nsi_garw.eth3d import read_cameras, read_images
from nsi_garw.experiments import run_garw_table, source_sha256, write_json
from nsi_garw.feature_builders import _load_local_features, _deterministic_subset
from nsi_garw.garw import standardized_projected_cosine
from nsi_garw.io import load_feature_npz, sha256_file
from nsi_garw.local_proxy import fit_local_nsi_proxy
from nsi_garw.metrics import (fit_ridge_probe,orthonormal_basis,residualized_appearance_basis,
    normalized_subspace_overlap,estimate_information_purity,r2_per_output,mean_finite)
from nsi_garw.statistics import paired_group_bootstrap,benjamini_hochberg,spearman_group_association
from eth3d_matching_common import project_visible,world_to_camera
from run_hpatches_suite import DATA,MODELS,RESULTS,ROOT

FOLDER=RESULTS/"eth3d_garw"
AUDIT=DATA/"manifests/eth3d/retrieval"
PAIRS=AUDIT/"pairs.csv"
Q=32
K=16
GATED=("loftr_outdoor_coarse_1024","roma_outdoor_scale16_560")


def freeze_pairs(split,records):
    """View overlap and annotation-window feasibility only; no matching outcomes."""
    protocol=AUDIT/"protocol.json"
    settings={"runner_sha256":sha256_file(Path(__file__)),"geometry_helper_sha256":sha256_file(Path(__file__).with_name("eth3d_matching_common.py")),
        "core_source_sha256":source_sha256(),"seed":20261023,"queries_per_pair_maximum":Q,"candidates_maximum":K,
        "pairs":"calibration3/test5scenes; at most16 unordered image pairs per scene by seeded SHA order after >=20 shared supplied SfM tracks and >=12 frozen NSI samples/image for both gated encoders; lexical a->b direction; no replacement based on descriptor outcomes",
        "query_selection":"All native source features with positive nearest laser depth, forward projection in target and target nearest laser depth within2percent projected z; choose up to32 by seeded index sampling",
        "labels":"forward reprojection error<=3 original-resolution pixels; target candidate depth not required; missing retrieved positives remain zero; different from HPatches symmetric-error/full-native-matcher tasks",
        "gate":"5train-scene probes using default validation-selected alphas;3calibration-scene group-OOF observable proxy;5testscenes never used for fitting; proxy inputs texture, projected disagreement, raw margin; lambda selected oncalibration",
        "texture":"source signed-Sobel magnitude channel sampled from verified full-resolution attribute cache at queries; observable image property",
        "statistics":"5 equal-weight test scenes,10000scene bootstraps and Wilcoxon; BH family8 exploratory GARW comparisons (2encoders x2metrics x2baselines);10 within-representation ETH3D association attempts, constant NSI is not estimable and excluded from BH denominator",
        "scope":"Exploratory frozen after prior HPatches/defaultETH3D outcomes and before these retrieval outcomes; camera/laser coverage conditional; not a full paired-matcher or pose benchmark",
        "split_sha256":sha256_file(DATA/"manifests/eth3d/frozen_scene_split.json")}
    if protocol.exists():
        if json.loads(protocol.read_text(encoding="utf-8"))["settings"]!=settings:
            raise ValueError("Frozen ETH3D retrieval protocol changed")
    else:
        write_json(protocol,{"created_at_utc":datetime.now(timezone.utc).isoformat(),"settings":settings})
    if PAIRS.exists():
        return pd.read_csv(PAIRS)
    common=None
    for identifier in GATED:
        d=load_feature_npz(DATA/"manifests/features"/("eth3d_"+identifier+".npz"))
        windows=Counter(":".join(str(v).split(":")[:2]) for v in d.sample_ids)
        good={w for w,n in windows.items() if n>=12}
        common=good if common is None else common&good
        del d
    rows=[]
    for label in ("validation","test"):
        for scene in split[label]:
            path=ROOT/".cache/eth3d_training"/scene/"dslr_calibration_undistorted"
            import pycolmap
            rec=pycolmap.Reconstruction(str(path))
            images=sorted(rec.images.values(),key=lambda v:v.name)
            tracks={Path(im.name).name:{p.point3D_id for p in im.points2D if p.has_point3D()} for im in images}
            names=[Path(im.name).name for im in images if scene+":"+Path(im.name).name in common]
            candidates=[]
            for i,a in enumerate(names):
                for b in names[i+1:]:
                    shared=len(tracks[a]&tracks[b])
                    if shared>=20:
                        pair_id=scene+":"+a+":"+b
                        candidates.append({"scene":scene,"split":label,"pair_id":pair_id,"source":a,"target":b,"shared_SfM_tracks":shared})
            selected=sorted(candidates,key=lambda r:hashlib.sha256(("20261023:"+r["pair_id"]).encode()).hexdigest())[:16]
            if not selected:
                raise ValueError(f"Frozen scene has no coverage-eligible pairs:{scene}")
            rows.extend(selected)
            print(f"ETH3D retrieval frozen {scene}:{len(selected)} pairs/{len(candidates)} eligible",flush=True)
    frame=pd.DataFrame(rows)
    frame.to_csv(PAIRS,index=False)
    write_json(AUDIT/"pairs_complete.json",{"status":"PASS","protocol_sha256":sha256_file(protocol),
        "pairs_sha256":sha256_file(PAIRS),"pairs":len(frame),"scenes":frame.groupby("scene").size().to_dict()})
    return frame


def model_run(identifier,pairs,records,config):
    destination=FOLDER/identifier
    destination.mkdir(parents=True,exist_ok=True)
    complete=destination/"retrieval_complete.json"
    if complete.exists():
        return
    dataset=load_feature_npz(DATA/"manifests/features"/("eth3d_"+identifier+".npz"))
    default=json.loads((RESULTS/"eth3d"/("eth3d_"+identifier+".json")).read_text(encoding="utf-8"))
    original_path=DATA/"manifests/features"/("eth3d_"+identifier+".npz")
    if sha256_file(original_path)!=default["provenance"]["input_sha256"].lower():
        raise ValueError("Default ETH3D feature export changed")
    gated=identifier in GATED
    if gated:
        train=dataset.split=="train"
        gp=fit_ridge_probe(dataset.features[train],dataset.geometry[train],default["result"]["geometry_alpha"])
        ap=fit_ridge_probe(dataset.features[train],dataset.appearance[train],default["result"]["appearance_alpha"])
        ug=orthonormal_basis(gp.coefficients_standardized,8,1e-6)
        ua_raw=orthonormal_basis(ap.coefficients_standardized,8,1e-6)
        ua=residualized_appearance_basis(ug,ap.coefficients_standardized,8,1e-6)
        separation=1-normalized_subspace_overlap(ug,ua_raw)
        window_ids=np.asarray([":".join(str(v).split(":")[:2]) for v in dataset.sample_ids])
    provenance={"runner_sha256":sha256_file(Path(__file__)),"core_source_sha256":source_sha256(),
        "pairs_sha256":sha256_file(PAIRS),"input_sha256":sha256_file(original_path),"native_cache_sha256":{}}
    raw,coverage,candidates=[],[],[]
    for scene,scene_pairs in pairs.groupby("scene",sort=False):
        root=ROOT/".cache/eth3d_training"/scene
        cameras=read_cameras(root/"dslr_calibration_undistorted/cameras.txt")
        images=read_images(root/"dslr_calibration_undistorted/images.txt")
        local,depths={},{}
        needed=sorted(set(scene_pairs.source)|set(scene_pairs.target))
        for name in needed:
            path=DATA/"cache/features"/("eth3d_"+identifier)/scene/(name+".npz")
            local[name]=_load_local_features(path)
            provenance["native_cache_sha256"][str(path)]=sha256_file(path)
            depths[name]=np.load(records[scene,name]["output"],mmap_mode="r",allow_pickle=False)
        for pair_number,(_,row) in enumerate(scene_pairs.iterrows()):
            first,second=local[row.source],local[row.target]
            t=world_to_camera(images[row.target])@np.linalg.inv(world_to_camera(images[row.source]))
            projected,visible=project_visible(first.coordinates_xy,depths[row.source],depths[row.target],
                cameras[images[row.source]["camera_id"]].params,cameras[images[row.target]["camera_id"]].params,t)
            valid_indices=np.flatnonzero(visible)
            selected=valid_indices[_deterministic_subset(len(valid_indices),Q,seed=config.seed,pair_id=row.pair_id)]
            coverage.append({"scene":scene,"pair_id":row.pair_id,"split":row.split,"native_source_features":len(first.descriptors),
                             "native_target_features":len(second.descriptors),"laser_visible_queries":len(valid_indices),"sampled_queries":len(selected)})
            if not len(selected) or not len(second.descriptors):
                raw.append({"scene":scene,"pair_id":row.pair_id,"split":row.split,"queries":len(selected),
                            **{f"accuracy_{threshold}":0. for threshold in (1,3,5,10)}})
                if gated:
                    raise ValueError("A frozen gated pair has no laser-evaluable query or native target feature; revise eligibility before outcomes in a new protocol version")
                continue
            x,y=first.descriptors[selected].astype(np.float32),second.descriptors.astype(np.float32)
            x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
            y/=np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-12)
            similarity=x@y.T
            k=min(K,len(y)); indices=np.argsort(-similarity,axis=1,kind="stable")[:,:k]
            scores=np.take_along_axis(similarity,indices,axis=1)
            errors=np.linalg.norm(second.coordinates_xy[indices]-projected[selected,None,:],axis=2)
            raw.append({"scene":scene,"pair_id":row.pair_id,"split":row.split,"queries":len(selected),
                        **{f"accuracy_{threshold}":float((errors[:,0]<=threshold).mean()) for threshold in (1,3,5,10)}})
            if gated:
                query_indices=np.repeat(selected,k); target_indices=indices.ravel()
                sg=standardized_projected_cosine(first.descriptors[query_indices],second.descriptors[target_indices],ug,gp.x_mean,gp.x_scale).reshape(len(selected),k)
                sa=standardized_projected_cosine(first.descriptors[query_indices],second.descriptors[target_indices],ua,gp.x_mean,gp.x_scale).reshape(len(selected),k)
                window=window_ids==(scene+":"+row.target)
                if window.sum()<12:
                    raise ValueError("Frozen gated target window lost NSI reference support")
                pg,pa=gp.predict(dataset.features[window]),ap.predict(dataset.features[window])
                a=.5*(np.clip(mean_finite(r2_per_output(dataset.geometry[window],pg)),0,1)+np.clip(mean_finite(r2_per_output(dataset.appearance[window],pa)),0,1))
                c=estimate_information_purity(pg,pa,replace(config,seed=config.seed+pair_number+1)).purity
                nsi_reference=float(a*separation*c)
                texture=np.load(ROOT/".cache/eth3d_attributes"/scene/(row.source+".npy"),mmap_mode="r",allow_pickle=False)
                from scipy.ndimage import map_coordinates
                texture_values=map_coordinates(texture[...,2],[first.coordinates_xy[selected,1],first.coordinates_xy[selected,0]],order=1,mode="nearest")
                disagreement=np.abs(sg-sa).mean(axis=1)
                margin=scores[:,0]-scores[:,min(1,k-1)]
                for qi,source_index in enumerate(selected):
                    for ci in range(k):
                        candidates.append({"query_id":row.pair_id+":"+str(source_index),"scene":scene,"pair_id":row.pair_id,
                            "split":"calibration" if row.split=="validation" else "evaluation","label":int(errors[qi,ci]<=3),
                            "s_geometry":sg[qi,ci],"s_appearance":sa[qi,ci],"s_baseline":scores[qi,ci],
                            "texture_energy":texture_values[qi],"projected_disagreement":disagreement[qi],"raw_margin":margin[qi],
                            "nsi_reference":nsi_reference,"reprojection_error":errors[qi,ci],"candidate_rank_raw":ci+1})
                del texture
        del local,depths
        print(f"ETH3D retrieval {identifier}:{scene} complete",flush=True)
    pd.DataFrame(raw).to_csv(destination/"raw_per_pair.csv",index=False)
    pd.DataFrame(coverage).to_csv(destination/"query_coverage.csv",index=False)
    if gated:
        table,proxy,trace=fit_local_nsi_proxy(pd.DataFrame(candidates),feature_columns=["texture_energy","projected_disagreement","raw_margin"],lambdas=config.ridge.lambdas,seed=config.seed)
        candidate_file=DATA/"manifests/garw"/("eth3d_"+identifier+"_q32_k16.csv")
        table.to_csv(candidate_file,index=False)
        write_json(destination/"proxy.json",{"model":proxy.to_dict(),"trace":trace})
        run_garw_table(candidate_file,destination/"non_oracle",config)
        oracle=table.copy();oracle["nsi_local"]=oracle.nsi_reference;oracle["nsi_mode"]="oracle_annotation"
        oracle_file=candidate_file.with_name(candidate_file.stem+"_oracle.csv");oracle.to_csv(oracle_file,index=False)
        run_garw_table(oracle_file,destination/"oracle_annotation",config)
    write_json(complete,{"status":"PASS","representation":identifier,"provenance":provenance,"pair_count":len(raw),
                        "test_scenes":5,"gated":gated,"coverage":coverage})
    del dataset
    gc.collect()


def analyze(pairs,config):
    associations,comparisons,summary=[],[],[]
    for identifier,_ in MODELS:
        root=FOLDER/identifier
        if not (root/"retrieval_complete.json").exists():
            raise ValueError("ETH3D retrieval matrix incomplete")
        frame=pd.read_csv(root/"raw_per_pair.csv")
        test=frame.loc[frame.split=="test"].groupby("scene",as_index=False).accuracy_3.mean()
        nsi=json.loads((RESULTS/"eth3d"/("eth3d_"+identifier+".json")).read_text(encoding="utf-8"))
        points=pd.DataFrame(nsi["groupwise_uncertainty"]["groups"])[["group","NSI"]].rename(columns={"group":"scene"}).merge(test,on="scene",validate="one_to_one")
        if len(points)!=5: raise ValueError("ETH3D association lost a frozen test scene")
        points.to_csv(root/"association_points.csv",index=False)
        row={"representation":identifier,"test_scenes":5,"target":"raw32query forward Top1@3"}
        if points.NSI.nunique()<2 or points.accuracy_3.nunique()<2:
            row.update({"status":"NOT_ESTIMABLE_CONSTANT_VECTOR","spearman_rho":None,"permutation_p_value":None})
        else:
            row.update(spearman_group_association(points.NSI,points.accuracy_3,points.scene,bootstrap_replicates=10000,permutation_replicates=1000,seed=config.seed).to_dict())
            row["status"]="ESTIMATED_EXPLORATORY_N5"
        associations.append(row)
        if identifier in GATED:
            block=pd.read_csv(root/"non_oracle/garw_per_scene.csv")
            summary.extend(block.assign(representation=identifier).to_dict("records"))
            for metric in ("top1","average_precision"):
                pivot=block.pivot(index="scene",columns="method",values=metric)
                for baseline in ("descriptor_baseline","fixed_weight"):
                    p=paired_group_bootstrap(pivot.garw,pivot[baseline],pivot.index,bootstrap_replicates=10000,seed=config.seed).to_dict()
                    comparisons.append({"representation":identifier,"metric":metric,"baseline":baseline,**p})
    valid=[r for r in associations if r.get("permutation_p_value") is not None]
    for row,q in zip(valid,benjamini_hochberg([r["permutation_p_value"] for r in valid])):row["BH_q_estimable_associations"]=float(q)
    for row,q in zip(comparisons,benjamini_hochberg([r["p_value_wilcoxon"] for r in comparisons])):row["BH_q_8_ETH_GARW_comparisons"]=float(q)
    pd.DataFrame(associations).to_csv(FOLDER/"association_summary.csv",index=False)
    pd.DataFrame(comparisons).to_csv(FOLDER/"paired_comparisons.csv",index=False)
    pd.DataFrame(summary).to_csv(FOLDER/"per_scene_garw.csv",index=False)
    write_json(FOLDER/"analysis_complete.json",{"status":"PASS","script_sha256":sha256_file(Path(__file__)),
        "pairs_sha256":sha256_file(PAIRS),"associations":associations,"comparisons":comparisons,
        "scope":"Real conditional laser-visible encoder-retrieval experiments;5testscenes; no full-matcher/pose or prospective confirmatory claim"})
    print("ETH3D retrieval/GARW/association analysis complete",flush=True)


def main():
    AUDIT.mkdir(parents=True,exist_ok=True);FOLDER.mkdir(parents=True,exist_ok=True)
    split=json.loads((DATA/"manifests/eth3d/frozen_scene_split.json").read_text(encoding="utf-8"))
    records=json.loads((DATA/"manifests/eth3d/depth_preprocessing/all_images.json").read_text(encoding="utf-8"))
    records={(r["scene"],Path(r["image"]).name):r for r in records if r["state"]=="prepared"}
    config=load_config(ROOT/"configs/default.toml")
    pairs=freeze_pairs(split,records)
    # Gated outcomes first; all ten raw retrieval associations follow.
    for identifier in [*GATED,*[i for i,_ in MODELS if i not in GATED]]:
        model_run(identifier,pairs,records,config)
    analyze(pairs,config)


if __name__=="__main__":main()
