"""Uniform CPU FP32 ResNet/DINO MNN matching on all85heldout HPatches pairs."""
from datetime import datetime,timezone
import gc
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from nsi_garw.adapters import ResNet50Adapter,DINOv2Adapter
from nsi_garw.degradations import apply_degradation
from nsi_garw.evaluation import homography_reprojection_errors
from nsi_garw.experiments import write_json
from nsi_garw.feature_builders import _load_local_features,_save_local_features
from nsi_garw.io import sha256_file
from run_hpatches_suite import DATA,RESULTS,ROOT


def mnn_indices(a,b):
    if not len(a) or not len(b):return np.empty((0,2),dtype=np.int64)
    a=np.asarray(a,dtype=np.float32);b=np.asarray(b,dtype=np.float32)
    a=a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12);b=b/np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-12)
    similarity=a@b.T
    ab=np.argmax(similarity,axis=1);ba=np.argmax(similarity,axis=0);first=np.arange(len(a));keep=ba[ab]==first
    return np.column_stack([first[keep],ab[keep]])


def main():
    cv2.setNumThreads(2)
    folder=RESULTS/"auxiliary_matching";folder.mkdir(parents=True,exist_ok=True)
    pairs_path=DATA/"manifests/hpatches/hpatches_pairs.csv"
    pairs=pd.read_csv(pairs_path);pairs=pairs.loc[pairs.split=="test"]
    settings={"runner_sha256":sha256_file(Path(__file__)),"pairs_manifest_sha256":sha256_file(pairs_path),
        "representations":["resnet50_layer3_crop224","dinov2_vits14_block11_518"],"precision":"uniformCPUFP32 forbothreferenceandtargets; freshseparatecaches",
        "matching":"cosineMNN;allnativefeatures;firstindexdeterministic ties;3pixelRMSsymmetrichomographyerror;zero-matchpairszero",
        "cohort":"all85pairs/17heldoutsequences;clean andtarget-onlytexture0.8;no post-outcome sequence selection",
        "degradation":"same publictextureoperator;mean-colorblend isdeterministic;RGBuint8rounding;cleanreference",
        "scope":"auxiliary encoder matching, nottrainedmatchingheads;center-crop/featuregridcoverage differs; withinmethod associations only; exploratory after prior matching outcomes"}
    protocol=DATA/"manifests/hpatches/auxiliary_matching_protocol.json"
    if protocol.exists():
        if json.loads(protocol.read_text(encoding="utf-8"))["settings"]!=settings:raise ValueError("Auxiliary matching protocol changed")
    else:write_json(protocol,{"created_at_utc":datetime.now(timezone.utc).isoformat(),"settings":settings})
    image_root=DATA/"datasets/HPatches/hpatches-sequences-release"
    for identifier in settings["representations"]:
        out=folder/identifier;out.mkdir(parents=True,exist_ok=True)
        if (out/"complete.json").exists():continue
        adapter=(ResNet50Adapter(ROOT/"weights/resnet50-11ad3fa6.pth",layer="layer3",device="cpu") if identifier.startswith("resnet")
                 else DINOv2Adapter(ROOT/"external/dinov2",DATA/"models/dinov2_vits14_pretrain.pth",layer=11,max_size=518,device="cpu"))
        records=[];inputs={}
        for condition in ("clean","texture"):
            for number,(_,row) in enumerate(pairs.iterrows()):
                local=[]
                for role,relative in (("reference",row.reference_image),("target",row.target_image)):
                    cache=DATA/"cache/features"/(identifier+"_rq5_cpu_"+("reference" if role=="reference" else condition))/row.sequence/(Path(relative).stem+".npz")
                    if cache.exists():features=_load_local_features(cache)
                    else:
                        image=np.asarray(Image.open(image_root/relative).convert("RGB"))
                        if role=="target" and condition=="texture":image=np.clip(np.rint(apply_degradation(image,"texture",.8,seed=20261023)*255),0,255).astype(np.uint8)
                        features=adapter.extract(image);_save_local_features(cache,features)
                    inputs[str(cache)]=sha256_file(cache);local.append(features)
                first,second=local
                matched=mnn_indices(first.descriptors,second.descriptors)
                h=np.loadtxt(image_root/row.homography)
                errors=homography_reprojection_errors(first.coordinates_xy[matched[:,0]],second.coordinates_xy[matched[:,1]],h,symmetric=True)
                record={"scene":row.sequence,"pair_id":row.pair_id,"condition":condition,"match_count":len(errors),
                        **{f"mma_at_{threshold}":float(np.mean(np.isfinite(errors)&(errors<=threshold))) if len(errors) else 0. for threshold in (1,3,5,10)}}
                records.append(record)
                if (number+1)%10==0:print(f"Auxiliary matching {identifier} {condition}:{number+1}/85",flush=True)
        frame=pd.DataFrame(records);frame.to_csv(out/"per_pair.csv",index=False)
        frame.groupby(["scene","condition"],as_index=False)[["mma_at_1","mma_at_3","mma_at_5","mma_at_10"]].mean().to_csv(out/"per_scene.csv",index=False)
        write_json(out/"complete.json",{"status":"PASS","protocol_sha256":sha256_file(protocol),"input_sha256":inputs,"pairs_per_condition":85,"test_sequences":17,
            "counts":frame.groupby("condition").match_count.agg(["mean","min","max"]).to_dict()})
        del adapter;gc.collect()


if __name__=="__main__":main()
