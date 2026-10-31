"""All five available HPatches methods, three declared exploratory targets each."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from nsi_garw.experiments import write_json
from nsi_garw.io import sha256_file
from nsi_garw.statistics import benjamini_hochberg
from reanalyze_associations import association
from run_hpatches_suite import RESULTS

OUT=RESULTS/"hpatches_complete_associations"


def scene_nsi(path):
    saved=json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame([{"scene":r["group"],"NSI":r["NSI"]} for r in saved["groupwise_uncertainty"]["groups"]])


def main(wait=False):
    OUT.mkdir(parents=True,exist_ok=True)
    auxiliary=RESULTS/"auxiliary_matching"
    models=[("SIFT","sift_default"),("ResNet-50 layer3","resnet50_layer3_crop224"),
            ("DINOv2 block11","dinov2_vits14_block11_518"),("LoFTR native","loftr"),("RoMa native","roma")]
    protocol=OUT/"protocol.json"
    settings={"runner_sha256":sha256_file(Path(__file__)),"association_helper_sha256":sha256_file(Path(__file__).with_name("reanalyze_associations.py")),
        "models":models,"targets":["cleanMMA3","texture08MMA3","paired_texture_minus_clean_MMA3"],
        "unit":"same17frozenheldoutsequences;fivepairs averaged per sequence;no outcome exclusions",
        "multiplicity":"BH overall15attemptedHPassociations;untestable entriesretainnullwithcorrection-onlyp1",
        "scope":"exploratory familyexpanded after prior SIFT/nativeclean outcomes;no preregistration or predictionvalidation claim;five different matcher/encoder protocols;withinmethod association only",
        "statistics":"Spearman;10000sequencebootstraps;undefineddraws neverimputed;CIwithheld ifn<10or>2.5percentundefined;1000labelpermutations plusone"}
    # JSON converts tuples into lists: compare their canonical serialization.
    if protocol.exists():
        if json.dumps(json.loads(protocol.read_text(encoding="utf-8"))["settings"],sort_keys=True)!=json.dumps(settings,sort_keys=True):raise ValueError("Association protocol changed")
    else:write_json(protocol,{"created_at_utc":datetime.now(timezone.utc).isoformat(),"settings":settings})
    deadline=time.monotonic()+8*3600
    markers=[auxiliary/m/"complete.json" for _,m in models[1:3]]
    while not all(p.exists() for p in markers):
        if not wait:raise FileNotFoundError("Auxiliary matching incomplete")
        if time.monotonic()>deadline:raise TimeoutError("Auxiliary matching absent8h")
        time.sleep(15)
    inputs={str(p):sha256_file(p) for p in markers};rows=[]
    for label,identifier in models:
        if identifier=="sift_default":
            point=RESULTS/"association/sift_clean_sequence_association.csv"
            frame=pd.read_csv(point).rename(columns={"mma_at_3":"cleanMMA3"})
            corrupt=RESULTS/"sift_degradation_matching/sift_degraded_texture_s080/per_pair.csv"
            d=pd.read_csv(corrupt);d=d.loc[d.split=="test"].groupby("scene",as_index=False).mma_at_3.mean().rename(columns={"mma_at_3":"texture08MMA3"})
            frame=frame.merge(d,on="scene",validate="one_to_one");sources=[point,corrupt]
        elif identifier in ("loftr","roma"):
            point=RESULTS/f"native_matchers/{identifier}_nsi_native_MMA3_points.csv"
            frame=pd.read_csv(point)[["scene","NSI","mma_at_3"]].rename(columns={"mma_at_3":"cleanMMA3"})
            corrupt=RESULTS/f"native_matchers/{identifier}_texture_s080/per_pair.csv"
            d=pd.read_csv(corrupt);d=d.loc[d.split=="test"].groupby("scene",as_index=False).mma_at_3.mean().rename(columns={"mma_at_3":"texture08MMA3"})
            frame=frame.merge(d,on="scene",validate="one_to_one");sources=[point,corrupt]
        else:
            nsi=(RESULTS/"layerwise"/f"{identifier}.json") if identifier.startswith("resnet") else RESULTS/f"{identifier}.json"
            source=auxiliary/identifier/"per_scene.csv"
            frame=pd.read_csv(source).pivot(index="scene",columns="condition",values="mma_at_3").reset_index().rename(columns={"clean":"cleanMMA3","texture":"texture08MMA3"})
            frame=scene_nsi(nsi).merge(frame,on="scene",validate="one_to_one");sources=[nsi,source]
        frame=frame.sort_values("scene").reset_index(drop=True)
        if len(frame)!=17 or not np.isfinite(frame[["NSI","cleanMMA3","texture08MMA3"]]).all().all():raise ValueError(f"Incomplete17sequences:{identifier}")
        frame["paired_texture_minus_clean_MMA3"]=frame.texture08MMA3-frame.cleanMMA3
        frame.to_csv(OUT/f"{identifier}_points.csv",index=False)
        inputs.update({str(p):sha256_file(p) for p in sources})
        for target in settings["targets"]:rows.append({"representation":label,"identifier":identifier,"target":target,**association(frame.NSI,frame[target])})
    q=benjamini_hochberg([r["permutation_p_value"] if r["permutation_p_value"] is not None else 1. for r in rows])
    for row,value in zip(rows,q):row["BH_q_15_attempted_associations"]=float(value) if row["permutation_p_value"] is not None else None
    pd.DataFrame(rows).to_csv(OUT/"association_summary.csv",index=False)
    write_json(OUT/"analysis_complete.json",{"status":"PASS","settings":settings,"inputs":inputs,"rows":rows,"protocol_sha256":sha256_file(protocol)})
    print(pd.DataFrame(rows)[["representation","target","spearman_rho","permutation_p_value","BH_q_15_attempted_associations"]].to_string(index=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--wait",action="store_true");main(parser.parse_args().wait)
