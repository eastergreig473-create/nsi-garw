"""Publication association audit: undefined correlations never become measured zeros."""
from itertools import permutations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from nsi_garw.experiments import write_json
from nsi_garw.io import sha256_file
from nsi_garw.statistics import benjamini_hochberg
from run_hpatches_suite import MODELS,RESULTS

FOLDER=RESULTS/"association_audit"


def rho(first,second):
    a=rankdata(first).astype(float);b=rankdata(second).astype(float)
    a-=a.mean();b-=b.mean();denominator=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denominator) if denominator>1e-12 else None


def association(first,second,seed=20261023,replicates=10000):
    a,b=np.asarray(first,dtype=float),np.asarray(second,dtype=float)
    observed=rho(a,b)
    n=len(a)
    base={"group_count":n,"spearman_rho":observed,"bootstrap_replicates_requested":replicates}
    if observed is None:
        return {**base,"status":"NOT_ESTIMABLE_CONSTANT_VECTOR","permutation_p_value":None,
                "confidence_interval":None,"bootstrap_undefined_count":replicates,
                "CI_status":"notdefinedforconstantobservedvector"}
    rng=np.random.default_rng(seed)
    selected=rng.integers(0,n,size=(replicates,n))
    x=rankdata(a[selected],axis=1);y=rankdata(b[selected],axis=1)
    x-=x.mean(axis=1,keepdims=True);y-=y.mean(axis=1,keepdims=True)
    denominator=np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)
    valid=denominator>1e-12
    boot=np.sum(x[valid]*y[valid],axis=1)/denominator[valid]
    undefined=int((~valid).sum())
    conditional=np.quantile(boot,[.025,.975]).tolist() if len(boot) else None
    # Conservative publication rule, stated for every analysis: no bootstrap
    # confidence claim with <10 groups or >2.5% undefined resamples.
    ci=conditional if n>=10 and undefined/replicates<=.025 else None
    if n<=8:
        null=[rho(a,b[list(order)]) for order in permutations(range(n))]
        p=sum(abs(v)>=abs(observed)-1e-12 for v in null)/len(null)
        method=f"exact all{len(null)}labelpermutations;tiesincluded"
        permutation_count=len(null)
    else:
        null=[rho(a,b[rng.permutation(n)]) for _ in range(1000)]
        p=(1+sum(abs(v)>=abs(observed)-1e-12 for v in null))/1001
        method="1000MonteCarlo labelpermutations;plusone correction;tiesincluded"
        permutation_count=1000
    return {**base,"status":"ESTIMATED_EXPLORATORY","permutation_p_value":float(p),"permutation_method":method,
            "permutation_replicates":permutation_count,"confidence_interval":ci,
            "CI_status":"95percentileclusterbootstrap" if ci else "withheld:fewerthan10groupsorover2.5percentundefineddraws",
            "bootstrap_undefined_count":undefined,"bootstrap_valid_count":int(valid.sum()),
            "conditional_nonconstant_resample_interval_for_sensitivity_only":conditional}


def main():
    FOLDER.mkdir(parents=True,exist_ok=True)
    rows=[];inputs={}
    hp=[("SIFT",RESULTS/"association/sift_clean_sequence_association.csv"),
        ("LoFTR",RESULTS/"native_matchers/loftr_nsi_native_MMA3_points.csv"),
        ("RoMa",RESULTS/"native_matchers/roma_nsi_native_MMA3_points.csv")]
    for name,path in hp:
        frame=pd.read_csv(path).sort_values("scene")
        rows.append({"dataset":"HPatches","representation":name,"target":"cleanMMA3",**association(frame.NSI,frame.mma_at_3)})
        inputs[str(path)]=sha256_file(path)
    native=[r for r in rows if r["representation"]!="SIFT"]
    for row,q in zip(native,benjamini_hochberg([r["permutation_p_value"] for r in native])):row["BH_q_two_native_associations"]=float(q)
    eth=[]
    for identifier,_ in MODELS:
        path=RESULTS/"eth3d_garw"/identifier/"association_points.csv"
        frame=pd.read_csv(path).sort_values("scene")
        eth.append({"dataset":"ETH3D","representation":identifier,"target":"raw32queryforwardTop1at3originalpixels",**association(frame.NSI,frame.accuracy_3)})
        inputs[str(path)]=sha256_file(path)
    # Untestable entries remain null. A correction-only value1 keeps all ten
    # attempted hypotheses in the conservative family without inventing a test.
    adjusted=benjamini_hochberg([r["permutation_p_value"] if r["permutation_p_value"] is not None else 1. for r in eth])
    for row,q in zip(eth,adjusted):
        row["BH_q_ten_attempted_ETH_associations"]=float(q) if row["permutation_p_value"] is not None else None
    rows+=eth
    pd.DataFrame(rows).to_csv(FOLDER/"association_summary.csv",index=False)
    write_json(FOLDER/"analysis_complete.json",{"status":"PASS","runner_sha256":sha256_file(Path(__file__)),"rows":rows,"inputs":inputs,
        "scope":"Exploratory statistical correction after inspecting constant-vector fallback in frozen generic core. Observed constant vectors are untestable; undefined bootstrap draws are counted, never assignedzero. Confidence intervals withheld uniformly for n<10 or >2.5percentundefineddraws. Conditional nonconstant-resample ranges are sensitivity outputs only. ETH5scene exact permutation usesall120arrangements;BH uses10attemptedtests conservatively. HP17 uses1000permutations and same original family separation.",
        "core_NSI_values_changed":False})
    print(pd.DataFrame([{k:r.get(k) for k in ("dataset","representation","spearman_rho","permutation_p_value","bootstrap_undefined_count","CI_status")} for r in rows]).to_string(index=False))


if __name__=="__main__":main()
