"""Matched online encoder-retrieval latency and GARW overhead, FP32 vs CUDA AMP."""
import argparse
from dataclasses import fields
from datetime import datetime,timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
import psutil
from scipy.ndimage import map_coordinates

from nsi_garw.attributes import sobel_magnitude
from nsi_garw.cli import _batch_adapter,build_parser
from nsi_garw.experiments import write_json,source_sha256
from nsi_garw.feature_builders import _deterministic_subset
from nsi_garw.garw import standardized_projected_cosine,garw_scores
from nsi_garw.io import load_feature_npz,sha256_file
from nsi_garw.local_proxy import LocalNSIProxyModel
from nsi_garw.metrics import fit_ridge_probe,orthonormal_basis,residualized_appearance_basis
from nsi_garw.runtime import measure_call
from run_hpatches_suite import DATA,MODELS,RESULTS,ROOT

FOLDER=RESULTS/"online_runtime"
IDENTIFIERS=("loftr_outdoor_coarse_1024","roma_outdoor_scale16_560")


def calibration(identifier):
    folder=RESULTS/"garw_hpatches"/(identifier+("_cuda_amp" if identifier.startswith("roma") else ""))
    summary=json.loads((folder/"non_oracle/garw_summary.json").read_text(encoding="utf-8"))
    record=json.loads((folder/"proxy.json").read_text(encoding="utf-8"))["model"]
    values={field.name:record[field.name] for field in fields(LocalNSIProxyModel)}
    for key in ("x_mean","x_scale","y_mean","y_scale","coefficients_standardized"):values[key]=np.asarray(values[key])
    values["feature_columns"]=tuple(values["feature_columns"])
    proxy=LocalNSIProxyModel(**values)
    data=load_feature_npz(DATA/"manifests/features"/(identifier+".npz"))
    default=json.loads((RESULTS/(identifier+".json")).read_text(encoding="utf-8"))["result"]
    train=data.split=="train"
    gp=fit_ridge_probe(data.features[train],data.geometry[train],default["geometry_alpha"])
    ap=fit_ridge_probe(data.features[train],data.appearance[train],default["appearance_alpha"])
    ug=orthonormal_basis(gp.coefficients_standardized,8,1e-6)
    ua=residualized_appearance_basis(ug,ap.coefficients_standardized,8,1e-6)
    return gp,ug,ua,proxy,summary["parameters"],{
        "summary_sha256":sha256_file(folder/"non_oracle/garw_summary.json"),"proxy_sha256":sha256_file(folder/"proxy.json")}


def online_pair(adapter,first_image,second_image,pair_id,model,method,precision):
    import torch
    gp,ug,ua,proxy,parameters,_=model
    # RoMa's encoder owns its autocast context; LoFTR uses this outer context.
    enabled=precision=="AMP_FP16" and not hasattr(adapter,"encoder")
    with torch.autocast("cuda",dtype=torch.float16,enabled=enabled):
        first,second=adapter.extract(first_image),adapter.extract(second_image)
    chosen=_deterministic_subset(len(first.descriptors),32,seed=20261023,pair_id=pair_id)
    x,y=first.descriptors[chosen].astype(np.float32),second.descriptors.astype(np.float32)
    x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    y/=np.maximum(np.linalg.norm(y,axis=1,keepdims=True),1e-12)
    similarity=x@y.T
    k=min(16,len(y));candidates=np.argsort(-similarity,axis=1,kind="stable")[:,:k]
    raw=np.take_along_axis(similarity,candidates,axis=1)
    if method=="raw_descriptor":scores=raw
    else:
        q=np.repeat(chosen,k);t=candidates.ravel()
        sg=standardized_projected_cosine(first.descriptors[q],second.descriptors[t],ug,gp.x_mean,gp.x_scale).reshape(len(chosen),k)
        sa=standardized_projected_cosine(first.descriptors[q],second.descriptors[t],ua,gp.x_mean,gp.x_scale).reshape(len(chosen),k)
        texture=map_coordinates(sobel_magnitude(first_image),[first.coordinates_xy[chosen,1],first.coordinates_xy[chosen,0]],order=1,mode="nearest")
        observable=np.column_stack([texture,np.abs(sg-sa).mean(axis=1),raw[:,0]-raw[:,min(1,k-1)]])
        local=proxy.predict(observable)
        scores=garw_scores(sg.ravel(),sa.ravel(),np.repeat(local,k),alpha=parameters["alpha"],beta=parameters["beta"],penalty_lambda=parameters["penalty_lambda"])[0].reshape(len(chosen),k)
    indices=candidates[np.arange(len(chosen)),np.argmax(scores,axis=1)]
    return {"source_xy":first.coordinates_xy[chosen],"target_xy":second.coordinates_xy[indices],
            "native_source_count":len(first.descriptors),"native_target_count":len(second.descriptors),"queries":len(chosen)}


def wait_gpu_export():
    marker=RESULTS/"megadepth/gpu_exports_complete.json"
    offline_markers=[RESULTS/name for name in ('megadepth/cpu_exports_complete.json','megadepth/analysis_complete.json',
        'eth3d_dense_sensitivity/analysis_complete.json','supporting_measurements/all_complete.json')]
    worker_names=('run_megadepth_revision.py','run_eth3d_dense_sensitivity.py','export_nsi_supporting_measurements.py','run_eth3d_colmap_dense.py')
    deadline=time.monotonic()+48*3600
    while True:
        running=False
        for process in psutil.process_iter(["cmdline"]):
            try:
                args=process.info["cmdline"] or []
                if any(any(name in p for name in worker_names) for p in args):running=True
            except (psutil.NoSuchProcess,psutil.AccessDenied):continue
        if marker.exists() and all(p.exists() for p in offline_markers) and not running:return
        if time.monotonic()>deadline:raise TimeoutError("Offline GPU/CPU fitting and export queue did not finish within48h")
        time.sleep(20)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--wait",action="store_true");parser.add_argument("--validate-inputs",action="store_true");args=parser.parse_args()
    FOLDER.mkdir(parents=True,exist_ok=True)
    pairs=pd.read_csv(DATA/"manifests/hpatches/hpatches_pairs.csv")
    pairs=pairs.loc[pairs.split=="validation"].copy()
    pairs["key"]=[hashlib.sha256(("20261023:runtime:"+s).encode()).hexdigest() for s in pairs.pair_id]
    pairs=pairs.sort_values("key").head(3)
    models={identifier:calibration(identifier) for identifier in IDENTIFIERS}
    if args.validate_inputs:
        for identifier,model in models.items():
            assert np.isfinite(model[3].predict(np.asarray([[.1,.2,.3]]))).all()
            print(f"Runtime inputs verified:{identifier}; actuallambda={model[4]['penalty_lambda']}",flush=True)
        return
    settings={"runner_sha256":sha256_file(Path(__file__)),"core_source_sha256":source_sha256(),"pair_ids":pairs.pair_id.tolist(),
        "calibration":{identifier:model[-1] for identifier,model in models.items()},"warmups_per_model_precision_method":3,"repeats_per_pair":5,
        "precision":"Fresh CPU FP32 checkpoint load for each mode; transfer originals toCUDA FP32 or officialRoMaAMP FP16; LoFTR outerCUDAautocastFP16. TF32 disabled in both modes.",
        "boundary":"PredecodedRGBpair -> extraction ofbothimages -> CPUdescriptorreadout -> uniform32queries -> normalizeddescriptor top16retrieval -> optional standardized projections/Sobelobservableproxy/GARW -> top1coordinates. Model loading, offline probe/proxy fitting, diskIO andGTlabels excluded. DecodeIO separately timed. Baseline uses same top16proposal, not an optimized argmax-only implementation.",
        "measurement":"CUDA sync before/after; processCPUtime,10msRSSsampler baseline/peak/delta,tracemallocPythonpeak,totalpeakCUDAallocation including model. Repeated-call median/p95 are descriptive over3fixedvalidationpairs, not scene-level inferential intervals.",
        "scope":"Encoder candidate pipeline overhead, not full nativeLoFTR/RoMa matcher overhead. Withinmodel precision/method comparisons have matched inputs; inputresolutions differ betweenmodels."}
    settings['queue_isolation']='Before any runtime outcome, wait for all MVS, MegaDepth GPU/CPU/NSI, dense-source NSI, and compact-NSI export workers to finish. This avoids concurrent offline CPU fitting as well as CUDA contention. V1 preserved; numerical inputs and benchmark boundary unchanged.'
    protocol=DATA/"manifests/hpatches/online_runtime_protocol_v2.json"
    if protocol.exists():
        if json.loads(protocol.read_text(encoding="utf-8"))["settings"]!=settings:raise ValueError("Frozen runtime protocol changed")
    else:write_json(protocol,{"created_at_utc":datetime.now(timezone.utc).isoformat(),"settings":settings})
    if args.wait:wait_gpu_export()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    import torch
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    decoded={};io=[]
    image_root=DATA/"datasets/HPatches/hpatches-sequences-release"
    for _,row in pairs.iterrows():
        start=time.perf_counter()
        decoded[row.pair_id]=(np.asarray(Image.open(image_root/row.reference_image).convert("RGB")),np.asarray(Image.open(image_root/row.target_image).convert("RGB")))
        io.append({"pair_id":row.pair_id,"decode_IO_seconds":time.perf_counter()-start})
    records=[];outputs={};rng=np.random.default_rng(20261023)
    for identifier in IDENTIFIERS:
        arguments=dict(MODELS)[identifier]
        for precision in ("FP32","AMP_FP16"):
            parsed=build_parser().parse_args(["build-hpatches-model",*arguments,"--export-id",identifier,"--pairs-manifest",str(DATA/"manifests/hpatches/hpatches_pairs.csv"),"--dataset-root",str(image_root),"--cache-root",str(DATA/"cache"),"--output",str(FOLDER/"unused.npz"),"--device","cpu"])
            adapter=_batch_adapter(parsed);adapter.device=torch.device("cuda")
            if hasattr(adapter,"encoder"):
                adapter.encoder.cnn.to("cuda")
                adapter.encoder.amp=precision=="AMP_FP16"
                adapter.encoder.amp_dtype=torch.float16 if adapter.encoder.amp else torch.float32
                adapter.encoder.dinov2_vitl14[0]=adapter.encoder.dinov2_vitl14[0].to(device="cuda",dtype=adapter.encoder.amp_dtype)
            else:adapter.model.to("cuda")
            model=models[identifier]
            first_pair=pairs.iloc[0].pair_id
            for method in ("raw_descriptor","GARW"):
                for _ in range(3):online_pair(adapter,*decoded[first_pair],first_pair,model,method,precision)
            torch.cuda.synchronize()
            for _,pair in pairs.iterrows():
                for repeat in range(5):
                    for method in rng.permutation(["raw_descriptor","GARW"]):
                        output,runtime=measure_call(online_pair,adapter,*decoded[pair.pair_id],pair.pair_id,model,str(method),precision)
                        records.append({"representation":identifier,"precision":precision,"method":str(method),"pair_id":pair.pair_id,"repeat":repeat,
                                        **runtime.to_dict(),"queries":output["queries"],"native_source_count":output["native_source_count"],"native_target_count":output["native_target_count"]})
                        outputs.setdefault((identifier,precision,str(method),pair.pair_id),output["target_xy"])
            del adapter;gc.collect();torch.cuda.empty_cache()
            print(f"Online runtime complete:{identifier} {precision}",flush=True)
    frame=pd.DataFrame(records);frame.to_csv(FOLDER/"per_call.csv",index=False)
    pd.DataFrame(io).to_csv(FOLDER/"decode_IO.csv",index=False)
    rows=[]
    for (identifier,precision,method),block in frame.groupby(["representation","precision","method"]):
        rows.append({"representation":identifier,"precision":precision,"method":method,"calls":len(block),"wall_median_seconds":float(block.wall_seconds.median()),
            "wall_p95_seconds":float(block.wall_seconds.quantile(.95)),"CPU_median_seconds":float(block.process_cpu_seconds.median()),
            **{key+"_max":float(block[key].max()) for key in ("python_peak_megabytes","rss_peak_megabytes","rss_delta_peak_megabytes","cuda_peak_megabytes")}})
    summary=pd.DataFrame(rows);summary.to_csv(FOLDER/"summary.csv",index=False)
    differences=[]
    for identifier in IDENTIFIERS:
        for method in ("raw_descriptor","GARW"):
            for pair_id in pairs.pair_id:
                first=outputs[identifier,"FP32",method,pair_id];second=outputs[identifier,"AMP_FP16",method,pair_id]
                differences.append({"representation":identifier,"method":method,"pair_id":pair_id,"selected_target_disagreement_fraction":float(np.mean(np.linalg.norm(first-second,axis=1)>1e-6))})
    pd.DataFrame(differences).to_csv(FOLDER/"precision_output_disagreement.csv",index=False)
    write_json(FOLDER/"analysis_complete.json",{"status":"PASS","protocol_sha256":sha256_file(protocol),"summary":rows,
        "per_call_sha256":sha256_file(FOLDER/"per_call.csv"),"precision_output_disagreement":differences,"scope":settings["boundary"],"measurement":settings["measurement"]})
    print(summary.to_string(index=False),flush=True)


if __name__=="__main__":main()
