"""Compact held-out measurements sufficient to recompute A/B/C and group NSI.

Original images and model weights remain public third-party assets. Full native
feature caches are regenerable; these files retain every held-out target and
prediction, fitted directions and training statistics used by the primary NSI.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from nsi_garw.experiments import write_json
from nsi_garw.feature_builders import _atomic_savez
from nsi_garw.io import load_feature_npz,canonical_split_array,sha256_file
from nsi_garw.metrics import fit_ridge_probe,r2_per_output,mean_finite,orthonormal_basis,normalized_subspace_overlap
from run_hpatches_suite import RESULTS

OUT=RESULTS/"supporting_measurements"
DEPENDENCIES=["megadepth/analysis_complete.json","megadepth/pca_complete.json","learned_texture_nsi/texture_analysis_complete.json",
              "eth3d_dense_sensitivity/analysis_complete.json","online_runtime/analysis_complete.json"]


def sources():
    paths=list((RESULTS/"frozen_verification").glob("*.json"))
    for name in ("eth3d","layerwise","sift_degradation_nsi","learned_texture_nsi","megadepth","eth3d_dense_sensitivity","eth3d_extended_ridge"):
        paths.extend((RESULTS/name).glob("*.json"))
    seen=set()
    for path in sorted(paths):
        p=json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(p.get("result"),dict) or 'NSI' not in p['result'] or 'input_path' not in p.get('provenance',{}):continue
        key=(p['provenance']['input_sha256'],p['provenance']['config_sha256'])
        if key in seen:continue
        seen.add(key);yield path,p


def export(path,p):
    provenance=p['provenance'];r=p['result'];config=provenance['config']
    key=hashlib.sha256((provenance['input_sha256']+provenance['config_sha256']).encode()).hexdigest()[:24]
    marker=OUT/(key+'.json');destination=OUT/(key+'.npz')
    if marker.exists():return
    source=Path(provenance['input_path'])
    if sha256_file(source)!=provenance['input_sha256']:raise ValueError(f"Changed primary feature export:{source}")
    d=load_feature_npz(source);split=canonical_split_array(d.split)
    train=split=='train';fit=(split=='train')|(split=='validation');test=split=='test'
    x=d.features[train];xm=x.mean(axis=0);xs=x.std(axis=0,ddof=0)
    probes=[];predictions=[]
    for attribute,alpha in ((d.geometry,r['geometry_alpha']),(d.appearance,r['appearance_alpha'])):
        probe=fit_ridge_probe(d.features[fit],attribute[fit],alpha,x_mean=xm,x_scale=xs,y_mean=attribute[train].mean(axis=0),y_scale=attribute[train].std(axis=0,ddof=0))
        probes.append(probe);predictions.append(probe.predict(d.features[test]))
    sub=config['subspace'];rank=sub['max_rank'];tol=sub['relative_singular_value_tolerance']
    bases=[orthonormal_basis(p.coefficients_standardized,rank,tol) for p in probes]
    observed=[mean_finite(r2_per_output(t[test],pred)) for t,pred in zip((d.geometry,d.appearance),predictions)]
    overlap=normalized_subspace_overlap(*bases,sub['overlap_metric'])
    if not np.allclose(observed+[overlap],[r['geometry_r2'],r['appearance_r2'],r['OSR']],atol=1e-8,rtol=1e-8):
        raise ValueError(f"Supporting measurement reconstruction differs:{path}")
    _atomic_savez(destination,compressed=True,geometry_target=d.geometry[test],appearance_target=d.appearance[test],
        geometry_prediction=predictions[0],appearance_prediction=predictions[1],groups=d.groups[test].astype(str),
        sample_ids=d.sample_ids[test].astype(str) if d.sample_ids is not None else np.flatnonzero(test).astype(str),
        geometry_basis=bases[0],appearance_raw_basis=bases[1],feature_mean=probes[0].x_mean,feature_scale=probes[0].x_scale,
        geometry_coefficients=probes[0].coefficients_standardized,appearance_coefficients=probes[1].coefficients_standardized,
        geometry_y_mean=probes[0].y_mean,geometry_y_scale=probes[0].y_scale,appearance_y_mean=probes[1].y_mean,appearance_y_scale=probes[1].y_scale)
    write_json(marker,{'status':'PASS','result_path':str(path),'result_sha256':sha256_file(path),'result':r,
        'config':config,'groupwise_uncertainty':p['groupwise_uncertainty'],'source_feature_sha256':provenance['input_sha256'],
        'output_sha256':sha256_file(destination),'core_source_sha256_of_original_result':provenance['source_sha256'],
        'reconstructed_r2_and_overlap':observed+[overlap],
        'scope':'Allheldouttargets/predictions/IDsandfittedbases;A,B,CandgroupNSIrecomputablewithoutimageorfullfeaturecache. ValidationtracesandselectedalphasareinresultJSON;completeprobe-fittingregeneratesfeaturesfrompublicimagesandpinnedmodels.'})
    print(f'Supporting NSI measurements:{path.relative_to(RESULTS)}',flush=True)
    del d,x,probes,predictions;gc.collect()


def main(wait=False):
    OUT.mkdir(parents=True,exist_ok=True);deadline=time.monotonic()+36*3600
    while True:
        current=list(sources())
        for path,p in current:export(path,p)
        if all((RESULTS/p).exists() for p in DEPENDENCIES):
            write_json(OUT/'all_complete.json',{'status':'PASS','exporter_sha256':sha256_file(Path(__file__)),
                'source_results':{str(path):sha256_file(path) for path,_ in current},'unique_measurement_sets':len(current)})
            break
        if not wait:break
        if time.monotonic()>deadline:raise TimeoutError('Remaining NSI sources absent36h')
        time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');main(parser.parse_args().wait)
