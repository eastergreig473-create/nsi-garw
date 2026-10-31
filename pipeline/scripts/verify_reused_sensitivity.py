"""Verify every reused early sensitivity configuration with the frozen core."""
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nsi_garw.config import load_config
from nsi_garw.experiments import run_sensitivity,source_sha256,write_json
from nsi_garw.io import sha256_file
from run_hpatches_suite import ROOT,RESULTS


def main():
    old=RESULTS/'sensitivity';out=RESULTS/'frozen_verification/sensitivity';out.mkdir(parents=True,exist_ok=True)
    reuse=json.loads((old/'completion_provenance.json').read_text(encoding='utf-8'))['reused_artifacts']
    manifest=pd.read_csv(ROOT/'manifests/hpatches_nsi_manifest.csv').set_index('experiment_id')
    source=source_sha256();config=load_config(ROOT/'configs/default.toml');reports=[]
    columns=['A_accessibility','B_separation','C_information_purity','NSI','OSR','geometry_r2','appearance_r2','geometry_rank','appearance_raw_rank']
    for item in reuse:
        identifier=Path(item['file']).stem;marker=out/(identifier+'.verification.json')
        if marker.exists():
            p=json.loads(marker.read_text(encoding='utf-8'))
            if p['core_source_sha256']!=source or p['original_table_sha256']!=sha256_file(Path(item['file'])):raise ValueError('Sensitivity verification inputs changed')
            reports.append(p);continue
        original=pd.read_csv(item['file']);input_path=Path(manifest.loc[identifier,'input_npz'])
        digest=sha256_file(input_path)
        if set(original.input_sha256)!={digest}:raise ValueError('Reused sensitivity feature export changed')
        print(f'Frozen sensitivity verification START:{identifier} (17 configurations)',flush=True)
        fresh=run_sensitivity(input_path,out/(identifier+'.csv'),config)
        a=original.set_index('config_sha256').sort_index();b=fresh.set_index('config_sha256').sort_index()
        if list(a.index)!=list(b.index):raise ValueError('Sensitivity configuration family changed')
        first,second=a[columns].to_numpy(dtype=float),b[columns].to_numpy(dtype=float)
        delta=float(np.max(np.abs(first-second)))
        if not np.allclose(first,second,atol=1e-9,rtol=1e-9):raise ValueError(f'Reused sensitivity differs under stable source:{identifier}; maximum={delta}')
        if source_sha256()!=source:raise ValueError('Core changed during sensitivity verification')
        p={'status':'PASS','representation':identifier,'configurations':len(fresh),'max_absolute_difference':delta,
           'core_source_sha256':source,'original_table_sha256':sha256_file(Path(item['file'])),'input_sha256':digest,
           'fresh_table_sha256':sha256_file(out/(identifier+'.csv'))}
        write_json(marker,p);reports.append(p)
        print(f'Frozen sensitivity verification PASS:{identifier};max_difference={delta:.6g}',flush=True);gc.collect()
    write_json(out/'all_verified.json',{'status':'PASS','representations':len(reports),'configurations':sum(r['configurations'] for r in reports),
        'rows':reports,'core_source_sha256':source,'script_sha256':sha256_file(Path(__file__)),
        'scope':'All six reused early 17-configuration sensitivity tables independently recomputed under the frozen core; four later tables already have this source in completion provenance. Fixed explicit scene labels; seed variants affect estimator randomness,not new scene splits.'})


if __name__=='__main__':main()
