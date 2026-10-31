"""Recompute published NSI from a small Supporting Information NPZ/JSON pair."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from nsi_garw.config import PipelineConfig,KSGConfig,StatisticsConfig,SubspaceConfig
from nsi_garw.io import sha256_file
from nsi_garw.metrics import estimate_information_purity,mean_finite,normalized_subspace_overlap,r2_per_output


def evaluate(actual_g,actual_a,predicted_g,predicted_a,separation,config):
    rg=mean_finite(r2_per_output(actual_g,predicted_g));ra=mean_finite(r2_per_output(actual_a,predicted_a))
    a=.5*(np.clip(rg,0,1)+np.clip(ra,0,1));c=estimate_information_purity(predicted_g,predicted_a,config).purity
    return {'geometry_r2':float(rg),'appearance_r2':float(ra),'A_accessibility':float(a),
            'B_separation':float(separation),'C_information_purity':float(c),'NSI':float(a*separation*c)}


def main(path,groups=False):
    meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if sha256_file(path)!=meta['output_sha256']:raise ValueError('Compact data SHA changed')
    raw=meta['config'];config=PipelineConfig(seed=raw['seed'],ksg=KSGConfig(**raw['ksg']),
        statistics=StatisticsConfig(**raw['statistics']),subspace=SubspaceConfig(**raw['subspace']))
    d=np.load(path,allow_pickle=False)
    b=1-normalized_subspace_overlap(d['geometry_basis'],d['appearance_raw_basis'],config.subspace.overlap_metric)
    values=evaluate(d['geometry_target'],d['appearance_target'],d['geometry_prediction'],d['appearance_prediction'],b,config)
    for key,value in values.items():
        if not np.isclose(value,meta['result'][key],atol=1e-8,rtol=1e-8):raise ValueError(f'Compact recomputation mismatch:{key}: {value} vs {meta["result"][key]}')
    output={'status':'PASS','source_result':meta['result_path'],'recomputed':values}
    if groups:
        table=[];old={r['group']:r for r in meta['groupwise_uncertainty']['groups']}
        for index,scene in enumerate(np.unique(d['groups'])):
            mask=d['groups']==scene
            r=evaluate(d['geometry_target'][mask],d['appearance_target'][mask],d['geometry_prediction'][mask],d['appearance_prediction'][mask],b,replace(config,seed=config.seed+index+1))
            if not np.isclose(r['NSI'],old[scene]['NSI'],atol=1e-8,rtol=1e-8):raise ValueError(f'Group mismatch:{scene}')
            table.append({'group':str(scene),**r})
        output['groups']=table
    print(json.dumps(output,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('input_npz',type=Path);parser.add_argument('--groups',action='store_true');args=parser.parse_args();main(args.input_npz,args.groups)
