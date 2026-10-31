"""Post-outcome localization-tolerance sensitivity on fixed ETH3D candidates/gates."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from nsi_garw.experiments import write_json
from nsi_garw.garw import garw_scores
from nsi_garw.io import sha256_file
from run_hpatches_suite import DATA,RESULTS


def main():
    folder=RESULTS/"eth3d_garw";rows=[];sources={}
    for identifier in ("loftr_outdoor_coarse_1024","roma_outdoor_scale16_560"):
        root=folder/identifier
        summary=json.loads((root/"non_oracle/garw_summary.json").read_text(encoding="utf-8"))
        source=DATA/"manifests/garw"/("eth3d_"+identifier+"_q32_k16.csv")
        table=pd.read_csv(source);table=table.loc[table.split=="evaluation"].copy()
        for name,key,penalty in (("garw","parameters",None),("garw_lambda_zero","lambda_zero_parameters",0.)):
            p=summary[key]
            table[name]=garw_scores(table.s_geometry,table.s_appearance,table.nsi_local,alpha=p["alpha"],beta=p["beta"],penalty_lambda=p["penalty_lambda"] if penalty is None else penalty)[0]
        table["fixed_weight"]=(table.s_geometry+table.s_appearance)/2
        columns={"descriptor_baseline":"s_baseline","geometry":"s_geometry","appearance":"s_appearance","fixed_weight":"fixed_weight","garw":"garw","garw_lambda_zero":"garw_lambda_zero"}
        for method,column in columns.items():
            winners=table.loc[table.groupby("query_id",sort=False)[column].idxmax()]
            for tolerance in (1,3,5,10,20,40):
                for scene,block in winners.groupby("scene"):
                    pool=table.loc[table.scene==scene]
                    coverage=pool.groupby("query_id").reprojection_error.min().le(tolerance).mean()
                    rows.append({"representation":identifier,"method":method,"scene":scene,"tolerance_original_pixels":tolerance,
                                 "top1":float((block.reprojection_error<=tolerance).mean()),"correct_candidate_coverage":float(coverage),"queries":len(block)})
                if tolerance==3:
                    means=pd.DataFrame(rows).loc[lambda f:(f.representation==identifier)&(f.method==method)&(f.tolerance_original_pixels==3)].top1.mean()
                    if abs(means-summary["summary"][method]["top1"])>1e-12:raise ValueError("Rescoring does not reproduce original3pixel result")
        sources[str(source)]=sha256_file(source);sources[str(root/"non_oracle/garw_summary.json")]=sha256_file(root/"non_oracle/garw_summary.json")
    frame=pd.DataFrame(rows);frame.to_csv(folder/"tolerance_per_scene.csv",index=False)
    frame.groupby(["representation","method","tolerance_original_pixels"],as_index=False)[["top1","correct_candidate_coverage"]].mean().to_csv(folder/"tolerance_summary.csv",index=False)
    write_json(folder/"tolerance_analysis_complete.json",{"status":"PASS","runner_sha256":sha256_file(Path(__file__)),"inputs":sources,
        "scope":"Exploratory after strict3pixel results exposed sparse positive coverage. Fixed candidates,query selection,calibration and gate parameters; no retuning or replacement of original3pixel primary analysis. Original-resolution tolerances1/3/5/10/20/40; curves diagnose localization/candidate-coverage limits, not an independent validation set."})
    print(frame.groupby(["representation","method","tolerance_original_pixels"]).top1.mean().to_string())


if __name__=="__main__":main()
