"""Separate empirical metric equality from equality of calibrated gate parameters."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from nsi_garw.experiments import write_json
from nsi_garw.io import sha256_file
from run_hpatches_suite import RESULTS


def main():
    rows=[];inputs={}
    for condition,folder in [("LoFTR clean","loftr_outdoor_coarse_1024"),("RoMa clean","roma_outdoor_scale16_560_cuda_amp"),
                             ("LoFTR texture0.8","loftr_outdoor_coarse_1024_texture_s080"),("RoMa texture0.8","roma_outdoor_scale16_560_texture_s080_cuda_amp")]:
        root=RESULTS/"garw_hpatches"/folder/"non_oracle"
        p=json.loads((root/"garw_summary.json").read_text(encoding="utf-8"))
        per_query=pd.read_csv(root/"garw_per_query.csv")
        row={"condition":condition,"selected_penalty_lambda":p["parameters"]["penalty_lambda"],
             "full_gate_alpha":p["parameters"]["alpha"],"full_gate_beta":p["parameters"]["beta"],
             "zero_penalty_gate_alpha":p["lambda_zero_parameters"]["alpha"],"zero_penalty_gate_beta":p["lambda_zero_parameters"]["beta"]}
        for metric in ("top1","average_precision","mrr"):
            pivot=per_query.pivot(index="query_id",columns="method",values=metric)
            difference=(pivot.garw-pivot.garw_lambda_zero).to_numpy()
            row[metric+"_different_queries"]=int(np.sum(np.abs(difference)>1e-12))
            row[metric+"_maximum_difference"]=float(np.max(np.abs(difference)))
        rows.append(row)
        for name in ("garw_summary.json","garw_per_query.csv"):inputs[str(root/name)]=sha256_file(root/name)
    folder=RESULTS/"garw_hpatches"
    pd.DataFrame(rows).to_csv(folder/"calibration_parameter_audit.csv",index=False)
    write_json(folder/"calibration_parameter_audit.json",{"status":"PASS","rows":rows,"inputs":inputs,
        "scope":"Equal observed test metrics do not imply equal score functions or a selected zero penalty. This audit does not retune any parameter or change prior outcomes."})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=="__main__":main()
