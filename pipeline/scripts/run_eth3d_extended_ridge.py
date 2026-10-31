"""Validation-only extended-ridge sensitivity after boundary alpha selections."""
from dataclasses import replace
import json
from pathlib import Path

from nsi_garw.config import load_config
from nsi_garw.experiments import compute_nsi_artifacts
from run_hpatches_suite import DATA, RESULTS, ROOT, MODELS


def main():
    base=load_config(ROOT / "configs/default.toml")
    grid=tuple(sorted(set(base.ridge.lambdas) | {1000.,10000.,100000.,1000000.}))
    config=replace(base,ridge=replace(base.ridge,lambdas=grid))
    output=RESULTS / "eth3d_extended_ridge"
    output.mkdir(exist_ok=True)
    for identifier,_ in MODELS:
        source=DATA / "manifests/features" / ("eth3d_"+identifier+".npz")
        result=output / ("eth3d_"+identifier+".json")
        if result.exists():
            continue
        print(f"Extended ridge START: {identifier}",flush=True)
        payload=compute_nsi_artifacts(source,result,config)
        payload["sensitivity_scope"]={"rationale":"default validation-selected alpha frequently lies at maximum100; extend grid without using test scores for tuning",
            "status":"exploratory sensitivity after default outcomes were examined; preserve default analyses separately",
            "ridge_grid":grid,"split":"same5/3/5 frozen scenes; normalization and all other settings unchanged"}
        result.write_text(json.dumps(payload,indent=2),encoding="utf-8")
        r=payload["result"]
        print(f"Extended ridge DONE: {identifier}; NSI={r['NSI']:.6g}; geometry_alpha={r['geometry_alpha']}; appearance_alpha={r['appearance_alpha']}",flush=True)


if __name__=="__main__":
    main()
