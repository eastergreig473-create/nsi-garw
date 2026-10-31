"""Compare ten completed exploratory ridge-grid extensions with frozen defaults."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from nsi_garw.io import sha256_file
from run_hpatches_suite import MODELS, RESULTS


def main():
    folder = RESULTS / "eth3d_extended_ridge"
    records, inputs = [], {}
    for identifier, _ in MODELS:
        base_path = RESULTS / "eth3d" / ("eth3d_"+identifier+".json")
        extra_path = folder / ("eth3d_"+identifier+".json")
        if not extra_path.exists():
            print(f"Extended grid incomplete: {identifier}; no final comparison written", flush=True)
            return
        base = json.loads(base_path.read_text(encoding="utf-8"))
        extra = json.loads(extra_path.read_text(encoding="utf-8"))
        if base["provenance"]["input_sha256"] != extra["provenance"]["input_sha256"]:
            raise ValueError("Default/extended feature inputs differ")
        configs = [dict(p["provenance"]["config"]) for p in (base,extra)]
        for config in configs:
            config.pop("ridge")
        if configs[0] != configs[1]:
            raise ValueError("Settings beyond ridge grid changed")
        record = {"representation": identifier}
        for label, payload in (("default",base),("extended",extra)):
            for key in ("NSI", "geometry_r2", "appearance_r2", "geometry_alpha", "appearance_alpha"):
                record[key+"_"+label] = payload["result"][key]
            maximum = max(payload["provenance"]["config"]["ridge"]["lambdas"])
            record["boundary_selections_"+label] = sum(payload["result"][key] == maximum for key in ("geometry_alpha","appearance_alpha"))
        record["NSI_difference"] = record["NSI_extended"]-record["NSI_default"]
        records.append(record)
        for path in (base_path,extra_path):
            inputs[str(path)] = sha256_file(path)
    frame = pd.DataFrame(records)
    frame.to_csv(folder / "extended_ridge_comparison.csv", index=False)
    payload = {"status": "PASS", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope":"Exploratory after viewing default boundary selections; each grid tuned on validation scenes; defaults retained; not prospectively preregistered",
        "inputs":inputs, "rows":records, "script_sha256":sha256_file(Path(__file__)),
        "geometry_positive_default":int((frame.geometry_r2_default>0).sum()),
        "geometry_positive_extended":int((frame.geometry_r2_extended>0).sum()),
        "both_attributes_positive_default":int(((frame.geometry_r2_default>0)&(frame.appearance_r2_default>0)).sum()),
        "both_attributes_positive_extended":int(((frame.geometry_r2_extended>0)&(frame.appearance_r2_extended>0)).sum()),
        "boundary_selections_default":int(frame.boundary_selections_default.sum()),
        "boundary_selections_extended":int(frame.boundary_selections_extended.sum()),
        "NSI_spearman_rho":float(spearmanr(frame.NSI_default,frame.NSI_extended).statistic),
        "NSI_kendall_tau":float(kendalltau(frame.NSI_default,frame.NSI_extended).statistic)}
    (folder / "extended_ridge_comparison.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(frame.to_string(index=False),flush=True)
    print(json.dumps({k:v for k,v in payload.items() if k not in ("rows","inputs")},indent=2),flush=True)


if __name__ == "__main__":
    main()
