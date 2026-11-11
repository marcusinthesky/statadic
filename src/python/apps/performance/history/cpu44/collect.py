"""Validate all paired numeric outputs, and summarize whole-process replicate results."""
import importlib
import json
import sys
from pathlib import Path
from statistics import median

import numpy as np
from econ_meta import META

HERE = Path(__file__).resolve().parent
SHA = "689d10a0e933cc8adb137289878f5bd31c931d28"
PREFIX = f"https://github.com/marcusinthesky/statadic/blob/{SHA}/"
GROUPS = [
    ("spatial_cases", "sar sem sar_durbin spatial_iv gmm_error_hom gmm_combo_het panel_fe_sar kkp"),
    ("scale_fit", "sar160"),
    ("survival_cases", "weibull-noentry lognormal-noentry weibull-entry lognormal-entry weibull-unit"),
    ("scale_fit", "weibull2400"),
    ("econometrics","panel_fe panel_dk panel_twoway panel_pooled panel_re panel_between panel_fd panel_fmb iv_gmm panel_fe_2000"),
    ("econometrics","ols_hc3 ols_hc3_10000 poisson_hc0 ols_hc3_jit ols_hc3_10000_jit"),
    ("more_refs","glum-poisson"),
    ("other_refs","energy-fixture energy-scaled w1-fixture w1-scaled"),
    ("pot_control","w1-fixture w1-scaled"),
    ("other_refs","cdist-fixture cdist-scaled pdist-fixture pdist-scaled"),
    ("more_refs","arch-bartlett skbio-mantel pypfopt-sample"),
]


def metadata(module, case):
    if module == "econometrics":
        m = META[case]
        return dict(method=m["name"], reference=m["reference"], fixture=m["size"], work=m["workload"],
                    note=m.get("note",""), source=m["source"], category=m["category"], rtol=m["rtol"], atol=m["atol"])
    m = dict(importlib.import_module(module).CASES[case])
    m["category"] = ("POT 1D control" if module=="pot_control" else
                     "Scale check" if case.endswith("-scaled") or module=="scale_fit" else "Fixture")
    if "#" not in m["source"]:
        anchors={"sar":"61-L99","sem":"61-L99","sar_durbin":"61-L99","spatial_iv":"148-L190",
                 "gmm_error_hom":"34-L102","gmm_combo_het":"34-L102","kkp":"49-L132","panel_fe_sar":"120-L175",
                 "glum-poisson":"660-L686"}
        anchor=anchors.get(case, "176-L245" if case.endswith("-unit") else "66-L134")
        m["source"]+="#L"+anchor
    m["source"]=PREFIX+m["source"]
    if module=="survival_cases" and case.endswith("-unit"):
        m["note"]+=" Panel data and fixed-effect design prepared before timing."
    return m


def collect():
    cases=[]
    missing=[]
    for module, keys in GROUPS:
        for case in keys.split():
            meta=metadata(module,case)
            trials=[]
            for i in range(3):
                stems=[HERE/"results"/f"{module}.{case}.{s}.{i}" for s in ("statadic","reference")]
                files=[Path(str(s)+".json") for s in stems]
                if not all(f.exists() for f in files):
                    missing.append((module,case,i)); continue
                j,r=[json.loads(f.read_text()) for f in files]
                a,b=[np.load(str(s)+".npy") for s in stems]
                if a.shape!=b.shape or not np.allclose(a,b,rtol=meta["rtol"],atol=meta["atol"]):
                    raise ValueError((module,case,i,a.shape,b.shape, np.max(abs(a-b))))
                trials.append(dict(trial=i+1, j=j, r=r, maxerr=float(np.max(abs(a-b))), size=int(a.size)))
            if not trials:
                continue
            summary={}
            for side in ["j","r"]:
                summary[side+"warm"]=median(median(t[side]["ms"][1:]) for t in trials)
                summary[side+"first"]=median(t[side]["ms"][0] for t in trials)
                summary[side+"rss"]=median(t[side]["rss_kib"]/1024 for t in trials)
            ratios=[median(t["j"]["ms"][1:])/median(t["r"]["ms"][1:]) for t in trials]
            summary["minratio"]=min(ratios)
            summary["maxratio"]=max(ratios)
            cases.append(dict(id=f"{module}/{case}",**meta,summary=summary,trials=trials))
    return cases,missing


if __name__=="__main__":
    cases, missing=collect()
    (HERE/"summary.json").write_text(json.dumps(cases))
    print(len(cases),"workloads represented;",len(missing),"pending paired trials",missing[:40])
    if "--strict" in sys.argv and missing:
        sys.exit(1)
