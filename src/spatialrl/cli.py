"""CLI : python -m spatialrl.cli {simulate,deconv,robustness,niche,coregister,report,all} --config config.yaml"""
import argparse
import os
import yaml
from . import simulate, deconv, niche, coregister


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step")
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out = cfg["outdir"]
    os.makedirs(out, exist_ok=True)
    steps = {"simulate": simulate.run, "deconv": deconv.run, "robustness": deconv.robustness,
             "niche": niche.run, "coregister": coregister.run}
    if a.step in ("report", "all"):
        from . import report
        steps["report"] = report.run
    order = ["simulate", "deconv", "robustness", "niche", "coregister", "report"]
    for s in (order if a.step == "all" else [a.step]):
        steps[s](cfg, out)


if __name__ == "__main__":
    main()
