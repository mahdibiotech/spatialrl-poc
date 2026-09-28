"""Test de bout en bout sur une config reduite (~30 s) : le pipeline tourne et retrouve la verite terrain."""
import json
import copy
import yaml
import pytest
from spatialrl import simulate, deconv, niche, coregister


@pytest.fixture(scope="module")
def out(tmp_path_factory):
    cfg = yaml.safe_load(open("config.yaml"))
    cfg.update(seed=7)
    cfg["sim"].update(grid=30, n_genes=200, n_ref_per_state=80)
    cfg["msi"].update(n_pix=15)
    cfg["robustness"].update(ref_sizes=[5, 50], n_rep=1)
    cfg["niche"].update(n_perm=100, n_null_pairs=10)
    o = str(tmp_path_factory.mktemp("res"))
    for f in (simulate.run, deconv.run, deconv.robustness, niche.run, coregister.run):
        f(cfg, o)
    return o


def test_deconvolution_recovers_truth(out):
    s = json.load(open(f"{out}/deconv_summary.json"))
    assert s["summary"][s["best_method"]]["mean_r"] > 0.8


def test_platform_effect_helps(out):
    s = json.load(open(f"{out}/deconv_summary.json"))["summary"]
    assert s["poisson_em_plat"]["rmse"] <= s["nnls"]["rmse"]


def test_msi_registration_accurate(out):
    r = json.load(open(f"{out}/msi_registration.json"))
    assert r["TRE_registered_spots"] < 1.5 < r["TRE_unregistered_spots"]


def test_niche_detected_in_tumor(out):
    import pandas as pd
    d = pd.read_csv(f"{out}/niche_ring.csv")
    g3 = d[(d["sample"] == "tumor") & (d.state == "G3_MB_tumor")].iloc[0]
    assert g3.z > 2 and g3.contrast_est > 0
