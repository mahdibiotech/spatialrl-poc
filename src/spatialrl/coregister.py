"""Recalage MSI (metabolomique) -> grille spatiale, puis association ions x etats cellulaires.

Ici l'ancre est une relation a priori ion <-> etat (ex. un lipide du GC_neuron).
Sur vraies donnees : utiliser H&E / coupe adjacente / coloration post-MSI comme reference de recalage.
"""
import json
import numpy as np
import pandas as pd
from scipy import ndimage, optimize
from .geometry import spot_to_msi

ANCHORS = {0: "GC_neuron", 1: "Endothelial"}  # ion -> etat attendu


def _sample(ion, P, theta, t, L, M):
    Q = spot_to_msi(P, theta, t, L, M)
    return ndimage.map_coordinates(ion, Q.T, order=1, mode="nearest")


def _cost(par, ions_s, targets, P, L, M):
    c = 0.0
    for i, tgt in targets.items():
        v = _sample(ions_s[i], P, par[0], par[1:], L, M)
        c -= np.corrcoef(v, tgt)[0, 1]
    return c


def register(ions, targets, L, M):
    ions_s = [ndimage.gaussian_filter(x, 1.0) for x in ions]
    P = np.stack(np.meshgrid(np.arange(L), np.arange(L), indexing="ij"), -1).reshape(-1, 2).astype(float)
    best = min(((_cost([np.deg2rad(th), a, b], ions_s, targets, P, L, M), th, a, b)
                for th in np.arange(-20, 20.1, 2.5) for a in np.arange(-4, 4.1, 1.0) for b in np.arange(-4, 4.1, 1.0)))
    r = optimize.minimize(_cost, [np.deg2rad(best[1]), best[2], best[3]], args=(ions_s, targets, P, L, M),
                          method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-6))
    return r.x, P


def tre(par_est, par_true, P, L, M):
    a = spot_to_msi(P, par_est[0], par_est[1:], L, M)
    b = spot_to_msi(P, par_true[0], par_true[1:], L, M)
    return float(np.linalg.norm(a - b, axis=1).mean() / 0.5)  # en unites de spots


def ion_state_corr(ions, E, par, P, L, M):
    """Correlation (n_ions x K) apres reechantillonnage des ions sur la grille spatiale."""
    R = np.stack([_sample(ions[i], P, par[0], par[1:], L, M) for i in range(len(ions))])
    K = E.shape[1]
    return np.array([[np.corrcoef(R[i], E[:, k])[0, 1] for k in range(K)] for i in range(len(ions))])


def run(cfg, outdir):
    d = np.load(f"{outdir}/sim.npz")
    est = np.load(f"{outdir}/deconv_est.npz")
    best = json.load(open(f"{outdir}/deconv_summary.json"))["best_method"]
    L, M = cfg["sim"]["grid"], cfg["msi"]["n_pix"]
    states = list(d["states"])
    E = est[f"fetal_{best}"]
    targets = {i: ndimage.gaussian_filter(E[:, states.index(s)].reshape(L, L), 1.0).ravel() for i, s in ANCHORS.items()}
    par, P = register(d["msi"], targets, L, M)
    par_true = np.array([np.deg2rad(cfg["msi"]["theta_deg"]), *cfg["msi"]["shift_px"]])
    par_id = np.array([0.0, 0.0, 0.0])
    C = {k: ion_state_corr(d["msi"], E, p, P, L, M) for k, p in
         (("registered", par), ("oracle", par_true), ("unregistered", par_id))}
    np.savez_compressed(f"{outdir}/msi_corr.npz", **C)
    expect = {"ion1~GC_neuron": (0, 3), "ion2~Endothelial": (1, 5), "ion3~RL_stem": (2, 0), "ion4(bruit)~max|r|": (3, None)}
    rows = []
    for name, (i, k) in expect.items():
        row = {"couple": name}
        for key, M_ in C.items():
            row[key] = float(M_[i, k]) if k is not None else float(np.abs(M_[i]).max())
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"{outdir}/msi_summary.csv", index=False)
    out = dict(theta_est_deg=float(np.rad2deg(par[0])), shift_est=[float(x) for x in par[1:]],
               theta_true_deg=cfg["msi"]["theta_deg"], shift_true=cfg["msi"]["shift_px"],
               TRE_registered_spots=tre(par, par_true, P, L, M), TRE_unregistered_spots=tre(par_id, par_true, P, L, M))
    json.dump(out, open(f"{outdir}/msi_registration.json", "w"), indent=1)
    print("[coregister]", json.dumps(out))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
