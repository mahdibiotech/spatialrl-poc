"""Deconvolution spatiale : 3 methodes comparees + robustesse a la taille de la reference.

Methodes (implementations maison, sans dependance lourde, pour comparer sur verite terrain) :
  - nnls            : moindres carres non negatifs pondérés (baseline classique)
  - poisson_em      : melange Poisson par EM (esprit RCTD, sans effet de plateforme)
  - poisson_em_plat : idem + effet de plateforme par gene estime iterativement (esprit RCTD)
Sur vraies donnees, remplacer/completer par cell2location, RCTD, Tangram (voir docs/).
"""
import json
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.spatial.distance import jensenshannon


def build_signature(ref, labels, K):
    lib = ref.sum(1, keepdims=True)
    prof = ref / lib
    S = np.vstack([prof[labels == k].mean(0) for k in range(K)])
    depth = np.array([lib[labels == k].mean() for k in range(K)])
    return S, depth


def select_markers(S, n_per_state, eps=1e-5):
    sel = set()
    for k in range(S.shape[0]):
        others = np.delete(S, k, 0).mean(0)
        lfc = np.log((S[k] + eps) / (others + eps))
        lfc[S[k] <= np.percentile(S[k], 50)] = -np.inf
        sel.update(np.argsort(lfc)[::-1][:n_per_state].tolist())
    return np.array(sorted(sel))


def _to_cell(W, depth):
    W = W / depth[None]
    return W / W.sum(1, keepdims=True)


def deconv_nnls(Y, S, depth, **_):
    Sn = S / S.sum(1, keepdims=True)
    Yn = Y / np.maximum(Y.sum(1, keepdims=True), 1)
    wg = 1 / np.sqrt(Sn.mean(0) + 1e-6)
    A = (Sn * wg).T
    W = np.zeros((len(Y), len(S)))
    for i in range(len(Y)):
        W[i], _r = nnls(A, Yn[i] * wg)
    s = W.sum(1, keepdims=True)
    W = np.where(s > 0, W / np.maximum(s, 1e-12), 1 / len(S))
    return _to_cell(W, depth)


def deconv_poisson(Y, S, depth, platform=False, n_iter=200):
    Sn = S / S.sum(1, keepdims=True)
    n, G = Y.shape
    K = len(S)
    W = np.full((n, K), 1 / K)
    logg, Ntot = np.zeros(G), Y.sum(1)
    for it in range(n_iter):
        Sg = Sn * np.exp(logg)
        Sg /= Sg.sum(1, keepdims=True)
        mix = W @ Sg + 1e-12
        upd = W * ((Y / mix) @ Sg.T)
        s = upd.sum(1, keepdims=True)
        W = np.where(s > 0, upd / np.maximum(s, 1e-12), W)
        if platform and it >= 20 and it % 10 == 0:
            mix = W @ Sg + 1e-12
            logg += 0.5 * np.log((Y.sum(0) + 1) / ((Ntot[:, None] * mix).sum(0) + 1))
            logg = np.clip(logg - logg.mean(), -1.5, 1.5)
    return _to_cell(W, depth)


METHODS = {
    "nnls": deconv_nnls,
    "poisson_em": lambda Y, S, d, n_iter=200: deconv_poisson(Y, S, d, False, n_iter),
    "poisson_em_plat": lambda Y, S, d, n_iter=200: deconv_poisson(Y, S, d, True, n_iter),
}


def metrics(est, true):
    with np.errstate(all="ignore"):
        r = [np.corrcoef(est[:, k], true[:, k])[0, 1] for k in range(true.shape[1])]
    js = jensenshannon(est + 1e-12, true + 1e-12, axis=1, base=2)
    return dict(mean_r=float(np.nanmean(r)), rmse=float(np.sqrt(((est - true) ** 2).mean())),
                js_dist=float(np.nanmean(js)), per_state_r=[float(x) for x in r])


def run(cfg, outdir):
    d = np.load(f"{outdir}/sim.npz")
    K, dc = len(d["states"]), cfg["deconv"]
    S, depth = build_signature(d["ref_counts"], d["ref_labels"], K)
    idx = select_markers(S, dc["n_markers_per_state"])
    rows, est = [], {}
    for name, fn in METHODS.items():
        for sample in ("fetal", "tumor"):
            E = fn(d[f"{sample}_counts"][:, idx].astype(float), S[:, idx], depth, n_iter=dc["n_iter"])
            est[f"{sample}_{name}"] = E
            m = metrics(E, d[f"{sample}_true"])
            rows.append(dict(method=name, sample=sample, **{k: m[k] for k in ("mean_r", "rmse", "js_dist")}))
            for k, s in enumerate(d["states"]):
                rows.append(dict(method=name, sample=sample, state=s, r=m["per_state_r"][k]))
    df = pd.DataFrame(rows)
    df.to_csv(f"{outdir}/deconv_metrics.csv", index=False)
    summ = df[df.state.isna()].groupby("method")[["mean_r", "rmse", "js_dist"]].mean()
    best = summ.rmse.idxmin() if dc["main_method"] == "auto" else dc["main_method"]
    np.savez_compressed(f"{outdir}/deconv_est.npz", **est)
    json.dump({"best_method": best, "n_marker_genes": int(len(idx)),
               "summary": summ.round(4).to_dict("index")}, open(f"{outdir}/deconv_summary.json", "w"), indent=1)
    print(f"[deconv] {len(idx)} genes marqueurs ; meilleure methode = {best}\n{summ.round(3)}")


def robustness(cfg, outdir):
    """Petit N : degrade la reference (n cellules/etat) et mesure la perte de performance."""
    d = np.load(f"{outdir}/sim.npz")
    K, rc, dc = len(d["states"]), cfg["robustness"], cfg["deconv"]
    rng = np.random.default_rng(cfg["seed"] + 1)
    rows = []
    for n in rc["ref_sizes"]:
        for rep in range(rc["n_rep"]):
            keep = np.concatenate([rng.choice(np.where(d["ref_labels"] == k)[0], n, replace=False) for k in range(K)])
            S, depth = build_signature(d["ref_counts"][keep], d["ref_labels"][keep], K)
            idx = select_markers(S, dc["n_markers_per_state"])
            for name, fn in METHODS.items():
                E = fn(d["fetal_counts"][:, idx].astype(float), S[:, idx], depth, n_iter=100)
                m = metrics(E, d["fetal_true"])
                rows.append(dict(n_ref=n, rep=rep, method=name, mean_r=m["mean_r"], rmse=m["rmse"]))
    pd.DataFrame(rows).to_csv(f"{outdir}/robustness.csv", index=False)
    g = pd.DataFrame(rows).groupby(["method", "n_ref"]).mean_r.mean().unstack()
    print(f"[robustness] r moyen selon n cellules de reference/etat :\n{g.round(3)}")
