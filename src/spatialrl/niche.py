"""Analyse de niche : distance aux vaisseaux, voisinage, communication ligand-recepteur.

Nulls par decalage torique aleatoire des cartes (preserve l'autocorrelation spatiale,
contrairement a un melange des labels de spots qui gonfle les faux positifs).
Approximation : les bords sont raccordes (tore).
"""
import json
import numpy as np
import pandas as pd
from scipy import ndimage

LR_PAIRS = [("VEGFA", "KDR"), ("JAG1", "NOTCH1")]


def _bh(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = np.empty_like(p, float)
    q[o] = np.minimum.accumulate((p[o] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def _roll(rng, a):
    L = a.shape[0]
    return np.roll(a, (rng.integers(1, L), rng.integers(1, L)), axis=(0, 1))


def estimated_vessel_distance(endo_grid, thr, min_size=4):
    m = endo_grid > thr
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    m = np.isin(lab, 1 + np.where(sizes >= min_size)[0])
    return ndimage.distance_transform_edt(~m), m


def ring_contrast(P, d, ring, rng, n_perm):
    valid = d >= ring[0]
    inside, outside = (d >= ring[0]) & (d <= ring[1]), d > ring[1]

    def stat(A):
        return A[inside].mean() - A[outside].mean()
    obs = stat(P)
    null = np.array([stat(_roll(rng, P)) for _ in range(n_perm)])
    p = (1 + np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean()))) / (n_perm + 1)
    return obs, (obs - null.mean()) / (null.std() + 1e-12), p


def distance_profile(P_grid, d, edges):
    b = np.digitize(d, edges) - 1
    return np.array([[P_grid[..., k][b == i].mean() if (b == i).any() else np.nan
                      for k in range(P_grid.shape[-1])] for i in range(len(edges) - 1)])


def neighbor_enrichment(lab, K, rng, n_perm):
    def counts(a):
        C = np.zeros((K, K))
        for x, y in ((a[:, :-1], a[:, 1:]), (a[:-1], a[1:])):
            np.add.at(C, (x.ravel(), y.ravel()), 1)
        return C + C.T
    obs = counts(lab)
    null = np.stack([counts(_roll(rng, lab)) for _ in range(n_perm // 2)])
    return (obs - null.mean(0)) / (null.std(0) + 1e-9)


def lr_scan(E, genes, pairs, rng, radius, n_perm):
    gi = {g: i for i, g in enumerate(genes)}
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    ker = ((yy ** 2 + xx ** 2) <= radius ** 2).astype(float)
    ker[radius, radius] = 0
    ker /= ker.sum()
    rows = []
    for name, (l, r) in pairs:
        C = ndimage.convolve(E[..., gi[l]], ker, mode="wrap")
        R = E[..., gi[r]]
        obs = (R * C).mean()
        null = np.array([(R * _roll(rng, C)).mean() for _ in range(n_perm)])
        rows.append(dict(pair=name, ligand=l, receptor=r, score=obs,
                         z=(obs - null.mean()) / (null.std() + 1e-12),
                         p=(1 + np.sum(null >= obs)) / (n_perm + 1)))
    return pd.DataFrame(rows)


def run(cfg, outdir):
    d = np.load(f"{outdir}/sim.npz")
    est = np.load(f"{outdir}/deconv_est.npz")
    best = json.load(open(f"{outdir}/deconv_summary.json"))["best_method"]
    nc, L = cfg["niche"], cfg["sim"]["grid"]
    rng = np.random.default_rng(cfg["seed"] + 2)
    states, genes = list(d["states"]), list(d["genes"])
    K, res, ring_rows, prof, nenr, lr_all = len(states), {}, [], {}, {}, []
    edges = np.array([-0.1, 0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 9.5, 100])
    for sample in ("fetal", "tumor"):
        P = est[f"{sample}_{best}"].reshape(L, L, K)
        T = d[f"{sample}_true"].reshape(L, L, K)
        dist, mask = estimated_vessel_distance(P[..., states.index("Endothelial")], nc["vessel_thr"])
        dt = d[f"{sample}_dist_true"]
        res[f"{sample}_dist_spearman"] = float(pd.Series(dist.ravel()).corr(pd.Series(dt.ravel()), method="spearman"))
        for k, s in enumerate(states):
            if sample == "fetal" and s == "G3_MB_tumor":
                continue
            obs, z, p = ring_contrast(P[..., k], dist, nc["ring"], rng, nc["n_perm"])
            tr = ring_contrast(T[..., k], dist, nc["ring"], rng, 50)[0]
            ring_rows.append(dict(sample=sample, state=s, contrast_est=obs, contrast_true=tr, z=z, p=p))
        prof[f"{sample}_est"] = distance_profile(P, dist, edges)
        prof[f"{sample}_true"] = distance_profile(T, dist, edges)
        nenr[sample] = neighbor_enrichment(P.argmax(-1), K, rng, nc["n_perm"])
        Y = d[f"{sample}_counts"].astype(float)
        E = np.log1p(Y / np.maximum(Y.sum(1, keepdims=True), 1) * 1e4).reshape(L, L, -1)
        filler = [g for g in genes if g.startswith("GENE")]
        ctrl = [(f"ctrl{i}", tuple(rng.choice(filler, 2, replace=False))) for i in range(nc["n_null_pairs"])]
        lr = lr_scan(E, genes, [(f"{a}->{b}", (a, b)) for a, b in LR_PAIRS] + ctrl, rng, nc["lr_radius"], nc["n_perm"])
        lr["sample"], lr["control"] = sample, lr.pair.str.startswith("ctrl")
        lr["q"] = _bh(lr.p.values)
        lr_all.append(lr)
    ring = pd.DataFrame(ring_rows)
    ring["q"] = ring.groupby("sample").p.transform(lambda x: _bh(x.values))
    ring.to_csv(f"{outdir}/niche_ring.csv", index=False)
    lr = pd.concat(lr_all)
    lr.to_csv(f"{outdir}/niche_lr.csv", index=False)
    np.savez_compressed(f"{outdir}/niche_arrays.npz", edges=edges, **{f"prof_{k}": v for k, v in prof.items()},
                        **{f"nenr_{k}": v for k, v in nenr.items()})
    ctrl_fp = lr[lr.control].groupby("sample").apply(lambda x: float((x.q < 0.05).mean()), include_groups=False)
    res["lr_control_false_positive_rate"] = ctrl_fp.to_dict()
    json.dump(res, open(f"{outdir}/niche_summary.json", "w"), indent=1)
    print(f"[niche] methode={best}\n{ring.round(3).to_string(index=False)}")
    print(lr[~lr.control].round(4).to_string(index=False), "\nFP controles:", res["lr_control_false_positive_rate"])
