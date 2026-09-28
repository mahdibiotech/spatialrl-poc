"""Simulation d'un jeu de donnees multi-omique avec verite terrain.

Deux coupes spatiales (foetale, tumorale) + reference scRNA-seq + couche MSI desalignee.
Les niches perivasculaires sont *injectees* : le pipeline doit les retrouver (validation
positive) et ne pas en trouver dans les controles negatifs.
ATTENTION : les profils d'expression sont synthetiques, aucune conclusion biologique.
"""
import numpy as np
from scipy import ndimage
from .geometry import msi_to_spot

STATES = ["RL_stem_PRTG+", "RL_diff", "GC_precursor", "GC_neuron",
          "Purkinje", "Endothelial", "Microglia", "G3_MB_tumor"]
MARKERS = {
    "RL_stem_PRTG+": ["PRTG", "OTX2", "LMX1A", "VEGFA", "NOTCH1"],
    "RL_diff": ["ATOH1", "PAX6", "BARHL1"],
    "GC_precursor": ["MKI67", "NEUROD1", "TOP2A"],
    "GC_neuron": ["GABRA6", "RBFOX3", "SLC17A7"],
    "Purkinje": ["CALB1", "PCP4", "ITPR1"],
    "Endothelial": ["PECAM1", "CLDN5", "KDR", "JAG1"],
    "Microglia": ["CX3CR1", "P2RY12", "AIF1"],
    "G3_MB_tumor": ["MYC", "OTX2", "NRL", "IMPG2", "VEGFA", "NOTCH1"],
}
DEPTH = np.array([1500, 1500, 2000, 1750, 2250, 1250, 900, 2000], float)  # UMI / cellule
PARENT = {1: 0, 2: 1, 3: 2, 7: 0}  # trajectoire de differenciation (profils correles)


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def _sig(x):
    return 1 / (1 + np.exp(-x))


def make_genes(n_genes):
    markers = list(dict.fromkeys(g for v in MARKERS.values() for g in v))
    return markers + [f"GENE{i:04d}" for i in range(n_genes - len(markers))]


def build_profiles(rng, genes):
    gi = {g: i for i, g in enumerate(genes)}
    K, G = len(STATES), len(genes)
    base = rng.lognormal(0, 1.0, G)
    eps = np.zeros((K, G))
    for k in range(K):
        own = rng.normal(0, 0.4, G)
        eps[k] = own if k not in PARENT else 0.8 * eps[PARENT[k]] + 0.6 * own
    logmu = np.log(base)[None] + eps
    for k, s in enumerate(STATES):
        for g in MARKERS[s]:
            logmu[k, gi[g]] += np.log(rng.uniform(6, 15))
    S = np.exp(logmu)
    return S / S.sum(1, keepdims=True)


def make_vessels(rng, L, n=4, length=45):
    m = np.zeros((L, L), bool)
    for _ in range(n):
        p, ang = rng.uniform(5, L - 5, 2), rng.uniform(0, 2 * np.pi)
        for _ in range(length):
            ang += rng.normal(0, 0.35)
            p = np.clip(p + [np.cos(ang), np.sin(ang)], 0, L - 1)
            m[int(p[0]), int(p[1])] = True
    return m


def fetal_logits(L, d):
    y = (np.arange(L) / (L - 1))[:, None] * np.ones((1, L))
    Z = np.full((L, L, 8), -3.0)
    Z[..., 0] = 0.5 + 2.0 * np.exp(-(d - 2.5) ** 2 / (2 * 1.5 ** 2))
    Z[..., 1] = 1.5 * _sig((0.35 - y) * 10)
    Z[..., 2] = 1.5 * np.exp(-(y - 0.45) ** 2 / (2 * 0.1 ** 2))
    Z[..., 3] = 2.0 * _sig((y - 0.6) * 10)
    Z[..., 4] = 1.5 * np.exp(-(y - 0.55) ** 2 / (2 * 0.05 ** 2)) - 0.5
    Z[..., 5] = -2 + 5 * np.exp(-d ** 2 / (2 * 0.6 ** 2))
    Z[..., 6] = -0.5
    Z[..., 7] = -8
    return Z


def tumor_logits(L, d):
    niche = np.exp(-(d - 2.5) ** 2 / (2 * 1.5 ** 2))
    Z = np.empty((L, L, 8))
    Z[..., 0] = 0.0 + 0.5 * niche
    Z[..., 1], Z[..., 2], Z[..., 3], Z[..., 4] = -3, -2, -1.5, -3
    Z[..., 5] = -2 + 5 * np.exp(-d ** 2 / (2 * 0.6 ** 2))
    Z[..., 6] = 0.3
    Z[..., 7] = 2.0 + 1.5 * niche
    return Z


def simulate_reference(rng, S, n_per_state):
    K = len(STATES)
    lab = np.repeat(np.arange(K), n_per_state)
    lib = DEPTH[lab] * rng.lognormal(0, 0.3, len(lab))
    # heterogeneite intra-etat : perturbation propre a chaque cellule et gene
    het = np.exp(rng.normal(0, 0.4, (len(lab), S.shape[1])))
    mu = lib[:, None] * S[lab] * het
    mu /= mu.sum(1, keepdims=True) / lib[:, None]
    return rng.poisson(rng.gamma(2.0, mu / 2.0)), lab


def simulate_spots(rng, Z, S, plat, cells_mean):
    P = _softmax(Z).reshape(-1, len(STATES))
    P = P / P.sum(1, keepdims=True)
    ncell = 1 + rng.poisson(cells_mean, len(P))
    nk = np.array([rng.multinomial(n, p) for n, p in zip(ncell, P)])
    lam = (nk * DEPTH[None]) @ S * plat[None] * rng.lognormal(0, 0.25, (len(P), 1)) * 0.6
    return rng.poisson(lam), nk / nk.sum(1, keepdims=True)


def simulate_msi(rng, true_fetal, L, cfg):
    """4 images ioniques : 3 informatives (GC_neuron, Endothelial, RL_stem) + 1 bruit spatial."""
    M, th, sh = cfg["n_pix"], np.deg2rad(cfg["theta_deg"]), cfg["shift_px"]
    g = true_fetal.reshape(L, L, -1)
    maps = [ndimage.gaussian_filter(g[..., k], 1.0) for k in (3, 5, 0)]
    maps.append(ndimage.gaussian_filter(rng.normal(size=(L, L)), 3.0))
    Q = np.stack(np.meshgrid(np.arange(M), np.arange(M), indexing="ij"), -1).reshape(-1, 2).astype(float)
    Pq = msi_to_spot(Q, th, sh, L, M)
    ions = []
    for m in maps:
        v = ndimage.map_coordinates(m, Pq.T, order=1, mode="nearest")
        v = (v - v.mean()) / (v.std() + 1e-9)
        ions.append((v + rng.normal(0, cfg["noise"], v.shape)).reshape(M, M))
    return np.stack(ions)


def run(cfg, outdir):
    rng = np.random.default_rng(cfg["seed"])
    sc, L = cfg["sim"], cfg["sim"]["grid"]
    genes = make_genes(sc["n_genes"])
    S = build_profiles(rng, genes)
    plat = np.exp(rng.normal(0, sc["platform_sd"], len(genes)))
    ref, lab = simulate_reference(rng, S, sc["n_ref_per_state"])
    out = dict(genes=np.array(genes), states=np.array(STATES), ref_counts=ref.astype(np.int32),
               ref_labels=lab, S_true=S)
    for name, logit in (("fetal", fetal_logits), ("tumor", tumor_logits)):
        ves = make_vessels(rng, L)
        d = ndimage.distance_transform_edt(~ves)
        Y, T = simulate_spots(rng, logit(L, d), S, plat, sc["cells_per_spot"])
        out.update({f"{name}_counts": Y.astype(np.int32), f"{name}_true": T,
                    f"{name}_vessel_true": ves, f"{name}_dist_true": d})
    out["msi"] = simulate_msi(rng, out["fetal_true"], L, cfg["msi"])
    np.savez_compressed(f"{outdir}/sim.npz", **out)
    print(f"[simulate] {L*L} spots x {len(genes)} genes x 2 coupes ; {len(lab)} cellules de reference")
