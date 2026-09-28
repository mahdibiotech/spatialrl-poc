"""Figure de synthese, results.json, report.md et one-pager PDF (tous alimentes par les resultats reels)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHORT = ["RLstem", "RLdiff", "GCprec", "GCneu", "Purk", "Endo", "Micro", "G3"]
MC = {"nnls": "#9aa0a6", "poisson_em": "#4c78a8", "poisson_em_plat": "#e4572e"}
BINS = ["0", "1", "2", "3", "4", "5-6", "7-9", ">9"]


def collect(cfg, o):
    dsum = json.load(open(f"{o}/deconv_summary.json"))
    nsum = json.load(open(f"{o}/niche_summary.json"))
    reg = json.load(open(f"{o}/msi_registration.json"))
    ring = pd.read_csv(f"{o}/niche_ring.csv")
    lr = pd.read_csv(f"{o}/niche_lr.csv")
    rob = pd.read_csv(f"{o}/robustness.csv").groupby(["method", "n_ref"]).mean_r.mean().unstack()
    msi = pd.read_csv(f"{o}/msi_summary.csv")
    best = dsum["best_method"]
    g = lambda s, st: ring[(ring["sample"] == s) & (ring.state == st)].iloc[0]
    pos = lr[(~lr.control)]
    res = dict(
        best_method=best, deconv=dsum["summary"], n_marker_genes=dsum["n_marker_genes"],
        robustness_r={m: {int(k): round(float(v), 3) for k, v in rob.loc[m].items()} for m in rob.index},
        niche_fetal_RLstem=dict(z=round(float(g("fetal", "RL_stem_PRTG+").z), 2), q=float(g("fetal", "RL_stem_PRTG+").q)),
        niche_tumor_G3=dict(z=round(float(g("tumor", "G3_MB_tumor").z), 2), q=float(g("tumor", "G3_MB_tumor").q)),
        vessel_distance_spearman=nsum["fetal_dist_spearman"],
        lr_positive={f"{r['sample']}:{r.pair}": dict(z=round(r.z, 2), q=round(r.q, 3)) for _, r in pos.iterrows()},
        lr_control_fp_rate=nsum["lr_control_false_positive_rate"],
        msi=reg, msi_corr=msi.round(3).to_dict("records"))
    json.dump(res, open(f"{o}/results.json", "w"), indent=1)
    return res, rob, ring, lr, msi


def figure(cfg, o, res, rob, lr, msi):
    d = np.load(f"{o}/sim.npz")
    est = np.load(f"{o}/deconv_est.npz")
    na = np.load(f"{o}/niche_arrays.npz")
    L, best = cfg["sim"]["grid"], res["best_method"]
    E = est[f"fetal_{best}"]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 4, figsize=(20, 8.4))
    # A1/A2 cartes
    T = d["fetal_true"][:, 0].reshape(L, L)
    for a, M, t in ((ax[0, 0], T, "A. Verite terrain : RL_stem (coupe foetale)"),
                    (ax[0, 1], E[:, 0].reshape(L, L), f"Estimation ({best})")):
        im = a.imshow(M, cmap="magma", vmin=0, vmax=1)
        a.contour(d["fetal_vessel_true"], levels=[0.5], colors="cyan", linewidths=0.8)
        a.set_title(t, fontsize=10); a.axis("off")
    r0 = np.corrcoef(T.ravel(), E[:, 0])[0, 1]
    ax[0, 1].text(1, L - 2, f"r = {r0:.2f}   (cyan = vaisseaux)", color="w", fontsize=9)
    # A3 comparaison
    a = ax[0, 2]
    ms = list(res["deconv"].keys())
    rm = [res["deconv"][m]["rmse"] for m in ms]
    a.bar(range(len(ms)), rm, color=[MC[m] for m in ms])
    for i, m in enumerate(ms):
        a.text(i, rm[i] + 0.001, f"r={res['deconv'][m]['mean_r']:.3f}", ha="center")
    a.set_xticks(range(len(ms))); a.set_xticklabels(["NNLS", "Poisson-EM", "Poisson-EM\n+ plateforme"])
    a.set_ylabel("RMSE (proportions)"); a.set_title("B. Comparaison de 3 methodes de deconvolution", fontsize=10)
    # A4 robustesse
    a = ax[0, 3]
    for m in rob.index:
        a.plot(rob.columns, rob.loc[m], "o-", color=MC[m], label=m)
    a.set_xscale("log"); a.set_xlabel("cellules de reference par etat"); a.set_ylabel("r moyen")
    a.set_title("C. Robustesse au petit N (reference degradee)", fontsize=10); a.legend(fontsize=8)
    # B1 profils distance
    a = ax[1, 0]
    x = np.arange(len(BINS))
    for key, k, lab, c in (("fetal", 0, "foetal RL_stem", "#e4572e"), ("tumor", 7, "tumeur G3", "#4c78a8")):
        a.plot(x, na[f"prof_{key}_est"][:, k], "o-", color=c, label=f"{lab} (est.)")
        a.plot(x, na[f"prof_{key}_true"][:, k], "--", color=c, alpha=.6, label=f"{lab} (vrai)")
    a.set_xticks(x); a.set_xticklabels(BINS); a.set_xlabel("distance au vaisseau (spots)")
    a.set_ylabel("proportion moyenne"); a.set_title("D. Niche perivasculaire (anneau 1.5-4 spots)", fontsize=10)
    a.legend(fontsize=7)
    n1 = res["niche_fetal_RLstem"]; n2 = res["niche_tumor_G3"]
    a.text(.98, .55, f"z foetal={n1['z']}\nz tumeur={n2['z']}", transform=a.transAxes, ha="right", fontsize=8)
    # B2 voisinage
    a = ax[1, 1]
    Z = np.clip(na["nenr_fetal"], -25, 25)
    im = a.imshow(Z, cmap="RdBu_r", vmin=-25, vmax=25)
    a.set_xticks(range(8)); a.set_xticklabels(SHORT, rotation=60, fontsize=7)
    a.set_yticks(range(8)); a.set_yticklabels(SHORT, fontsize=7)
    a.set_title("E. Enrichissement de voisinage (z, foetal)", fontsize=10); plt.colorbar(im, ax=a, fraction=.046)
    # B3 LR
    a = ax[1, 2]
    for i, s in enumerate(("fetal", "tumor")):
        c = lr[(lr["sample"] == s) & lr.control]
        a.scatter(i + np.random.default_rng(0).uniform(-.15, .15, len(c)), c.z, s=14, color="#bbb", label="controles (paires aleatoires)" if i == 0 else None)
        for j, (_, r) in enumerate(lr[(lr["sample"] == s) & ~lr.control].iterrows()):
            a.scatter(i + .3 + .1 * j, r.z, s=60, color="#e4572e" if r.q < .05 else "#4c78a8", zorder=3)
            a.annotate(r.pair, (i + .3 + .1 * j, r.z), fontsize=7, xytext=(3, 3), textcoords="offset points")
    a.axhline(0, color="k", lw=.5); a.set_xticks([0, 1]); a.set_xticklabels(["foetal", "tumeur"])
    a.set_ylabel("z (null par decalage torique)"); a.set_title("F. Ligand-recepteur : positifs vs controles", fontsize=10); a.legend(fontsize=7)
    # B4 MSI
    a = ax[1, 3]
    w = .27
    for j, (key, c) in enumerate((("unregistered", "#bbb"), ("registered", "#e4572e"), ("oracle", "#4c78a8"))):
        a.bar(np.arange(len(msi)) + (j - 1) * w, msi[key], w, color=c, label={"unregistered": "sans recalage", "registered": "recale (estime)", "oracle": "oracle"}[key])
    a.set_xticks(range(len(msi))); a.set_xticklabels(["ion1\nGC_neuron", "ion2\nEndo", "ion3\nRL_stem", "ion4\nbruit"], fontsize=8)
    a.set_ylabel("correlation ion-etat"); a.legend(fontsize=7)
    m = res["msi"]
    a.set_title(f"G. Recalage MSI : erreur {m['TRE_unregistered_spots']:.1f} -> {m['TRE_registered_spots']:.2f} spots", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{o}/figure_synthese.png", dpi=170)
    plt.close(fig)


def pdf(cfg, o, res):
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table, TableStyle
    W, H = landscape(A4)
    doc = SimpleDocTemplate(f"{o}/SpatialRL-POC_onepager.pdf", pagesize=(W, H), leftMargin=1.1 * cm,
                            rightMargin=1.1 * cm, topMargin=.9 * cm, bottomMargin=.8 * cm)
    T = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=14, leading=16)
    S = ParagraphStyle("s", fontName="Helvetica-Oblique", fontSize=8, leading=10, textColor=colors.HexColor("#444"))
    B = ParagraphStyle("b", fontName="Helvetica", fontSize=7.2, leading=8.8)
    d, b = res["deconv"], res["best_method"]
    lrp = res["lr_positive"]
    fp = res["lr_control_fp_rate"]
    m = res["msi"]
    c1 = ("<b>Ce que le POC demontre (simulation)</b><br/>"
          f"- Deconvolution : r moyen {d[b]['mean_r']:.3f}, RMSE {d[b]['rmse']:.3f} ({b}) contre {d['nnls']['mean_r']:.3f} / {d['nnls']['rmse']:.3f} pour NNLS.<br/>"
          f"- Petit N : r = {res['robustness_r'][b][min(res['robustness_r'][b])]:.2f} avec {min(res['robustness_r'][b])} cellules de reference par etat, {res['robustness_r'][b][max(res['robustness_r'][b])]:.2f} avec {max(res['robustness_r'][b])}.<br/>"
          f"- Niche perivasculaire retrouvee : z = {res['niche_fetal_RLstem']['z']} (RL_stem, foetal) et {res['niche_tumor_G3']['z']} (G3, tumeur).<br/>"
          f"- Recalage MSI : erreur {m['TRE_unregistered_spots']:.1f} -> {m['TRE_registered_spots']:.2f} spots.")
    c2 = ("<b>Limites identifiees (a corriger sur vos donnees)</b><br/>"
          f"- Ligand-recepteur : {sum(v['q']<.05 for v in lrp.values())}/{len(lrp)} paires positives detectees a q&lt;0.05 ; "
          f"taux de faux positifs sur controles = {fp['fetal']:.0%} (foetal) vs 5% nominal : null a durcir.<br/>"
          "- Donnees simulees : profils synthetiques, aucune conclusion biologique.<br/>"
          "- Contrastes de niche compositionnels : un etat enrichi en deprime mecaniquement les autres.<br/>"
          "- Recalage MSI : ancre ion-etat supposee ; sur tissu, H&amp;E ou coupe adjacente.")
    c3 = ("<b>Mes 90 premiers jours sur vos donnees</b><br/>"
          "- J1-30 : audit des jeux (spatial, sc, MSI), QC harmonise, pipeline reproductible + conteneur.<br/>"
          "- J31-60 : reference foetale commune, benchmark cell2location / RCTD / Tangram, harmonisation multi-plateformes.<br/>"
          "- J61-90 : cartographie tumeur vs lignee rhombique, niche neurovasculaire, premier rapport interne + figures.")
    tb = Table([[Paragraph(c, B) for c in (c1, c2, c3)]], colWidths=[(W - 2.2 * cm) / 3] * 3)
    tb.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOX", (0, 0), (-1, -1), .4, colors.grey),
                            ("INNERGRID", (0, 0), (-1, -1), .4, colors.lightgrey), ("LEFTPADDING", (0, 0), (-1, -1), 5)]))
    iw = W - 2.2 * cm
    story = [Paragraph("SpatialRL-POC : cartographie spatiale des etats cellulaires et de la niche perivasculaire", T),
             Paragraph("Preuve de concept sur donnees 100% simulees avec verite terrain - valide la methode, ne prouve aucune biologie. "
                       "Code, pipeline et environnement reproductibles : voir le depot.", S),
             Spacer(1, 3), Image(f"{o}/figure_synthese.png", width=iw, height=iw * 8.4 / 20), Spacer(1, 4), tb]
    doc.build(story)


def markdown(o, res):
    b, d = res["best_method"], res["deconv"]
    lines = ["# Resultats SpatialRL-POC (simulation)", "",
             "Donnees 100 % simulees avec verite terrain. Voir `figure_synthese.png`.", "",
             "## Deconvolution", "", "| Methode | r moyen | RMSE | Distance JS |", "|---|---|---|---|"]
    lines += [f"| {m} | {v['mean_r']:.3f} | {v['rmse']:.3f} | {v['js_dist']:.3f} |" for m, v in d.items()]
    lines += ["", f"Methode retenue : **{b}**. Les metriques ne s'accordent pas toutes (la distance JS penalise "
              "plus les petites masses fausses que la RMSE) : l'arbitrage est a discuter, pas a masquer.", "",
              "## Robustesse (r moyen selon cellules de reference/etat)", "", pd.DataFrame(res["robustness_r"]).T.round(3).to_markdown(), "",
              "## Niche", "", f"- RL_stem foetal : z = {res['niche_fetal_RLstem']['z']}, q = {res['niche_fetal_RLstem']['q']:.3f}",
              f"- G3 tumeur : z = {res['niche_tumor_G3']['z']}, q = {res['niche_tumor_G3']['q']:.3f}",
              f"- Rho de Spearman distance estimee vs vraie (foetal) : {res['vessel_distance_spearman']:.2f}", "",
              "## Ligand-recepteur", ""] + [f"- {k} : z = {v['z']}, q = {v['q']}" for k, v in res["lr_positive"].items()]
    lines += [f"- Faux positifs sur controles (q<0.05) : {res['lr_control_fp_rate']}", "",
              "## Recalage MSI", "", "```", json.dumps(res["msi"], indent=1), "```"]
    open(f"{o}/report.md", "w").write("\n".join(lines))


def run(cfg, o):
    res, rob, ring, lr, msi = collect(cfg, o)
    figure(cfg, o, res, rob, lr, msi)
    pdf(cfg, o, res)
    markdown(o, res)
    print("[report] figure_synthese.png, SpatialRL-POC_onepager.pdf, report.md, results.json")
