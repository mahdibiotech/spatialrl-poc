# Passer des données simulées aux vraies données

> Extraits **non exécutés ici** (pas de scanpy/cell2location dans l'environnement de rédaction).
> Vérifier les signatures d'API dans la documentation de la version installée.
> **Avant de télécharger un jeu public : vérifier accession, licence et conditions d'usage.**

## 1. Sources publiques candidates (à vérifier)
- Atlas single-cell du cervelet foetal humain (Aldinger et al., Nat Neurosci 2021).
- Données de médulloblastome single-cell / spatial déposées avec les publications de l'équipe
  (Hendrikse 2022, Visvanathan 2024) *si elles sont publiques*.
- Si aucun jeu spatial adapté n'est disponible : conserver la simulation (comme ici) et le dire.

## 2. QC (Scanpy / Squidpy)
```python
import scanpy as sc, squidpy as sq
adata_vis = sc.read_visium("path/to/spaceranger_out")     # ou sc.read_h5ad(...)
sc.pp.calculate_qc_metrics(adata_vis, percent_top=None, inplace=True)
# seuils justifiés par coupe (distribution, pas valeurs par défaut) ; consigner les spots retirés
```

## 3. Déconvolution : remplacer les méthodes maison
```python
# cell2location (référence -> signatures -> cartographie)
import cell2location as c2l
c2l.models.RegressionModel.setup_anndata(adata_ref, batch_key="sample", labels_key="cell_state")
mod = c2l.models.RegressionModel(adata_ref); mod.train(max_epochs=250)
adata_ref = mod.export_posterior(adata_ref)
sig = adata_ref.varm["means_per_cluster_mu_fg"]            # gènes x états
c2l.models.Cell2location.setup_anndata(adata_vis, batch_key="sample")
m = c2l.models.Cell2location(adata_vis, cell_state_df=sig, N_cells_per_location=5, detection_alpha=20)
m.train(max_epochs=30000)
```
Comparer avec RCTD (R, `spacexr`) et Tangram sur **les mêmes gènes et la même référence**, avec
`deconv.metrics` si vérité terrain simulée à partir de la référence (spots synthétiques),
ou concordance aux marqueurs sur données réelles.

## 4. Niche et communication (Squidpy)
```python
sq.gr.spatial_neighbors(adata_vis, coord_type="grid", n_neighs=6)
sq.gr.nhood_enrichment(adata_vis, cluster_key="dominant_state")
sq.gr.ligrec(adata_vis, cluster_key="dominant_state", n_perms=1000)   # + contrôles négatifs appariés
```
Conserver les contrôles de `niche.lr_scan` : ils ont révélé un taux de faux positifs de 15 % ici.

## 5. Recalage MSI
Remplacer l'ancre ion↔état par H&E / coupe adjacente / coloration post-MSI ; conserver
`coregister.tre` comme métrique (à calculer sur landmarks posés à la main).
