# SpatialRL-POC

Preuve de concept reproductible : **cartographie spatiale des états cellulaires de la lèvre rhombique
foetale, projection sur un médulloblastome de Groupe 3, analyse de la niche périvasculaire et
intégration d'une couche métabolomique (MSI)**.

Conçu comme réponse à l'offre d'ingénieur spatial transcriptomics / cancer genomics de l'équipe
Génomique et Développement des cancers de l'enfant (Institut Curie).

## Ce que c'est, et ce que ce n'est pas

- **100 % données simulées, avec vérité terrain.** Les profils d'expression sont synthétiques
  (les symboles de gènes sont des étiquettes de lisibilité). Le POC valide la **méthode** : peut-on
  retrouver une structure connue, avec quelle erreur, et où le pipeline échoue-t-il ?
  Il ne démontre **aucune biologie**.
- Les trois déconvolutions sont des **implémentations maison** (NNLS ; mélange Poisson par EM ;
  Poisson-EM avec effet de plateforme, dans l'esprit de RCTD) choisies pour tourner sans
  dépendance lourde. Elles ne remplacent pas cell2location / RCTD / Tangram sur vos données
  (`docs/real_data_adapters.md`).
- Le Snakefile n'a pas été exécuté (Snakemake absent de l'environnement de rédaction) ; la logique
  est testée via la CLI (`make all`, `make test`).

## Lancer

```bash
conda env create -f environment.yml && conda activate spatialrl
make all      # ~30 s : simulation -> déconvolution -> robustesse -> niche -> MSI -> rapport
make test     # test bout en bout sur config réduite
# ou : snakemake -c4   /   docker build -t spatialrl . && docker run --rm -v $PWD/results:/work/results spatialrl
```

Résultats : `results/` (`SpatialRL-POC_onepager.pdf`, `figure_synthese.png`, `report.md`, `results.json`).
Tout est déterministe (`seed` dans `config.yaml`) : deux exécutions donnent des résultats identiques.

## Résultats obtenus (seed 42, 2 × 2500 spots, 400 gènes, 8 états)

| Méthode | r moyen | RMSE | Distance JS |
|---|---|---|---|
| NNLS | 0.923 | 0.053 | **0.168** |
| Poisson-EM | 0.973 | 0.049 | 0.219 |
| Poisson-EM + plateforme | **0.975** | **0.038** | 0.200 |

Les métriques ne s'accordent pas : NNLS gagne en distance JS (il produit des zéros exacts, la JS
pénalise les petites masses fausses). Le choix de la métrique est un vrai arbitrage.

- **Petit N** : r = 0.95 avec 3 cellules de référence par état, 0.99 avec 200 (dégradation modérée
  car les états simulés sont bien séparés ; sur données réelles, attendre pire).
- **Niche périvasculaire** : retrouvée dans les deux coupes (RL_stem foetal z = 5.4 ; G3 tumeur z = 5.9),
  contraste estimé proche du contraste réel (0.317 vs 0.338 ; 0.147 vs 0.159).
- **MSI** : erreur de recalage 4.2 → 0.16 spots ; corrélation ion↔état ×2 après recalage
  (ex. ion2↔Endothelial 0.24 → 0.58), ion de bruit à 0.11.

## Limites identifiées (à corriger, pas à cacher)

1. **Ligand-récepteur faible** : 1 paire positive sur 4 détectée à q < 0.05 (VEGFA→KDR foetal).
   Le null par décalage torique est conservateur quand le ligand est exprimé partout (tumeur).
2. **Faux positifs gonflés sur les contrôles foetaux** : 15 % (6/40) à q < 0.05 au lieu de 5 %.
   Cause probable : les gènes « neutres » portent de la structure d'état. Pistes : contrôles
   appariés en expression, null stratifié, plus de permutations, échantillon de contrôles plus grand
   (40 paires = intervalle de confiance large).
3. **Contrastes de niche compositionnels** : un état enrichi en déprime mécaniquement les autres
   (contrastes négatifs « significatifs » pour RL_diff, Microglia). À lire avec des modèles
   compositionnels sur vraies données.
4. **Recalage MSI** : l'ancre ion↔état est supposée connue et la transformation rigide. Sur tissu :
   H&E / coupe adjacente / coloration post-MSI, transformation non rigide possible.
5. **Épissage alternatif (OTX2, NCB 2024)** : non traité. Les données spatiales à lectures courtes
   limitent l'analyse d'isoformes ; piste = lectures longues.

## Structure

```
src/spatialrl/  simulate · deconv · niche · coregister · geometry · report · cli
tests/          test bout en bout (config réduite)
docs/           real_data_adapters.md · plan_90_jours.md · email_candidature.md
Snakefile · Makefile · Dockerfile · environment.yml · config.yaml
```
