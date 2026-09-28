# Resultats SpatialRL-POC (simulation)

Donnees 100 % simulees avec verite terrain. Voir `figure_synthese.png`.

## Deconvolution

| Methode | r moyen | RMSE | Distance JS |
|---|---|---|---|
| nnls | 0.923 | 0.053 | 0.168 |
| poisson_em | 0.973 | 0.049 | 0.219 |
| poisson_em_plat | 0.975 | 0.038 | 0.200 |

Methode retenue : **poisson_em_plat**. Les metriques ne s'accordent pas toutes (la distance JS penalise plus les petites masses fausses que la RMSE) : l'arbitrage est a discuter, pas a masquer.

## Robustesse (r moyen selon cellules de reference/etat)

|                 |     3 |     5 |    10 |    25 |    50 |   200 |
|:----------------|------:|------:|------:|------:|------:|------:|
| nnls            | 0.954 | 0.962 | 0.974 | 0.98  | 0.972 | 0.974 |
| poisson_em      | 0.953 | 0.963 | 0.978 | 0.985 | 0.981 | 0.982 |
| poisson_em_plat | 0.953 | 0.964 | 0.98  | 0.988 | 0.987 | 0.989 |

## Niche

- RL_stem foetal : z = 5.41, q = 0.003
- G3 tumeur : z = 5.91, q = 0.005
- Rho de Spearman distance estimee vs vraie (foetal) : 1.00

## Ligand-recepteur

- fetal:VEGFA->KDR : z = 3.16, q = 0.036
- fetal:JAG1->NOTCH1 : z = 0.72, q = 0.731
- tumor:VEGFA->KDR : z = -1.69, q = 0.998
- tumor:JAG1->NOTCH1 : z = -4.49, q = 0.998
- Faux positifs sur controles (q<0.05) : {'fetal': 0.15, 'tumor': 0.0}

## Recalage MSI

```
{
 "theta_est_deg": 8.221693823975588,
 "shift_est": [
  1.5251051124103712,
  -0.9297945101393998
 ],
 "theta_true_deg": 8.0,
 "shift_true": [
  1.5,
  -1.0
 ],
 "TRE_registered_spots": 0.15974184230520255,
 "TRE_unregistered_spots": 4.197902649347742
}
```