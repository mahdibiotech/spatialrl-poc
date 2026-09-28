"""Transformation rigide spots -> pixels MSI (rotation + translation + echelle fixe 1/2)."""
import numpy as np


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def spot_to_msi(P, theta, t, L, M, scale=0.5):
    """P : (N,2) en (ligne, colonne) sur la grille spatiale L x L -> coordonnees pixel MSI (M x M)."""
    c, cm = (L - 1) / 2, (M - 1) / 2
    return scale * (P - c) @ _rot(theta).T + cm + np.asarray(t)


def msi_to_spot(Q, theta, t, L, M, scale=0.5):
    c, cm = (L - 1) / 2, (M - 1) / 2
    return (Q - cm - np.asarray(t)) @ _rot(-theta).T / scale + c
