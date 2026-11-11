"""Estructura de un tema: dónde cambia de sección y qué compases son «fuertes» (estribillo / tutti) o serenos (intro, quiebre, final).

Se calcula sobre el perfil por compás que ya tiene el análisis (cromas, energía, batería, bajo, voz), sin leer el audio otra vez.
Lo usa el planificador (idea «mezclar por estructura») para que la fusión empiece y termine en cambios de sección y para no pisar
un estribillo con otro.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VENTANA = 4                    # compases a cada lado para medir cuánto cambia el tema
SEPARACION = 4                 # mínimo de compases entre dos cambios de sección


@dataclass
class Estructura:
    bordes: np.ndarray         # compases donde empieza una sección nueva
    fuerte: np.ndarray         # (n,) 1.0 = compás de sección fuerte, 0.0 = serena
    dist_borde: np.ndarray     # (n+1,) distancia (en compases) al cambio de sección más cercano


def _perfil(A) -> np.ndarray:
    n = A.n_compases
    cols = []
    ch = np.asarray(A.chroma, np.float64)[:n]
    if len(ch) == n and ch.shape[1] == 12:
        ch = ch / (np.linalg.norm(ch, axis=1, keepdims=True) + 1e-9)
        cols.append(0.6 * ch)
    for nombre, peso in (("energia", 1.5), ("bateria", 1.0), ("bajo", 1.0), ("voz", 0.8)):
        v = np.asarray(getattr(A, nombre), np.float64)[:n]
        if len(v) == n:
            cols.append(peso * v[:, None])
    return np.hstack(cols) if cols else np.zeros((n, 1))


def analizar(A) -> Estructura:
    n = A.n_compases
    if n < 2 * VENTANA + 2:
        return Estructura(np.zeros(0, int), np.ones(max(n, 0)), np.zeros(max(n, 0) + 1))
    X = _perfil(A)
    cs = np.vstack([np.zeros((1, X.shape[1])), np.cumsum(X, axis=0)])
    nov = np.zeros(n)
    for i in range(VENTANA, n - VENTANA + 1):
        izq = (cs[i] - cs[i - VENTANA]) / VENTANA
        der = (cs[min(n, i + VENTANA)] - cs[i]) / max(1, min(n, i + VENTANA) - i)
        nov[i] = float(np.linalg.norm(der - izq))
    umbral = max(0.25, float(np.median(nov[nov > 0])) * 1.3) if (nov > 0).any() else 1e9
    cand = [i for i in range(1, n - 1) if nov[i] >= umbral and nov[i] >= nov[i - 1] and nov[i] > nov[i + 1]]
    cand.sort(key=lambda i: -nov[i])                       # los cambios más claros mandan; los cercanos a uno ya elegido se descartan
    elegidos: list[int] = []
    for i in cand:
        if all(abs(i - j) >= SEPARACION for j in elegidos):
            elegidos.append(i)
    bordes = np.array(sorted(elegidos), int)
    e = np.asarray(A.energia, np.float64)[:n]
    ref = float(np.median(e[e > 0.06])) if (e > 0.06).any() else 0.5
    suave = np.convolve(e, np.ones(3) / 3, mode="same")
    fuerte = (suave >= 1.05 * ref).astype(np.float64)
    pos = np.arange(n + 1)
    dist = np.abs(pos[:, None] - bordes[None, :]).min(axis=1).astype(np.float64) if len(bordes) else np.full(n + 1, 99.0)
    return Estructura(bordes, fuerte, dist)
