"""Filtros de mediana de 17 puntos (los del HPSS del análisis), compilados con numba (código de máquina, vectorizado, sin GIL).

Dan EXACTAMENTE los mismos valores que `scipy.ndimage.median_filter(..., mode="reflect")` (la mediana es un valor de la ventana, no hay
redondeos), pero tardan unas 8–10 veces menos. Si numba no está o algo falla, `disponible` queda en False y el análisis sigue con scipy."""
from __future__ import annotations

import numpy as np

K = 17                      # tamaño de ventana de librosa.decompose.hpss por defecto (kernel_size=31 → aquí 17, el que usa el análisis)
H = K // 2
disponible = False

try:
    from numba import njit

    C = 128                  # columnas que se procesan juntas (ancho fijo y datos contiguos: el compilador las vectoriza)

    @njit(cache=True, nogil=True, fastmath=True)
    def _mediana_filas(w, res):
        """w: (K, C) con las 17 vistas; res: (C,) salida. Mediana por conteo de rangos (sin ramas en el lazo interno: se vectoriza)."""
        cnt = np.zeros(C, np.int32)
        for j in range(C):
            res[j] = 0.0
        for i in range(K):
            for j in range(C):
                cnt[j] = 0
            for k in range(i):
                for j in range(C):
                    cnt[j] += (w[k, j] <= w[i, j])
            for k in range(i + 1, K):
                for j in range(C):
                    cnt[j] += (w[k, j] < w[i, j])
            for j in range(C):
                if cnt[j] == H:
                    res[j] = w[i, j]

    @njit(cache=True, nogil=True, fastmath=True)
    def _med_tiempo(S, out):
        """Mediana a lo largo del tiempo (cada fila por separado), borde 'reflect' de scipy: ... c b a | a b c ..."""
        F, T = S.shape
        pad = np.zeros(T + 2 * H + C, np.float32)
        w = np.zeros((K, C), np.float32)
        res = np.zeros(C, np.float32)
        for f in range(F):
            for i in range(H):
                pad[H - 1 - i] = S[f, i]
                pad[H + T + i] = S[f, T - 1 - i]
            for t in range(T):
                pad[H + t] = S[f, t]
            for t0 in range(0, T, C):
                m = min(C, T - t0)
                for i in range(K):
                    for j in range(m):
                        w[i, j] = pad[t0 + j + i]
                _mediana_filas(w, res)
                for j in range(m):
                    out[f, t0 + j] = res[j]

    @njit(cache=True, nogil=True, fastmath=True)
    def _med_frec(S, out):
        """Mediana a lo largo de la frecuencia (cada columna por separado), mismo borde 'reflect'."""
        F, T = S.shape
        w = np.zeros((K, C), np.float32)
        res = np.zeros(C, np.float32)
        for f in range(F):
            for t0 in range(0, T, C):
                m = min(C, T - t0)
                for i in range(K):
                    ff = f + i - H
                    if ff < 0:
                        ff = -ff - 1
                    if ff >= F:
                        ff = 2 * F - 1 - ff
                    for j in range(m):
                        w[i, j] = S[ff, t0 + j]
                _mediana_filas(w, res)
                for j in range(m):
                    out[f, t0 + j] = res[j]

    disponible = True
except Exception:                                             # noqa: BLE001  (sin numba: se usa scipy)
    disponible = False


def mediana_tiempo(S: np.ndarray) -> np.ndarray:
    """Igual que ndimage.median_filter(S, size=(1, 17), mode="reflect")."""
    S = np.ascontiguousarray(S, dtype=np.float32)
    if S.shape[1] < K or not disponible:                          # ventana más larga que el bloque: reflejo múltiple, lo resuelve scipy
        from scipy import ndimage
        return ndimage.median_filter(S, size=(1, K), mode="reflect")
    out = np.empty_like(S)
    _med_tiempo(S, out)
    return out


def mediana_frecuencia(S: np.ndarray) -> np.ndarray:
    """Igual que ndimage.median_filter(S, size=(17, 1), mode="reflect")."""
    S = np.ascontiguousarray(S, dtype=np.float32)
    if S.shape[0] < K or not disponible:
        from scipy import ndimage
        return ndimage.median_filter(S, size=(K, 1), mode="reflect")
    out = np.empty_like(S)
    _med_frec(S, out)
    return out
