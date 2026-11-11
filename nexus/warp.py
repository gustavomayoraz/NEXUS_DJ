"""Estiramiento de tiempo y cambio de tono con mapa temporal arbitrario (vocoder de fase con bloqueo en transitorios).

`render_capa` recibe una capa de audio y, para cada muestra de salida, la posición de origen y la razón de tono.
Así todas las capas de un tema se estiran EXACTAMENTE igual (siguen alineadas) y, donde el mapa vale 1:1,
el resultado es idéntico bit a bit al original (empalmes inaudibles).
"""
from __future__ import annotations

import numpy as np


def _catmull(y1: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Remuestreo cúbico (Catmull-Rom) de y1 (n,) en las posiciones fraccionarias `pos`."""
    n = len(y1)
    i = np.floor(pos).astype(np.int64)
    t = (pos - i).astype(np.float32)
    idx = np.clip(np.stack([i - 1, i, i + 1, i + 2]), 0, n - 1)
    p0, p1, p2, p3 = y1[idx[0]], y1[idx[1]], y1[idx[2]], y1[idx[3]]
    t2, t3 = t * t, t * t * t
    return (0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)).astype(np.float32)


def _vocoder(D: np.ndarray, pos: np.ndarray, hop: int, n_fft: int, transitorio: np.ndarray, reset_uno: np.ndarray) -> np.ndarray:
    """D: (ch, bins, F) STFT de origen. pos[m] = posición (en cuadros, fraccionaria) de origen del cuadro de salida m.

    Devuelve el STFT de salida (ch, bins, M). `transitorio[i]` marca cuadros de ataque (se reinicia la fase ahí);
    `reset_uno[m]` fuerza fase original cuando el mapa es 1:1 y cae justo sobre un cuadro (reconstrucción exacta).
    """
    ch, nb, F = D.shape
    M = len(pos)
    omega = 2 * np.pi * hop * np.arange(nb) / n_fft
    ang = np.angle(D)
    mag = np.abs(D)
    sal = np.empty((ch, nb, M), np.complex64)
    fase = ang[:, :, 0].copy()
    ult_reset = -1
    for m in range(M):
        p = pos[m]
        i = min(max(int(np.floor(p)), 0), F - 2)
        fr = min(max(p - i, 0.0), 1.0)
        a0, a1 = ang[:, :, i], ang[:, :, i + 1]
        mg = (1.0 - fr) * mag[:, :, i] + fr * mag[:, :, i + 1]
        ni = min(int(round(p)), F - 1)
        if m == 0 or reset_uno[m]:
            fase = ang[:, :, ni].copy()
            ult_reset = ni
        elif transitorio[ni] and ni != ult_reset:                      # ataque: se recupera la fase original (golpes nítidos)
            fase = ang[:, :, ni].copy()
            ult_reset = ni
        else:
            d = a1 - a0 - omega[None, :]
            d -= 2 * np.pi * np.round(d / (2 * np.pi))
            fase = fase + omega[None, :] + d
        sal[:, :, m] = mg * np.exp(1j * fase)
    return sal


def render_capa(capa: np.ndarray, s_seg0: int, src_de_n: np.ndarray, pitch_de_n: np.ndarray | None,
                n_fft: int = 2048, hop: int = 512, ancla: str = "fin") -> np.ndarray:
    """Renderiza una capa con mapa temporal.

    capa        : (2, L) audio de origen; la muestra 0 es la muestra absoluta `s_seg0` del tema.
    src_de_n    : (N,) muestra de origen (absoluta, fraccionaria) que debe sonar en la muestra de salida n.
    pitch_de_n  : (N,) razón de tono (1 = sin cambio) o None. Debe valer 1 al final para empalmar con el original.
    ancla       : "inicio" o "fin": extremo donde el mapa es 1:1 y debe empalmar bit a bit con el audio original.
    Devuelve (2, N) float32.
    """
    import librosa
    N = len(src_de_n)
    sin_tono = pitch_de_n is None or np.allclose(pitch_de_n, 1.0)
    src_arr = np.asarray(src_de_n, np.float64)
    ia = 0 if ancla == "inicio" else N - 1
    if sin_tono:
        t1 = np.arange(N, dtype=np.float64)
    else:
        p = np.asarray(pitch_de_n, np.float64)
        # t1(n): tiempo en la línea del vocoder; en el extremo ancla coincide con n (empalme exacto con el original)
        acum = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]))])
        t1 = (ia + acum - acum[ia]).astype(np.float64)
    # los cuadros del vocoder caen justo sobre cuadros del origen en el extremo ancla (así el 1:1 es exacto)
    r = int((s_seg0 - int(round(src_arr[ia])) + int(round(t1[ia]))) % hop)
    K = int(np.ceil(max(0.0, -t1[0]) / hop)) + n_fft // hop + 1      # cuadros de sobra a la izquierda (el solape del inicio queda completo)
    T0 = r - K * hop
    n_t1 = int(np.ceil(t1[-1] - T0)) + 1
    grid = T0 + np.arange(n_t1, dtype=np.float64)
    n_de_t1 = np.interp(grid, t1, np.arange(N, dtype=np.float64))
    src1_de_t1 = np.interp(n_de_t1, np.arange(N, dtype=np.float64), src_arr)
    antes = grid < t1[0]                                              # antes del inicio: se continúa con la pendiente inicial
    if antes.any():
        pend0 = (src_arr[min(N - 1, 64)] - src_arr[0]) / max(1e-9, (t1[min(N - 1, 64)] - t1[0]))
        src1_de_t1[antes] = src_arr[0] + (grid[antes] - t1[0]) * pend0
    # cuadros de salida del vocoder: centros en T0 + m*hop
    M = int(np.ceil(n_t1 / hop)) + n_fft // hop + 2
    centros = T0 + np.arange(M, dtype=np.float64) * hop
    s_c = np.interp(centros, grid, src1_de_t1)
    izq, der = centros < grid[0], centros > grid[-1]
    s_c[izq] = src1_de_t1[0] + (centros[izq] - grid[0])               # fuera del rango: se continúa con pendiente 1
    s_c[der] = src1_de_t1[-1] + (centros[der] - grid[-1])
    pos = (s_c - s_seg0) / hop
    L = capa.shape[1]
    D = np.stack([librosa.stft(capa[c], n_fft=n_fft, hop_length=hop) for c in range(capa.shape[0])]).astype(np.complex64)
    F = D.shape[2]
    pos = np.clip(pos, 0.0, F - 1.001)
    # ¿mapa 1:1 sobre un cuadro exacto?  pendiente 1 y posición entera
    pend = np.gradient(pos)
    reset_uno = (np.abs(pend - 1.0) < 1e-4) & (np.abs(pos - np.round(pos)) < 1e-4)
    # transitorios del origen (flujo espectral del canal medio)
    mm = np.abs(D).mean(axis=0)
    flujo = np.maximum(0.0, np.diff(mm, axis=1, prepend=mm[:, :1])).sum(axis=0)
    ref = np.convolve(flujo, np.ones(9) / 9, mode="same") + 1e-9
    transitorio = (flujo > 2.2 * ref) & (flujo > 0.15 * np.percentile(flujo, 99))
    S = _vocoder(D, pos, hop, n_fft, transitorio, reset_uno)
    y1 = np.stack([librosa.istft(S[c], hop_length=hop, n_fft=n_fft) for c in range(S.shape[0])]).astype(np.float32)
    # y1[:, j] corresponde al tiempo T0 + j
    pos1 = t1 - T0
    sal = np.empty((y1.shape[0], N), np.float32)
    if sin_tono:
        j0 = int(round(-T0))
        sal[:] = y1[:, j0:j0 + N] if y1.shape[1] >= j0 + N else np.pad(y1[:, j0:], ((0, 0), (0, j0 + N - y1.shape[1])))
        return sal
    for c in range(y1.shape[0]):
        sal[c] = _catmull(y1[c], pos1)
    return sal
