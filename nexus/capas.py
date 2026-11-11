"""Separación de un tema en 4 capas sumables (batería · bajo · melodía/voz · ambiente) sin redes neuronales.

Se usa HPSS (armónico/percusivo con máscaras suaves), un cruce de graves, y la "centralidad" estéreo
(lo que está en el centro suele ser voz o melodía principal). Las máscaras suman 1 en cada punto del espectro,
así que  bateria + bajo + lead + ambiente == audio original  (reconstrucción exacta).
"""
from __future__ import annotations

import numpy as np

CAPAS = ("bateria", "bajo", "lead", "ambiente")
NOMBRES_CAPAS = {"bateria": "Batería", "bajo": "Bajo", "lead": "Voz/Melodía", "ambiente": "Ambiente"}

N_FFT = 2048
HOP = 512


def _escalon(f: np.ndarray, f0: float, ancho: float) -> np.ndarray:
    """Rampa suave 0→1 centrada en f0 (en Hz)."""
    x = np.clip((f - (f0 - ancho)) / (2 * ancho), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x)


def separar(y: np.ndarray, sr: int, corte_bajo: float = 150.0) -> dict[str, np.ndarray]:
    """y: (2, n) float32. Devuelve {capa: (2, n)} cuya suma es y."""
    import librosa
    n = y.shape[1]
    XL = librosa.stft(y[0], n_fft=N_FFT, hop_length=HOP)
    XR = librosa.stft(y[1], n_fft=N_FFT, hop_length=HOP)
    XM, XS = (XL + XR) / 2, (XL - XR) / 2
    S = np.abs(XM)
    mh, mp = librosa.decompose.hpss(S, kernel_size=17, power=2.0, mask=True, margin=1.0)
    tot = mh + mp + 1e-9
    mh, mp = mh / tot, mp / tot                                       # suman exactamente 1
    f = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)[:, None]
    m_grave = _escalon(f, corte_bajo, 40.0)                           # 1 en graves
    m_medio = (_escalon(f, 220.0, 70.0) * (1.0 - _escalon(f, 6500.0, 1500.0)))
    e_m, e_s = np.abs(XM) ** 2, np.abs(XS) ** 2
    centro = e_m / (e_m + e_s + 1e-12)                                # 1 = centrado, 0 = muy abierto
    c = np.clip((centro - 0.35) / 0.55, 0.0, 1.0)

    m_bajo = mh * m_grave
    m_arm = mh * (1.0 - m_grave)
    m_lead = m_arm * m_medio * c
    m_amb = m_arm - m_lead
    m_dr = mp
    capas = {}
    for nombre, mascara in (("bateria", m_dr), ("bajo", m_bajo), ("lead", m_lead), ("ambiente", m_amb)):
        sal = np.empty_like(y)
        sal[0] = librosa.istft(XL * mascara, hop_length=HOP, n_fft=N_FFT, length=n)
        sal[1] = librosa.istft(XR * mascara, hop_length=HOP, n_fft=N_FFT, length=n)
        capas[nombre] = sal
    return capas
