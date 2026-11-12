"""Análisis musical de un tema: rejilla de beats, compases, tonalidad y perfil de cada compás.

El resultado alimenta al planificador de fusión: qué compás es el mejor para salir de un tema y entrar en otro,
qué tanta voz/graves/batería hay en cada compás y qué desplazamiento de tono armoniza dos temas.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal

from .audio_io import a_mono, bajo_db, cargar_estereo, sonoridad_db
from . import basedatos
from .config import SR, carpeta_datos

VERSION = 4
SR_AN = 22050
HOP = 512
N_FFT = 2048
OBJETIVO_DB = -14.0                    # sonoridad objetivo de cada tema
BAJO_REF_DB = -18.3                   # nivel de «golpe» (bajo/bombo) de referencia, ya con la ganancia aplicada
BAJO_PESO = 0.4                       # cuánto del desvío de golpe se corrige (0 = nada, 1 = igualar el golpe del todo)
BAJO_TOPE_DB = 2.5                    # corrección máxima por golpe, en dB

_CAMELOT_MAYOR = {0: "8B", 1: "3B", 2: "10B", 3: "5B", 4: "12B", 5: "7B", 6: "2B", 7: "9B", 8: "4B", 9: "11B", 10: "6B", 11: "1B"}
_CAMELOT_MENOR = {0: "5A", 1: "12A", 2: "7A", 3: "2A", 4: "9A", 5: "4A", 6: "11A", 7: "6A", 8: "1A", 9: "8A", 10: "3A", 11: "10A"}
_PERFIL_MAYOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_PERFIL_MENOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTAS = ["Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]


@dataclass
class Analisis:
    ruta: str
    nombre: str
    duracion: float
    bpm: float
    beats: np.ndarray                  # segundos de cada beat (cubre todo el tema)
    fase: int                          # índice (0..3) del beat que es el "uno" del compás
    tonica: int
    menor: bool
    camelot: str
    afinacion: float                   # desvío de afinación en semitonos (−0.5 … 0.5)
    chroma_total: np.ndarray           # (12,)
    ganancia_db: float                 # ganancia para igualar la sonoridad de todos los temas
    sonoridad: float
    ultimo_compas: int                 # último compás con música (para no planificar sobre el silencio)
    chroma: np.ndarray = field(default_factory=lambda: np.zeros((0, 12)))   # (compases, 12)
    energia: np.ndarray = field(default_factory=lambda: np.zeros(0))        # 0..1 por compás
    voz: np.ndarray = field(default_factory=lambda: np.zeros(0))            # 0..1
    bateria: np.ndarray = field(default_factory=lambda: np.zeros(0))        # 0..1
    bajo: np.ndarray = field(default_factory=lambda: np.zeros(0))           # 0..1
    onda: np.ndarray = field(default_factory=lambda: np.zeros(0))           # pico por 0,05 s para dibujar

    # ---- rejilla ----
    @property
    def compases(self) -> np.ndarray:
        """Índices de beat donde empieza cada compás."""
        return np.arange(self.fase, len(self.beats) - 4, 4)

    @property
    def n_compases(self) -> int:
        return len(self.compases)

    def tiempo_compas(self, i: int) -> float:
        return float(self.beats[self.compases[int(np.clip(i, 0, self.n_compases - 1))]])

    @property
    def periodo(self) -> float:
        return 60.0 / self.bpm

    @property
    def clave_texto(self) -> str:
        return f"{NOTAS[self.tonica]} {'menor' if self.menor else 'mayor'}"

    # ---- (de)serialización ----
    def a_dict(self) -> dict:
        d = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        d["_v"] = VERSION
        return d

    @staticmethod
    def de_dict(d: dict) -> "Analisis":
        d = dict(d)
        d.pop("_v", None)
        for k in ("beats", "chroma_total", "chroma", "energia", "voz", "bateria", "bajo", "onda"):
            d[k] = np.asarray(d[k], dtype=np.float64 if k in ("beats",) else np.float32)
        return Analisis(**d)


def leer_cache(ruta: str) -> "Analisis | None":
    """Análisis ya guardado en la base de datos (datos/nexus.db) o None si el tema todavía no se analizó."""
    try:
        d = basedatos.buscar(ruta, VERSION)
        return Analisis.de_dict(d) if d is not None else None
    except (ValueError, TypeError, KeyError):
        return None


def _guardar_cache(a: "Analisis") -> None:
    basedatos.guardar(a.ruta, a.a_dict(), VERSION)


# --------------------------------------------------------------------------------------- rejilla de beats
def _rejilla_beats(y22: np.ndarray, sr: int, s_perc: np.ndarray, dur: float) -> tuple[np.ndarray, float]:
    """Beats afinados al milisegundo. Si el tema es de tempo fijo (electrónica) devuelve una rejilla perfecta."""
    import librosa
    mel = librosa.feature.melspectrogram(S=s_perc ** 2, sr=sr, n_mels=64)
    oenv = librosa.onset.onset_strength(S=librosa.power_to_db(mel), sr=sr, hop_length=HOP)
    tempo, fr = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=HOP, tightness=120, start_bpm=124, trim=False)
    t_beats = librosa.frames_to_time(np.asarray(fr), sr=sr, hop_length=HOP)
    tempo = float(np.atleast_1d(tempo)[0])
    if len(t_beats) < 8:                                           # no encontró pulso: rejilla plana con el tempo estimado
        per = 60.0 / max(60.0, min(200.0, tempo or 120.0))
        return np.arange(0.0, dur, per), per
    # afinado: buscar el pico real del ataque (hop de 128 muestras = 5,8 ms) cerca de cada beat.
    # El bombo (graves) manda: así la rejilla cae en los golpes fuertes y no en los contratiempos de los platillos.
    sos_g = signal.butter(4, 150.0, "lowpass", fs=sr, output="sos")
    o_g = _flujo_lineal(signal.sosfilt(sos_g, y22).astype(np.float32), sr)        # bombo: aumento de energía en graves (lineal)
    o_f = librosa.onset.onset_strength(y=y22, sr=sr, hop_length=128)
    o2 = o_f / (o_f.max() + 1e-9) + 1.5 * o_g / (o_g.max() + 1e-9)
    t2 = librosa.frames_to_time(np.arange(len(o2)), sr=sr, hop_length=128)
    ref = np.empty_like(t_beats)
    for i, t in enumerate(t_beats):
        a, b = np.searchsorted(t2, [t - 0.04, t + 0.04])
        if b - a >= 3 and o2[a:b].max() > 0:
            ref[i] = t2[a + int(np.argmax(o2[a:b]))]
        else:
            ref[i] = t
    # --- tempo fijo (electrónica, pop de estudio): búsqueda global del período y la fase que mejor encajan en los ataques
    p0 = 60.0 / max(60.0, min(200.0, tempo))
    from scipy.ndimage import gaussian_filter1d
    o_busq = gaussian_filter1d(o_f / (o_f.max() + 1e-9) + 3.0 * o_g / (o_g.max() + 1e-9), 1.5)    # suavizada: el pico ancho tolera el paso de búsqueda
    p, a0, aciertos = _rejilla_global(o_busq, t2, p0, dur)
    if aciertos >= 0.72:
        a0, p = _fase_bombo(o_g, t2, a0, p)
        n0 = int(np.floor(-a0 / p)) if a0 > 0 else int(np.ceil(-a0 / p))
        kk = np.arange(n0, int((dur - a0) / p) + 2)
        beats = a0 + p * kk
        return beats[beats >= 0], float(p)
    # tempo con variaciones: se respetan los beats detectados, rellenando extremos con el período local
    per = np.diff(ref)
    per_m = signal.medfilt(per, 5 if len(per) >= 5 else 1)
    per_m = np.clip(per_m, np.median(per) * 0.85, np.median(per) * 1.15)
    ref2 = np.concatenate([[ref[0]], ref[0] + np.cumsum(per_m)])
    ref2 = ref                                                                  # se conservan los beats afinados
    izq = []
    t = ref2[0] - per_m[0]
    while t > 0:
        izq.append(t)
        t -= per_m[0]
    der = []
    t = ref2[-1] + per_m[-1]
    while t < dur:
        der.append(t)
        t += per_m[-1]
    return np.concatenate([np.array(izq[::-1]), ref2, np.array(der)]), float(np.median(per_m))


def _flujo_lineal(y: np.ndarray, sr: int) -> np.ndarray:
    """Ataques de un audio de graves: flujo espectral de magnitud lineal (la escala en dB tapa al bombo cuando hay bajo sostenido)."""
    import librosa
    X = librosa.stft(y, n_fft=1024, hop_length=128)
    m = np.abs(X[:12])                                                       # hasta ~260 Hz (solo se calcula el módulo de esas filas)
    del X
    d = np.maximum(0.0, np.diff(m, axis=1, prepend=m[:, :1]))
    return d.sum(axis=0)


def _rejilla_global(o: np.ndarray, t: np.ndarray, p0: float, dur: float) -> tuple[float, float, float]:
    """Mejor (período, fase) de una rejilla fija sobre la curva de ataques `o` y fracción de beats que caen sobre un ataque."""
    t_fin = min(dur, t[-1])

    def puntaje(pp: float, fases: np.ndarray) -> tuple[np.ndarray, int]:
        kk = np.arange(0, max(4, int((t_fin - 1.0) / pp)))
        pos = fases[:, None] + pp * kk[None, :]
        return np.interp(pos, t, o).mean(axis=1), len(kk)

    mejor = (-1.0, p0, 0.0)
    fases = np.arange(0.0, p0, 0.005)
    for pp in p0 * np.linspace(0.96, 1.04, 321):
        v, _ = puntaje(pp, fases)
        i = int(np.argmax(v))
        if v[i] > mejor[0]:
            mejor = (v[i], pp, fases[i])
    _, p1, a1 = mejor
    mejor = (-1.0, p1, a1)
    for pp in p1 * np.linspace(0.9985, 1.0015, 61):                   # ajuste fino
        v, _ = puntaje(pp, a1 + np.linspace(-0.008, 0.008, 33))
        i = int(np.argmax(v))
        if v[i] > mejor[0]:
            mejor = (v[i], pp, a1 + np.linspace(-0.008, 0.008, 33)[i])
    _, p2, a2 = mejor
    kk = np.arange(0, max(4, int((t_fin - 1.0) / p2)))
    pasos = max(1, int(0.015 / (t[1] - t[0])))
    umbral = 0.35 * np.percentile(o, 90)
    golpes = 0
    for k in kk:
        c = int(np.searchsorted(t, a2 + p2 * k))
        seg = o[max(0, c - pasos):c + pasos + 1]
        golpes += int(len(seg) > 0 and seg.max() >= umbral)
    return float(p2), float(a2), golpes / max(1, len(kk))


def _fase_bombo(o_g: np.ndarray, t: np.ndarray, a0: float, p: float) -> tuple[float, float]:
    """Si los graves caen a mitad de beat (contratiempo), corre la rejilla medio beat para que pise el bombo."""
    kk = np.arange(0, int((t[-1] - a0) / p) - 1)
    v0 = np.interp(a0 + p * kk, t, o_g).mean()
    v1 = np.interp(a0 + p * kk + p / 2, t, o_g).mean()
    return (a0 + p / 2, p) if v1 > 1.25 * v0 else (a0, p)


def _normalizar_octava(beats: np.ndarray, per: float, o_env_fn) -> tuple[np.ndarray, float]:
    """Deja el tempo en 80–165 BPM (duplica o divide la rejilla si hace falta)."""
    bpm = 60.0 / per
    while bpm < 80.0:
        medios = (beats[:-1] + beats[1:]) / 2
        beats = np.sort(np.concatenate([beats, medios]))
        per /= 2
        bpm *= 2
    while bpm >= 165.0:
        par = o_env_fn(beats[0::2]) >= o_env_fn(beats[1::2])
        beats = beats[0::2] if par else beats[1::2]
        per *= 2
        bpm /= 2
    return beats, per


# --------------------------------------------------------------------------------------- tonalidad
def _clave_de_chroma(c: np.ndarray) -> tuple[int, bool]:
    mejor = (-9.0, 0, False)
    for tonica in range(12):
        for perfil, menor in ((_PERFIL_MAYOR, False), (_PERFIL_MENOR, True)):
            r = np.corrcoef(c, np.roll(perfil, tonica))[0, 1]
            if r > mejor[0]:
                mejor = (r, tonica, menor)
    return mejor[1], mejor[2]


def camelot_de(tonica: int, menor: bool) -> str:
    return (_CAMELOT_MENOR if menor else _CAMELOT_MAYOR)[tonica % 12]


def _hpss_rapido(S: np.ndarray, kernel: int = 17, power: float = 2.0, margin: float = 1.0, bloque: int = 512):
    """Igual que librosa.decompose.hpss(mask=True) pero por bloques de columnas (con el margen que necesita el filtro)
    y repartido en hilos. El resultado es idéntico; la memoria temporal queda acotada (no se copia todo el espectrograma
    varias veces) y los hilos usan los núcleos disponibles."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    from scipy import ndimage
    import librosa
    hilos = max(1, int(os.environ.get("NEXUS_HILOS", "2")))
    n_t = S.shape[1]
    h = kernel // 2
    mh = np.empty_like(S)
    mp = np.empty_like(S)

    def trabajo(c0: int) -> None:
        c1 = min(n_t, c0 + bloque)
        a, b = max(0, c0 - h), min(n_t, c1 + h)                                   # columnas vecinas que el filtro necesita
        arm = ndimage.median_filter(S[:, a:b], size=(1, kernel), mode="reflect")[:, c0 - a:c1 - a]
        per = ndimage.median_filter(S[:, c0:c1], size=(kernel, 1), mode="reflect")
        mh[:, c0:c1] = librosa.util.softmask(arm, per * margin, power=power)
        mp[:, c0:c1] = librosa.util.softmask(per, arm * margin, power=power)

    with ThreadPoolExecutor(hilos) as ex:
        list(ex.map(trabajo, range(0, n_t, bloque)))
    return mh, mp


# --------------------------------------------------------------------------------------- análisis completo
def analizar(ruta: str, y: "np.ndarray | None" = None, usar_cache: bool = True) -> Analisis:
    """Analiza un tema completo (tarda unos segundos). `y` opcional: audio ya cargado (2, n) a SR."""
    import librosa
    ruta = str(ruta)
    if usar_cache:
        c = leer_cache(ruta)
        if c is not None:
            return c
    if y is None:
        y = cargar_estereo(ruta)
    dur = y.shape[1] / SR
    if dur < 8.0:
        raise ValueError("El tema es demasiado corto para fusionarlo (menos de 8 s).")
    son = sonoridad_db(y)
    g = float(np.clip(OBJETIVO_DB - son, -12.0, 12.0)) if son > -60 else 0.0
    if son > -60:                                                  # mismo nivel de golpe en todos los temas: se corrige parte del desvío de bajo/bombo
        bj = bajo_db(y)
        if bj > -80:
            g = float(np.clip(g + np.clip(BAJO_PESO * (BAJO_REF_DB - (bj + g)), -BAJO_TOPE_DB, BAJO_TOPE_DB), -12.0, 12.0))

    # --- onda para dibujar (pico por 0,05 s); se hace ya mismo para poder soltar el audio completo
    paso = int(SR * 0.05)
    nb = y.shape[1] // paso
    onda = np.abs(mono_44(y)[:nb * paso]).reshape(nb, paso).max(axis=1).astype(np.float32) if nb > 0 else np.zeros(1, np.float32)

    # --- espectrogramas a 22 kHz (estéreo: mid/side para saber qué está al centro). Se suelta cada cosa apenas deja de servir:
    #     con temas largos y poca RAM, los picos de memoria son lo que cuelga la PC.
    y22 = signal.resample_poly(y, 1, 2, axis=1).astype(np.float32)
    del y
    mono = y22.mean(axis=0)
    # la STFT es lineal: mid/side se calcula en el tiempo y solo se guardan 2 espectrogramas complejos a la vez (no 4)
    XM = librosa.stft((y22[0] + y22[1]) * 0.5, n_fft=N_FFT, hop_length=HOP)
    S = np.abs(XM)
    del XM
    XS = librosa.stft((y22[0] - y22[1]) * 0.5, n_fft=N_FFT, hop_length=HOP)
    del y22
    pm = S ** 2
    centro = pm / (pm + np.abs(XS) ** 2 + 1e-9)
    del XS, pm
    mh, mp = _hpss_rapido(S, kernel=17, power=2.0, margin=1.0)
    SH, SP = S * mh, S * mp
    del mh, mp
    frec = librosa.fft_frequencies(sr=SR_AN, n_fft=N_FFT)
    banda_voz = (frec >= 300) & (frec <= 3400)
    e_tot = (S ** 2).sum(axis=0)
    e_voz = ((SH[banda_voz] * centro[banda_voz]) ** 2).sum(axis=0)
    e_baj = (S[frec < 150] ** 2).sum(axis=0)
    e_per = (SP ** 2).sum(axis=0)
    del centro
    n_fr = S.shape[1]
    del S

    beats, per = _rejilla_beats(mono, SR_AN, SP, dur)
    # peso de cada beat = energía de graves + ataque (sirve para el "uno" y para elegir la paridad)
    bajo_f = SP[frec < 150].sum(axis=0) + 0.3 * SP.sum(axis=0) / 8
    del SP
    t_fr = librosa.frames_to_time(np.arange(n_fr), sr=SR_AN, hop_length=HOP)

    def fuerza(tb: np.ndarray) -> float:
        idx = np.clip(np.searchsorted(t_fr, tb), 0, len(bajo_f) - 1)
        return float(bajo_f[idx].mean()) if len(idx) else 0.0

    beats, per = _normalizar_octava(beats, per, fuerza)
    bpm = 60.0 / per

    # --- el "uno" del compás: la fase de 4 beats con más graves/ataque + cambio armónico
    idx_b = np.clip(np.searchsorted(t_fr, beats), 0, len(bajo_f) - 1)
    punt = np.array([bajo_f[idx_b[f::4]].mean() if len(idx_b[f::4]) else 0 for f in range(4)])
    fase = int(np.argmax(punt))

    # --- tonalidad y afinación
    SH2 = SH ** 2
    del SH
    try:
        afin = float(librosa.estimate_tuning(S=SH2, sr=SR_AN, n_fft=N_FFT))
    except Exception:
        afin = 0.0
    chroma_f = librosa.feature.chroma_stft(S=SH2, sr=SR_AN, n_fft=N_FFT, tuning=afin)        # (12, frames)
    del SH2
    chroma_t = chroma_f.mean(axis=1)
    tonica, menor = _clave_de_chroma(chroma_t)

    # --- perfil por compás
    comp = np.arange(fase, len(beats) - 4, 4)
    fr_c = np.clip(np.searchsorted(t_fr, beats[comp]), 0, len(t_fr) - 1)
    fr_fin = np.append(fr_c[1:], min(len(t_fr), fr_c[-1] + int(4 * per * SR_AN / HOP))) if len(fr_c) else fr_c
    n = len(comp)
    chroma_c = np.zeros((n, 12), np.float32)
    energia = np.zeros(n, np.float32)
    voz = np.zeros(n, np.float32)
    bateria = np.zeros(n, np.float32)
    bajo = np.zeros(n, np.float32)
    for i in range(n):
        a, b = int(fr_c[i]), max(int(fr_fin[i]), int(fr_c[i]) + 1)
        v = chroma_f[:, a:b].mean(axis=1)
        chroma_c[i] = v / (np.linalg.norm(v) + 1e-9)
        et = e_tot[a:b].sum() + 1e-9
        energia[i] = np.sqrt(et / (b - a))
        voz[i] = e_voz[a:b].sum() / et
        bateria[i] = e_per[a:b].sum() / et
        bajo[i] = e_baj[a:b].sum() / et

    def norm(v: np.ndarray, p_lo: float = 10, p_hi: float = 90) -> np.ndarray:
        if len(v) == 0:
            return v
        lo, hi = np.percentile(v, p_lo), np.percentile(v, p_hi)
        return np.clip((v - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)

    energia = (energia / (np.percentile(energia, 95) + 1e-9)).clip(0, 1.2).astype(np.float32) if n else energia
    voz, bateria, bajo = norm(voz), norm(bateria), norm(bajo)
    audible = np.where(energia > 0.04)[0]
    ultimo = int(audible[-1]) if len(audible) else max(0, n - 1)

    res = Analisis(ruta=ruta, nombre=Path(ruta).stem, duracion=float(dur), bpm=float(bpm), beats=np.asarray(beats, np.float64),
                   fase=fase, tonica=int(tonica), menor=bool(menor), camelot=camelot_de(tonica, menor), afinacion=afin,
                   chroma_total=(chroma_t / (np.linalg.norm(chroma_t) + 1e-9)).astype(np.float32), ganancia_db=g,
                   sonoridad=float(son), ultimo_compas=ultimo, chroma=chroma_c, energia=energia, voz=voz,
                   bateria=bateria, bajo=bajo, onda=onda)
    if usar_cache:
        _guardar_cache(res)
    return res


def mono_44(y: np.ndarray) -> np.ndarray:
    return a_mono(y)
