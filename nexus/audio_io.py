"""Carga de audio a estéreo float32 a SR fijo y medición de sonoridad."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal

from .config import SR


def cargar_estereo(ruta: str | Path, sr: int = SR) -> np.ndarray:
    """Devuelve el audio como array (2, n) float32 a `sr` Hz. Lanza OSError/ValueError si no se puede leer."""
    ruta = str(ruta)
    y = None
    sr_orig = sr
    try:
        import soundfile as sf
        datos, sr_orig = sf.read(ruta, dtype="float32", always_2d=True)
        y = datos.T
    except Exception:                                   # formato que libsndfile no abre (m4a, wma…): librosa/audioread
        try:
            import librosa
            y, sr_orig = librosa.load(ruta, sr=None, mono=False)
            y = np.atleast_2d(y).astype(np.float32)
        except Exception as e:
            raise OSError(f"No se pudo leer el archivo: {e or type(e).__name__}") from e
    if y.shape[0] == 1:
        y = np.vstack([y, y])
    elif y.shape[0] > 2:
        y = y[:2]
    if int(sr_orig) != sr:
        g = np.gcd(int(sr_orig), sr)
        y = signal.resample_poly(y, sr // g, int(sr_orig) // g, axis=1).astype(np.float32)
    if not np.isfinite(y).all():
        y = np.nan_to_num(y)
    return np.ascontiguousarray(y, dtype=np.float32)


def a_mono(y: np.ndarray) -> np.ndarray:
    return y.mean(axis=0) if y.ndim == 2 else y


def sonoridad_db(y: np.ndarray, sr: int = SR) -> float:
    """Sonoridad aproximada (RMS con filtro tipo K y compuerta de silencio), en dB."""
    m = a_mono(y)
    if m.size < sr:
        return -70.0
    sos_hp = signal.butter(2, 60.0, "highpass", fs=sr, output="sos")
    sos_sh = signal.butter(1, 1700.0, "highpass", fs=sr, output="sos")
    f = signal.sosfilt(sos_hp, m)
    f = f + 0.6 * signal.sosfilt(sos_sh, f)              # realce de agudos como el filtro K
    paso = sr // 2
    n = f.size // paso
    if n < 2:
        return -70.0
    bloques = (f[:n * paso].reshape(n, paso) ** 2).mean(axis=1)
    db = 10 * np.log10(bloques + 1e-12)
    util = bloques[db > -50.0]
    if util.size == 0:
        return -70.0
    corte = 10 * np.log10(util.mean() + 1e-12) - 10.0     # segunda compuerta relativa
    util = bloques[db > max(-50.0, corte)]
    return float(10 * np.log10(util.mean() + 1e-12))


def bajo_db(y: np.ndarray, sr: int = SR) -> float:
    """Nivel del «golpe» (bombo y bajo, 40–120 Hz) en los tramos fuertes del tema: percentil 90 de bloques de 1 s, en dB."""
    m = a_mono(y)
    n = m.size // sr
    if n < 2:
        return -70.0
    sos = signal.butter(4, [40.0, 120.0], "bandpass", fs=sr, output="sos")
    f = signal.sosfilt(sos, m[:n * sr])
    b = (f.reshape(n, sr) ** 2).mean(axis=1)
    return float(10 * np.log10(np.percentile(b, 90) + 1e-12))


class AudioParcial:
    """Un trozo de un tema que se comporta como el tema entero para `_recorte` (shape y audio[:, a:b]); fuera del trozo da ceros.
    Sirve para fusionar sin decodificar los dos temas completos (mucha menos memoria y tiempo)."""

    def __init__(self, n_total: int, lo: int, datos: np.ndarray) -> None:
        self.shape = (datos.shape[0], int(n_total))
        self.lo = int(lo)
        self.datos = datos

    def __getitem__(self, idx):
        _, sl = idx
        a, b = int(sl.start or 0), int(sl.stop)
        sal = np.zeros((self.shape[0], max(0, b - a)), np.float32)
        x0, x1 = max(a, self.lo), min(b, self.lo + self.datos.shape[1])
        if x1 > x0:
            sal[:, x0 - a:x1 - a] = self.datos[:, x0 - self.lo:x1 - self.lo]
        return sal


def cargar_ventana(ruta: str | Path, lo: int, hi: int, sr: int = SR):
    """Decodifica solo las muestras [lo, hi) (a `sr` Hz) y devuelve un AudioParcial. Si el formato no permite saltar
    (m4a, wma…) o algo falla, decodifica el tema entero como antes."""
    try:
        import soundfile as sf
        with sf.SoundFile(str(ruta)) as f:
            sr0, total = f.samplerate, f.frames
            if total <= 0:
                raise ValueError("sin duración")
            n_total = int(round(total * sr / sr0))
            lo = max(0, int(lo))
            hi = min(n_total, int(hi))
            if hi <= lo:
                raise ValueError("ventana vacía")
            g = int(np.gcd(sr0, sr))
            pad = 2048                                                    # unas muestras de más para que el remuestreo no deje bordes
            o0 = max(0, int(lo * sr0 / sr) - pad)
            o1 = min(total, int(hi * sr0 / sr) + pad)
            f.seek(o0)
            y = f.read(o1 - o0, dtype="float32", always_2d=True).T
        if y.shape[0] == 1:
            y = np.vstack([y, y])
        elif y.shape[0] > 2:
            y = y[:2]
        if sr0 != sr:
            y = signal.resample_poly(y, sr // g, sr0 // g, axis=1).astype(np.float32)
        ini = int(round(o0 * sr / sr0))
        y = np.nan_to_num(np.ascontiguousarray(y, dtype=np.float32))
        return AudioParcial(n_total, ini, y)
    except Exception:                                                     # noqa: BLE001
        return cargar_estereo(ruta, sr)
