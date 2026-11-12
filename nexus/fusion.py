"""Renderizador de fusión: construye, por adelantado, el tramo donde dos temas se funden en uno solo.

Cada tema se separa en 4 capas (batería, bajo, voz/melodía, ambiente). Las capas se alinean a una rejilla común de
beats (el tempo se desliza de A a B), B se afina para armonizar con A y vuelve a su tono real, y cada capa entra o
sale con su propia curva: el bajo se intercambia en la línea de compás (nunca suenan dos graves a la vez), la batería
se relevan sobre el compás, y las partes armónicas se disuelven con un filtro que "abre" a B y "cierra" a A mientras un
procesador espectral baja, bin por bin, al tema que choca con el otro. Al final el audio de B queda IDÉNTICO al
original, así el empalme con la reproducción normal es inaudible.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.interpolate import PchipInterpolator

from . import capas as _capas
from .analisis import Analisis
from .capas import CAPAS
from .config import SR
from .planificador import Plan
from .postura import elegir as elegir_postura
from .warp import render_capa
from .wsola import WsolaFlujo

PARAMS_FFT = {"bateria": (1024, 256), "bajo": (4096, 1024), "lead": (2048, 512), "ambiente": (2048, 512)}
MARGEN = 16384
BANDAS_VIZ = 48
FPS_VIZ = 50


@dataclass
class Fusion:
    audio: np.ndarray                      # (2, n) float32: lo que suena durante la fusión (ya con ganancias y limitador)
    a_inicio: int                          # muestra de A donde empieza (A suena normal hasta acá)
    b_continua: int                        # muestra de B desde donde sigue la reproducción normal
    plan: Plan
    gan_a: float = 1.0
    gan_b: float = 1.0
    viz_bandas: np.ndarray = field(default_factory=lambda: np.zeros((0, BANDAS_VIZ, 2), np.float16))   # (cuadros, bandas, A|B)
    viz_gan: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 2), np.float16))               # ganancia de cada capa
    viz_nivel: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 2), np.float16))             # nivel real de cada capa
    t_swap: float = 0.5                    # fracción de la fusión donde se cambia el bajo
    carriles: tuple = ("Batería", "Bajo", "Voz/Melodía", "Ambiente")      # nombre de cada carril de la matriz
    siglas: tuple = ("BAT", "BAJ", "VOZ", "AMB")                           # nombre corto de cada medidor de deck
    motor: str = "capas"
    compas_seg: float = 2.0
    congelado: tuple = (0.0, 1.0)          # (segundos congelados en la línea blanca, k: velocidad del fader después)

    @property
    def n(self) -> int:
        return self.audio.shape[1]

    @property
    def duracion(self) -> float:
        return self.n / SR


# ------------------------------------------------------------------------------------------- utilidades
def limitador(x: np.ndarray, umbral: float = 0.92) -> np.ndarray:
    """Limitador suave: transparente por debajo del umbral y redondea los picos por encima."""
    a = np.abs(x)
    exceso = np.maximum(a - umbral, 0.0)
    sal = np.where(a > umbral, np.sign(x) * (umbral + (1 - umbral) * np.tanh(exceso / (1 - umbral))), x)
    return sal.astype(np.float32)


def limitar_mezcla(x: np.ndarray, techo: float = 0.90, extremos: float = 0.5) -> np.ndarray:
    """Limitador de la fusión que baja el VOLUMEN (no deforma la onda): cuando la suma de los dos temas pasa el techo, calcula cuánto
    hay que bajar y lo hace con una curva suave que se anticipa al pico (40 ms antes y después). Así los graves, medios y agudos no se
    saturan (un recorte sobre la forma de onda mete distorsión, sobre todo en los graves). Los `extremos` (segundos del principio y del
    final) no se tocan: ahí la mezcla es idéntica al tema solo y tiene que empalmar sin saltos. Al final queda el limitador suave de
    siempre como red de seguridad."""
    x = np.asarray(x, np.float32)
    n = x.shape[1]
    if n < 8:
        return limitador(x)
    pico = np.abs(x).max(axis=0)                                      # estéreo ligado: los dos canales bajan igual
    necesita = np.minimum(1.0, techo / np.maximum(pico, 1e-9)).astype(np.float32)
    if float(necesita.min()) >= 0.9999:
        return limitador(x)
    h = max(2, int(0.040 * SR))
    g = minimum_filter1d(necesita, size=2 * h + 1, mode="nearest")      # la bajada llega completa ANTES del pico y dura después
    g = uniform_filter1d(g, size=2 * h + 1, mode="nearest")             # curva suave (nunca por encima de lo necesario en el pico)
    ext = min(int(extremos * SR), n // 4)
    if ext > 0:
        rampa = np.minimum(1.0, np.minimum(np.arange(n), n - 1 - np.arange(n)) / float(ext)).astype(np.float32)
        g = 1.0 + (g - 1.0) * rampa
    return limitador(x * g[None, :].astype(np.float32))


def _suavizado(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _recorte(audio: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """audio[:, lo:hi] rellenando con ceros fuera de rango."""
    n = audio.shape[1]
    sal = np.zeros((audio.shape[0], hi - lo), np.float32)
    a, b = max(lo, 0), min(hi, n)
    if b > a:
        sal[:, a - lo:b - lo] = audio[:, a:b]
    return sal


def _beats_b_efectivos(B: Analisis, ib0: int, n_total: int, k: float) -> np.ndarray:
    """Beats de B tomados desde el beat ib0, a doble o mitad de pulso si k != 1."""
    if k == 1.0:
        return B.beats[ib0:ib0 + n_total]
    if k == 2.0:
        base = B.beats[ib0:ib0 + n_total // 2 + 3]
        mid = (base[:-1] + base[1:]) / 2
        return np.stack([base[:-1], mid], axis=1).reshape(-1)[:n_total]
    return B.beats[ib0:ib0 + 2 * n_total:2][:n_total]


def _db(x: np.ndarray) -> float:
    return 10 * np.log10(float(np.mean(x ** 2)) + 1e-12)


# ------------------------------------------------------------------------------------------- recetas
def disposicion(plan: Plan, A: Analisis, aj: dict | None = None) -> tuple[float, float, float, float]:
    """Reparto de la fusión en beats: (P, Q, rampa_in, rampa_out).

    La rampa de entrada va de P-rampa_in a P; la PAUSA ocupa de P a Q (los dos temas estables); la rampa de salida va de Q
    a Q+rampa_out. Q cae sobre una línea de compás: ahí se cambia el bajo. Rampa 0 = Auto (usa el tiempo que sobra).
    """
    aj = aj or {}
    N = 4.0 * plan.L
    bps = float(A.bpm) / 60.0
    H = min(max(0.0, float(aj.get("pausa", 5.0))) * bps, N - 8.0)
    ri = max(0.0, float(aj.get("rampa_in", 0.0))) * bps
    ro = max(0.0, float(aj.get("rampa_out", 0.0))) * bps
    libre = N - H
    if ri <= 0 and ro <= 0:
        ri = ro = libre / 2.0
    elif ri <= 0:
        ri = max(2.0, libre - ro)
    elif ro <= 0:
        ro = max(2.0, libre - ri)
    if ri + ro > libre:                                         # no entra todo: se achican las rampas en proporción
        k = libre / (ri + ro)
        ri, ro = ri * k, ro * k
    ri, ro = max(2.0, ri), max(2.0, ro)
    q_ideal = (N - (ri + H + ro)) / 2.0 + ri + H
    Q = 4.0 * round(q_ideal / 4.0)
    Q = min(max(Q, 4.0 * np.ceil((H + 2.0) / 4.0)), N - 4.0)
    P = max(0.0, Q - H)
    return float(P), float(Q), float(min(ri, max(P, 2.0))), float(min(ro, N - Q))


def ventana_voz(plan: Plan, A: Analisis, aj: dict | None) -> tuple[float, float]:
    """Beats (inicio, fin) del relevo de voz y melodía: SIEMPRE dentro de la pausa del cruce, centrado en ella."""
    P, Q, _ri, _R = disposicion(plan, A, aj)
    rv = float((aj or {}).get("voz_rampa", 2.0)) * float(A.bpm) / 60.0
    ancho = max(1.0, min(rv, 0.8 * (Q - P)))                     # nunca más largo que la pausa
    c = (P + Q) / 2.0
    if float((aj or {}).get("i_voces", 0.0)) >= 0.5:               # voces por frases: el relevo cae donde A hace una pausa de voz
        vA = np.asarray(A.voz, np.float64)[plan.a0:plan.a0 + plan.L]
        if len(vA) >= 2:
            pres = np.clip((vA - 0.35) / 0.30, 0.0, 1.0)
            cands = np.arange(P + ancho / 2.0, Q - ancho / 2.0 + 0.01, 0.5)
            if len(cands):
                def cant(cc):
                    t = np.clip(((np.linspace(cc - ancho / 2.0, cc + ancho / 2.0, 9)) / 4.0).astype(int), 0, len(pres) - 1)
                    return float(pres[t].mean()) + 0.01 * abs(cc - (P + Q) / 2.0)
                c = float(min(cands, key=cant))
    return c - ancho / 2.0, c + ancho / 2.0


def ventana_bajo(plan: Plan, A: Analisis, aj: dict | None) -> tuple[float, float]:
    """Beats (inicio, fin) del relevo del bajo: SIEMPRE dentro de la pausa, centrado en una línea de compás cerca de su mitad."""
    P, Q, _ri, _R = disposicion(plan, A, aj)
    seg = float((aj or {}).get("bajo_rampa", 1.0))
    ancho = max(0.25, min(seg * float(A.bpm) / 60.0, 0.9 * (Q - P)))
    c = 4.0 * round(((P + Q) / 2.0) / 4.0)
    c = min(max(c, P + ancho / 2.0), Q - ancho / 2.0)             # si no hay línea de compás útil, queda en la mitad exacta
    if not (P + ancho / 2.0 <= c <= Q - ancho / 2.0):
        c = (P + Q) / 2.0
    return c - ancho / 2.0, c + ancho / 2.0


def _congelar(pts: list[tuple[float, float]], h: float, fb: float, k: float, N: float) -> list[tuple[float, float]]:
    """Congela una curva (beat, valor) `fb` beats a partir del beat `h`: el valor se queda quieto y lo que sigue se desplaza `fb` beats
    (a la misma velocidad si k=1; k<1 solo comprime lo justo para que el relevo termine antes de que acabe la fusión)."""
    xs = [x for x, _ in pts]
    ys = [y for _, y in pts]
    v = float(np.interp(h, xs, ys))
    antes = [(x, y) for x, y in pts if x < h]
    despues = [(h + fb + (x - h) * k, y) for x, y in pts if x > h]
    sal = antes + [(h, v), (h + fb, v)] + despues
    if sal[-1][0] > N:                                                        # lo que se corre más allá del final se descarta
        xs2, ys2 = [x for x, _ in sal], [y for _, y in sal]
        sal = [(x, y) for x, y in sal if x < N - 1e-6] + [(float(N), float(np.interp(N, xs2, ys2)))]
    return sal


def _beats_congelado(plan: Plan, A: Analisis, aj: dict | None) -> tuple[float, float]:
    """(h, fb): beat donde cae la línea blanca punteada (la mitad del relevo de graves) y cuántos beats se congela ahí. fb=0: sin congelar."""
    seg = max(0.0, float((aj or {}).get("congelar", 0.0)))
    if seg <= 0.0:
        return 0.0, 0.0
    N = 4.0 * plan.L
    b0, b1 = ventana_bajo(plan, A, aj)
    h = (b0 + b1) / 2.0
    fb = min(seg * float(A.bpm) / 60.0, 0.6 * (N - h))      # siempre queda lugar para terminar el relevo
    return h, max(0.0, fb)


def parametros_congelado(plan: Plan, A: Analisis, aj: dict | None) -> tuple[float, float, float]:
    """(h, fb, k): beat de la línea blanca, beats congelados y factor de velocidad del resto (1 = misma velocidad; solo baja de 1
    si el relevo no entra desplazado en lo que queda de fusión)."""
    h, fb = _beats_congelado(plan, A, aj)
    if fb <= 0.0:
        return 0.0, 0.0, 1.0
    N = 4.0 * plan.L
    P, Q, ri, R = disposicion(plan, A, aj)
    fin = max(Q + R, ventana_bajo(plan, A, aj)[1], ventana_voz(plan, A, aj)[1])          # donde termina el último movimiento de los faders
    k = 1.0 if fin <= h else min(1.0, max(0.3, (N - 0.5 - h - fb) / (fin - h)))
    return h, fb, k


def _meta_congelado(plan: Plan, A: Analisis, aj: dict | None) -> tuple[float, float]:
    """(segundos congelados, k) que se guardan en la fusión para que la interfaz mueva el cursor igual que el fader."""
    h, fb, k = parametros_congelado(plan, A, aj)
    return (max(0.0, float((aj or {}).get("congelar", 0.0))) if fb > 0 else 0.0, float(k))


def congelar_receta(r: dict, plan: Plan, A: Analisis, aj: dict | None) -> None:
    """Los faders (todas las curvas) quedan quietos unos segundos cuando el cursor toca la línea blanca punteada."""
    h, fb, k = parametros_congelado(plan, A, aj)
    if fb <= 0.0:
        return
    for clave in list(r):
        r[clave] = _congelar(r[clave], h, fb, k, 4.0 * plan.L)


def _curva_relevo(b0: float, b1: float, sale: bool) -> list[tuple[float, float]]:
    """Puntos (beat, ganancia) de un relevo suave (potencia constante) entre b0 y b1."""
    fr = (0.0, 0.25, 0.5, 0.75, 1.0)
    v = [float(np.cos(f * np.pi / 2)) if sale else float(np.sin(f * np.pi / 2)) for f in fr]
    return [(b0 + f * (b1 - b0), g) for f, g in zip(fr, v)]


def _separar_voces(r: dict, plan: Plan, A: Analisis, B: Analisis, aj: dict | None, N: int, Q: float,
                   banda: str = "lead", fuerza: float = 1.0) -> None:
    """Hace que las voces de A y B no suenen juntas: manda la de A hasta el relevo de voz (dentro de la pausa) y después la de B,
    con un pequeño hueco entre las dos."""
    L = plan.L
    vA = np.asarray(A.voz[plan.a0:plan.a0 + L], np.float64)
    vB = np.asarray(B.voz[plan.b0:plan.b0 + plan.Lb], np.float64)
    if len(vA) < 2 or len(vB) < 1:
        return
    pos_b = np.clip(np.round(np.arange(len(vA)) / plan.k).astype(int), 0, len(vB) - 1)
    presencia = lambda v: np.clip((v - 0.35) / 0.30, 0.0, 1.0)                # 0 = sin voz, 1 = voz clara
    t = np.arange(0.0, N + 0.01, 0.5)
    centros = np.arange(len(vA)) * 4.0 + 2.0
    pa = np.interp(t, centros, presencia(vA))
    pb = np.interp(t, centros, presencia(vB[pos_b]))
    gA = np.clip(_pchip_mono([p[0] for p in r[("A", banda)]], [p[1] for p in r[("A", banda)]])(t), 0, 1)
    gB = np.clip(_pchip_mono([p[0] for p in r[("B", banda)]], [p[1] for p in r[("B", banda)]])(t), 0, 1)
    v0, v1 = ventana_voz(plan, A, aj)
    w = _suavizado((t - v0) / (v1 - v0))                                         # 0 = manda A, 1 = manda B
    nuevo_b = gB * (1.0 - fuerza * np.clip(pa * gA * 2.0, 0, 1) * (1.0 - w))
    nuevo_a = gA * (1.0 - fuerza * np.clip(pb * gB * 2.0, 0, 1) * w)
    r[("A", banda)] = [(float(x), float(y)) for x, y in zip(t, nuevo_a)]
    r[("B", banda)] = [(float(x), float(y)) for x, y in zip(t, nuevo_b)]


def receta(plan: Plan, A: Analisis, B: Analisis, aj: dict | None = None) -> dict[tuple[str, str], list[tuple[float, float]]]:
    """Puntos (beat, ganancia) de cada capa de cada tema. Las curvas se suavizan con PCHIP.

    Forma: entrada gradual hasta la PAUSA (los dos temas sonando juntos y estables ~5 s), después el relevo:
    el bajo cambia sobre la línea de compás (en ~0.6 beat) y la batería de B termina de entrar durante un compás.
    """
    N = 4 * plan.L
    L = plan.L
    P, Q, ri, R = disposicion(plan, A, aj)
    profunda = plan.modo == "profunda"
    r: dict[tuple[str, str], list[tuple[float, float]]] = {}
    S = max(0.0, P - ri)                                  # donde empieza la rampa de entrada
    dri, dro = min(ri, 8.0), min(R, 8.0)                  # la batería no necesita más de 2 compases para relevarse
    r[("A", "bateria")] = [(0, 1), (P - dri, 1), (P, 0.7), (Q, 0.7), (Q + dro, 0), (N, 0)]
    r[("B", "bateria")] = [(0, 0), (P - dri, 0), (P, 0.55), (Q, 0.55), (Q + dro, 1), (N, 1)]
    bb0, bb1 = ventana_bajo(plan, A, aj)                   # el relevo del bajo cae en la MITAD de la pausa
    r[("A", "bajo")] = [(0, 1)] + _curva_relevo(bb0, bb1, True) + [(N, 0)]
    r[("B", "bajo")] = [(0, 0)] + _curva_relevo(bb0, bb1, False) + [(N, 1)]
    # voces: si ambos temas cantan en la zona, el que entra espera a que el que sale haya terminado
    vA = A.voz[plan.a0:plan.a0 + L]
    vB = B.voz[plan.b0:plan.b0 + plan.Lb]
    pos_b = np.clip(np.round(np.arange(L) / plan.k).astype(int), 0, len(vB) - 1)
    vBz = vB[pos_b] if len(vB) else np.zeros(L)
    espera = bool(len(vA) and vA[len(vA) // 2:].max() > 0.6 and vBz.max() > 0.6)
    v0, v1 = ventana_voz(plan, A, aj)                      # el relevo de voz/melodía cae DENTRO de la pausa
    lb = 0.0 if espera else 0.3
    if profunda:
        r[("A", "lead")] = [(0, 1), (S, 1), (P, 0.8), (v0, 0.8), (v1, 0), (N, 0)]
        r[("B", "lead")] = [(0, 0), (S + 0.5 * ri, 0), (P, lb), (v0, lb), (v1, 1), (N, 1)]
        r[("A", "ambiente")] = [(0, 1), (S, 1), (P, 0.7), (Q, 0.7), (Q + 0.65 * R, 0), (N, 0)]
        r[("B", "ambiente")] = [(0, 0), (S, 0), (P, 0.7), (Q, 0.7), (Q + 0.35 * R, 1), (N, 1)]
    else:                                              # armonías que no combinan: casi sin solape de melodías
        r[("A", "lead")] = [(0, 1), (S, 1), (P, 0.6), (v0, 0.6), (v1, 0), (N, 0)]
        r[("B", "lead")] = [(0, 0), (P, 0), (v0, 0), (v1, 1), (N, 1)]
        r[("A", "ambiente")] = [(0, 1), (S, 1), (P, 0.6), (Q, 0.6), (Q + 0.50 * R, 0), (N, 0)]
        r[("B", "ambiente")] = [(0, 0), (S, 0), (P, 0.5), (Q, 0.5), (Q + 0.40 * R, 1), (N, 1)]
    _separar_voces(r, plan, A, B, aj, N, Q)
    congelar_receta(r, plan, A, aj)                                      # al final: nada vuelve a mover los faders durante el congelado
    return r


def _pchip_mono(xs, ys) -> PchipInterpolator:
    """PCHIP con abscisas estrictamente crecientes (los puntos repetidos se separan una pizca)."""
    xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
    o = np.argsort(xs, kind="stable")                                    # los puntos se ordenan por tiempo
    x = np.maximum.accumulate(xs[o] + np.arange(len(xs)) * 1e-6)
    return PchipInterpolator(x, ys[o], extrapolate=True)


def _curva(puntos: list[tuple[float, float]], beat_a_n: PchipInterpolator, n: int) -> np.ndarray:
    xs = np.array([p[0] for p in puntos], np.float64)
    ys = np.array([p[1] for p in puntos], np.float64)
    o = np.argsort(xs, kind="stable")
    xs, ys = xs[o], ys[o]
    xs = np.maximum.accumulate(xs + np.arange(len(xs)) * 1e-6)           # estrictamente creciente
    ns = beat_a_n(np.clip(xs, 0, None))
    f = PchipInterpolator(ns, ys, extrapolate=False)
    g = f(np.arange(n))
    g = np.where(np.isnan(g), ys[0] if np.arange(n)[0] < ns[0] else ys[-1], g)
    g[np.arange(n) >= ns[-1]] = ys[-1]
    g[np.arange(n) <= ns[0]] = ys[0]
    return np.clip(g, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------------------------------- etapa espectral
def _etapa_espectral(GA: np.ndarray, GB: np.ndarray, u: np.ndarray, profunda: bool) -> tuple[np.ndarray, np.ndarray]:
    """Procesador de fusión: baja, bin por bin, al tema que choca con el otro y "abre/cierra" cada tema con un filtro."""
    import librosa
    n = GA.shape[1]
    n_fft, hop = 2048, 512
    XA = np.stack([librosa.stft(GA[c], n_fft=n_fft, hop_length=hop) for c in range(2)])
    XB = np.stack([librosa.stft(GB[c], n_fft=n_fft, hop_length=hop) for c in range(2)])
    F = XA.shape[2]
    fr = librosa.fft_frequencies(sr=SR, n_fft=n_fft)[:, None]
    t_c = np.minimum(np.arange(F) * hop, n - 1)
    uc = u[t_c][None, :]                                           # progreso del relevo (0 = todo A, 1 = todo B)
    eA = np.abs(XA).mean(axis=0) ** 2
    eB = np.abs(XB).mean(axis=0) ** 2
    from scipy.ndimage import uniform_filter
    eA, eB = uniform_filter(eA, (5, 3)), uniform_filter(eB, (5, 3))
    choque = np.minimum(eA, eB) / (np.maximum(eA, eB) + 1e-12)       # 1 = los dos suenan igual de fuerte en ese punto
    k = 0.55 if profunda else 0.35
    # prioridad: el que domina en ese momento se queda; el otro baja en los puntos donde chocan
    dom_b = _suavizado((uc - 0.15) / 0.7)
    mA = 1.0 - k * choque * dom_b
    mB = 1.0 - k * choque * (1.0 - dom_b)
    # filtros de disolución: A se va cerrando hacia graves-medios; B se abre desde medios-graves
    fcA = 18000.0 * (1.0 - _suavizado((uc - 0.30) / 0.65)) + 1400.0 * _suavizado((uc - 0.30) / 0.65)
    fcB = 900.0 * (1.0 - _suavizado((uc - 0.10) / 0.55)) + 20000.0 * _suavizado((uc - 0.10) / 0.55)
    mA = mA / (1.0 + (fr / fcA) ** 4)
    mB = mB / (1.0 + (fr / fcB) ** 4)
    mA = uniform_filter(mA, (1, 3))
    mB = uniform_filter(mB, (1, 3))
    # sin procesado en los extremos: al inicio A pasa tal cual, al final B pasa tal cual (empalme exacto)
    mA = 1.0 - _suavizado(uc / 0.10) * (1.0 - mA)
    mB = 1.0 - (1.0 - _suavizado((uc - 0.80) / 0.18)) * (1.0 - mB)
    ra = np.stack([librosa.istft(XA[c] * mA, hop_length=hop, n_fft=n_fft, length=n) for c in range(2)]).astype(np.float32)
    rb = np.stack([librosa.istft(XB[c] * mB, hop_length=hop, n_fft=n_fft, length=n) for c in range(2)]).astype(np.float32)
    return ra, rb


def _eco(x: np.ndarray, envio: np.ndarray, retardo: int, fb: float = 0.5, repet: int = 6) -> np.ndarray:
    """Eco ping-pong sincronizado (retardo en muestras) con los agudos cayendo en cada repetición."""
    f = x * envio[None, :]
    sos = signal.butter(1, 3500.0, "lowpass", fs=SR, output="sos")
    sal = np.zeros_like(x)
    cur = f
    for i in range(1, repet + 1):
        cur = signal.sosfilt(sos, cur, axis=1) * fb
        d = i * retardo
        if d >= x.shape[1]:
            break
        mov = np.roll(cur, d, axis=1)
        mov[:, :d] = 0.0
        if i % 2 == 1:
            mov = mov[::-1]                                       # ping-pong entre canales
        sal += mov
    return sal


# ------------------------------------------------------------------------------------------- renderizado principal
def desviacion_tempo(A: Analisis, B: Analisis, plan: Plan) -> float:
    """Cuánto se aparta (como fracción) la velocidad de lectura de cada tema de 1.0 durante la fusión: decide si alcanza el motor
    natural (rápido) o hace falta el de capas (lento). Misma cuenta que `renderizar`, sin tocar audio."""
    N = 4 * plan.L
    ia0, ib0 = int(A.compases[plan.a0]), int(B.compases[plan.b0])
    bA = A.beats[ia0:ia0 + N + 13]
    bB = _beats_b_efectivos(B, ib0, N + 13, plan.k)
    n_b = min(len(bA), len(bB))
    if n_b < N + 2:
        return 1.0
    bA, bB = bA[:n_b], bB[:n_b]
    perA, perB = np.diff(bA), np.diff(bB)
    b_idx = np.arange(n_b - 1, dtype=np.float64)
    w = _suavizado((b_idx - 0.18 * N) / (0.50 * N))
    per = np.exp((1 - w) * np.log(perA) + w * np.log(perB))
    t_out = np.concatenate([[0.0], np.cumsum(per)])
    n_axis = np.arange(int(round(t_out[N] * SR)), dtype=np.float64)
    src_a = PchipInterpolator(t_out, bA * SR)(n_axis / SR)
    src_b = PchipInterpolator(t_out, bB * SR)(n_axis / SR)
    return max(float(np.abs(np.gradient(src_a) - 1.0).max()), float(np.abs(np.gradient(src_b) - 1.0).max()))


def renderizar(A: Analisis, audio_a: np.ndarray, B: Analisis, audio_b: np.ndarray, plan: Plan,
               progreso=None, ajustes: dict | None = None) -> Fusion:
    """Construye la fusión A→B según `plan`. `audio_a/b`: (2, n) float32 a SR. `progreso(frac, texto)` opcional."""
    def avance(f: float, t: str = "") -> None:
        if progreso:
            progreso(f, t)

    ajustes, postura = elegir_postura(plan, A, B, ajustes)           # postura automática: NEXUS decide cómo mezclar ESTE par
    if postura:
        plan = replace(plan, razon=f"{plan.razon} · postura {postura}")
    if float((ajustes or {}).get("i_puentes", 0.0)) >= 0.5 and necesita_puente(plan, A, B):
        try:
            fp = _renderizar_puente(A, audio_a, B, audio_b, plan, ajustes, avance)
            plan.detalle.pop("gusto", None)                           # en un puente no se usó ninguna postura: un voto no debe enseñarle nada a NEXUS
            return fp
        except (ValueError, FloatingPointError, IndexError):
            pass                                                      # si no se pudo, sigue la fusión de siempre
    N = 4 * plan.L
    pad = 12
    ia0 = int(A.compases[plan.a0])
    ib0 = int(B.compases[plan.b0])
    bA = A.beats[ia0:ia0 + N + 1 + pad]
    bB = _beats_b_efectivos(B, ib0, N + 1 + pad, plan.k)
    n_b = min(len(bA), len(bB))
    if n_b < N + 2:
        raise ValueError("No hay beats suficientes para fusionar en ese punto.")
    bA, bB = bA[:n_b], bB[:n_b]
    if (ajustes or {}).get("motor", "natural") != "capas":           # motor en vivo (receta de AutoMix); si no aplica, sigue el flujo de siempre
        try:
            fv = _renderizar_vivo(A, audio_a, B, audio_b, plan, ajustes, bA, bB, N, avance)
        except (ValueError, FloatingPointError):
            fv = None
        if fv is not None:
            return fv
    perA = np.diff(bA)
    perB = np.diff(bB)
    b_idx = np.arange(n_b - 1, dtype=np.float64)
    w = _suavizado((b_idx - 0.18 * N) / (0.50 * N))                  # 0 = tempo de A, 1 = tempo de B
    per = np.exp((1 - w) * np.log(perA) + w * np.log(perB))
    t_out = np.concatenate([[0.0], np.cumsum(per)])                   # tiempo de salida de cada beat
    Nout = int(round(t_out[N] * SR))
    n_axis = np.arange(Nout, dtype=np.float64)
    src_a = PchipInterpolator(t_out, bA * SR)(n_axis / SR)
    src_b = PchipInterpolator(t_out, bB * SR)(n_axis / SR)
    src_a += round(src_a[0]) - src_a[0]                               # el inicio de A y el final de B caen en muestras enteras
    c = src_b[-1] - (Nout - 1)
    src_b += round(c) - c
    a_inicio = int(round(src_a[0]))
    b_continua = int(round(src_b[-1])) + 1
    beat_a_n = PchipInterpolator(np.arange(len(t_out), dtype=np.float64), t_out * SR)   # beat → muestra de salida
    # progreso del relevo (0 = todo A, 1 = todo B): sube hasta la pausa, se queda casi quieto y termina de cruzar
    P, Q, ri_, R_ = disposicion(plan, A, ajustes)
    idx_b = np.minimum(np.arange(n_b, dtype=np.float64), N)
    nodos = list(zip([0, max(0.0, P - ri_), P, Q, Q + 0.6 * R_, Q + R_, N], [0.0, 0.0, 0.38, 0.55, 0.92, 1.0, 1.0]))
    h_c, fb_c, k_c = parametros_congelado(plan, A, ajustes)
    if fb_c > 0:
        nodos = _congelar(nodos, h_c, fb_c, k_c, N)
    u_b = _pchip_mono([x for x, _ in nodos], [y for _, y in nodos])(idx_b)
    u = np.clip(PchipInterpolator(t_out * SR, u_b)(n_axis), 0.0, 1.0).astype(np.float32)
    if (ajustes or {}).get("motor", "natural") == "natural":
        desv = max(float(np.abs(np.gradient(src_a) - 1.0).max()), float(np.abs(np.gradient(src_b) - 1.0).max()))
        if desv <= float((ajustes or {}).get("umbral_natural", 0.07)):   # tempos parecidos: se puede mezclar el audio original sin deformarlo
            return _renderizar_natural(A, audio_a, B, audio_b, plan, ajustes, src_a, src_b, a_inicio, b_continua, beat_a_n,
                                       u, per, Nout, N, avance)
    avance(0.05, "Separando capas del tema que sale…")

    # --- separación en capas (con margen para los filtros)
    lo_a, hi_a = int(src_a.min()) - MARGEN, int(src_a.max()) + MARGEN
    lo_b, hi_b = int(src_b.min()) - MARGEN, int(src_b.max()) + MARGEN
    capas_a = _capas.separar(_recorte(audio_a, lo_a, hi_a), SR)
    avance(0.25, "Separando capas del tema que entra…")
    capas_b = _capas.separar(_recorte(audio_b, lo_b, hi_b), SR)

    # --- tono: B arranca armonizado con A y vuelve a su tono real antes de terminar
    if abs(plan.shift) >= 0.05:
        # B se queda armonizado con A durante la pausa y vuelve a su tono real después del cambio de bajo
        semi_b = plan.shift * (1.0 - _suavizado((idx_b - (Q - 2)) / max(1.0, (Q + R_) - (Q - 2))))
        semi = PchipInterpolator(t_out * SR, semi_b)(n_axis)
        pitch_b = 2.0 ** (semi / 12.0)
    else:
        pitch_b = None

    # --- render de cada capa con el mapa temporal común
    ren_a, ren_b = {}, {}
    for i, nombre in enumerate(CAPAS):
        n_fft, hop = PARAMS_FFT[nombre]
        avance(0.40 + 0.10 * i, f"Alineando {_capas.NOMBRES_CAPAS[nombre]}…")
        ren_a[nombre] = render_capa(capas_a[nombre], lo_a, src_a, None, n_fft, hop, ancla="inicio")
        ren_b[nombre] = render_capa(capas_b[nombre], lo_b, src_b, None if nombre == "bateria" else pitch_b, n_fft, hop, ancla="fin")
    del capas_a, capas_b

    # --- automatización de las capas
    avance(0.82, "Fusionando…")
    rec = receta(plan, A, B, ajustes)
    gan = {key: _curva(pts, beat_a_n, Nout) for key, pts in rec.items()}
    lin_a, lin_b = 10 ** (A.ganancia_db / 20), 10 ** (B.ganancia_db / 20)
    dr = ren_a["bateria"] * gan[("A", "bateria")] * lin_a + ren_b["bateria"] * gan[("B", "bateria")] * lin_b
    GA = sum(ren_a[n] * gan[("A", n)] for n in ("bajo", "lead", "ambiente")) * lin_a
    GB = sum(ren_b[n] * gan[("B", n)] for n in ("bajo", "lead", "ambiente")) * lin_b
    ra, rb = _etapa_espectral(GA, GB, u, plan.modo == "profunda")

    # --- eco puente sincronizado (cola de A que se desvanece mientras entra B)
    ret = int(round(float(per[int(0.55 * N)]) * SR * 0.75))
    campana = np.exp(-0.5 * ((np.arange(Nout) / Nout - min(0.85, (Q + 2) / N)) / 0.14) ** 2) * (0.30 if plan.modo == "profunda" else 0.22)
    fuente = (ren_a["lead"] * gan[("A", "lead")] + ren_a["ambiente"] * gan[("A", "ambiente")]) * lin_a
    eco = _eco(fuente, campana.astype(np.float32), ret)
    eco[:, :int(0.04 * Nout)] = 0.0
    eco[:, int(0.97 * Nout):] = 0.0
    mezcla = dr + ra + rb + eco

    # --- nivel parejo: sin baches ni picos a mitad de la fusión (los extremos quedan intactos)
    ventana = int(1.0 * SR)
    pot = signal.fftconvolve((mezcla ** 2).mean(axis=0), np.ones(ventana) / ventana, mode="same")
    db_m = 10 * np.log10(pot + 1e-10)
    borde = int(1.5 * per[0] * SR * 2)
    ref_a, ref_b = db_m[min(borde, Nout - 1)], db_m[max(0, Nout - 1 - borde)]
    objetivo = ref_a + (ref_b - ref_a) * u
    corr_db = np.clip(objetivo - db_m, -4.0, 3.0)
    corr_db = signal.fftconvolve(corr_db, np.ones(int(0.8 * SR)) / int(0.8 * SR), mode="same")
    taper = np.minimum(1.0, np.minimum(np.arange(Nout), Nout - 1 - np.arange(Nout)) / max(1, borde)).astype(np.float32)
    mezcla = mezcla * (10 ** (corr_db * taper / 20)).astype(np.float32)[None, :]
    mezcla = limitar_mezcla(mezcla)

    # --- datos para la visualización (a 50 cuadros por segundo)
    avance(0.93, "Preparando visualización…")
    viz_b, viz_g, viz_n = _viz(ren_a, ren_b, gan, lin_a, lin_b, ra, rb, dr, Nout)
    avance(1.0, "Listo")
    return Fusion(audio=mezcla, a_inicio=a_inicio, b_continua=b_continua, plan=plan, gan_a=lin_a, gan_b=lin_b,
                  viz_bandas=viz_b, viz_gan=viz_g, viz_nivel=viz_n, t_swap=float(sum(ventana_bajo(plan, A, ajustes)) / 2.0 / N),
                  compas_seg=float(4 * per.mean()), congelado=_meta_congelado(plan, A, ajustes))


# ------------------------------------------------------------------------------------------- motor natural
CORTES_BANDAS = (150.0, 900.0, 4000.0)                  # Hz: graves | cuerpo | presencia | brillo
CARRILES_NATURAL = ("Graves", "Cuerpo", "Presencia", "Brillo")
SIGLAS_NATURAL = ("GRA", "CUE", "PRE", "BRI")


def _leer_fraccionado(audio: np.ndarray, src: np.ndarray, lo: int) -> np.ndarray:
    """Lee `audio` en posiciones fraccionarias `src` con interpolación cúbica (Catmull-Rom). Si src es entero, es una copia exacta."""
    n = audio.shape[1]
    i0 = np.floor(src).astype(np.int64)
    fr = (src - i0).astype(np.float32)
    out = np.empty((audio.shape[0], len(src)), np.float32)
    idx = [np.clip(i0 + d - lo, 0, n - 1) for d in (-1, 0, 1, 2)]
    c0 = (-0.5 * fr ** 3 + fr ** 2 - 0.5 * fr)
    c1 = (1.5 * fr ** 3 - 2.5 * fr ** 2 + 1.0)
    c2 = (-1.5 * fr ** 3 + 2.0 * fr ** 2 + 0.5 * fr)
    c3 = (0.5 * fr ** 3 - 0.5 * fr ** 2)
    for c in range(audio.shape[0]):
        a = audio[c]
        out[c] = c0 * a[idx[0]] + c1 * a[idx[1]] + c2 * a[idx[2]] + c3 * a[idx[3]]
    return out


def _bandas(x: np.ndarray) -> list[np.ndarray]:
    """Parte el audio en 4 bandas que SUMAN exactamente el original (filtros de fase cero)."""
    pasa = []
    for fc in CORTES_BANDAS:
        sos = signal.butter(2, fc / (SR / 2), btype="low", output="sos")
        pasa.append(signal.sosfiltfilt(sos, x, axis=1).astype(np.float32))
    return [pasa[0], pasa[1] - pasa[0], pasa[2] - pasa[1], x - pasa[2]]


def receta_natural(plan: Plan, A: Analisis, B: Analisis, aj: dict | None) -> dict:
    """Curvas (beat, ganancia) de cada banda de cada tema para la fusión natural."""
    N = 4 * plan.L
    P, Q, ri, R = disposicion(plan, A, aj)
    S = max(0.0, P - ri)
    limpia = plan.modo == "limpia" or abs(plan.shift) >= 0.5                   # sin armonía clara: menos solape en medios
    k = 0.75 if limpia else 1.0
    r: dict = {}
    bb0, bb1 = ventana_bajo(plan, A, aj)                   # el relevo de graves cae en la MITAD de la pausa
    r[("A", "graves")] = [(0, 1)] + _curva_relevo(bb0, bb1, True) + [(N, 0)]
    r[("B", "graves")] = [(0, 0)] + _curva_relevo(bb0, bb1, False) + [(N, 1)]
    v0, v1 = ventana_voz(plan, A, aj)                      # el relevo de voz y melodía cae DENTRO de la pausa
    r[("A", "cuerpo")] = [(0, 1), (S, 1), (P, 0.85 * k), (v0, 0.85 * k), (v1, 0), (N, 0)]
    r[("B", "cuerpo")] = [(0, 0), (S, 0), (P, 0.60 * k), (v0, 0.60 * k), (v1, 1), (N, 1)]
    r[("A", "presencia")] = [(0, 1), (S, 1), (P, 0.80 * k), (v0, 0.80 * k), (v1, 0), (N, 0)]
    r[("B", "presencia")] = [(0, 0), (S + 0.4 * ri, 0), (P, 0.50 * k), (v0, 0.50 * k), (v1, 1), (N, 1)]
    r[("A", "brillo")] = [(0, 1), (S, 1), (P, 0.70), (Q, 0.70), (Q + 0.50 * R, 0), (N, 0)]
    r[("B", "brillo")] = [(0, 0), (S, 0), (P, 0.70), (Q, 0.70), (Q + 0.30 * R, 1), (N, 1)]
    _separar_voces(r, plan, A, B, aj, N, Q, banda="presencia", fuerza=0.9)        # las voces de A y B no suenan juntas
    _separar_voces(r, plan, A, B, aj, N, Q, banda="cuerpo", fuerza=0.55)
    congelar_receta(r, plan, A, aj)
    return r


def _renderizar_natural(A, audio_a, B, audio_b, plan, ajustes, src_a, src_b, a_inicio, b_continua, beat_a_n, u, per, Nout, N, avance,
                        xa=None, xb=None, rec=None, motor: str = "natural") -> Fusion:
    """Fusión con el audio ORIGINAL de los dos temas (sin separar ni deformar): rejilla común de beats, 4 bandas con su propia
    curva (el cambio de graves cae en la línea de compás), voces que no se pisan y nivel parejo."""
    if xa is None:                                       # (el motor en vivo trae sus propios tramos ya alineados)
        avance(0.10, "Alineando los temas a un mismo pulso…")
        lo_a, hi_a = int(src_a.min()) - 8, int(src_a.max()) + 8
        lo_b, hi_b = int(src_b.min()) - 8, int(src_b.max()) + 8
        xa = _leer_fraccionado(_recorte(audio_a, lo_a, hi_a), src_a, lo_a)
        xb = _leer_fraccionado(_recorte(audio_b, lo_b, hi_b), src_b, lo_b)
    avance(0.40, "Separando en bandas…")
    ba, bb = _bandas(xa), _bandas(xb)
    del xa, xb
    avance(0.70, "Fusionando…")
    P, Q, ri, R = disposicion(plan, A, ajustes)
    rec = rec if rec is not None else receta_natural(plan, A, B, ajustes)
    nombres = ("graves", "cuerpo", "presencia", "brillo")
    gan = {key: _curva(pts, beat_a_n, Nout) for key, pts in rec.items()}
    lin_a, lin_b = 10 ** (A.ganancia_db / 20), 10 ** (B.ganancia_db / 20)
    ca = sum(ba[i] * gan[("A", n)] for i, n in enumerate(nombres)) * lin_a
    cb = sum(bb[i] * gan[("B", n)] for i, n in enumerate(nombres)) * lin_b
    mezcla = ca + cb

    # nivel parejo (los extremos quedan intactos para empalmar bit a bit)
    ventana = int(1.0 * SR)
    pot = signal.fftconvolve((mezcla ** 2).mean(axis=0), np.ones(ventana) / ventana, mode="same")
    db_m = 10 * np.log10(pot + 1e-10)
    borde = int(1.5 * per[0] * SR * 2)
    ref_a, ref_b = db_m[min(borde, Nout - 1)], db_m[max(0, Nout - 1 - borde)]
    objetivo = ref_a + (ref_b - ref_a) * u
    corr_db = np.clip(objetivo - db_m, -4.0, 3.0)
    corr_db = signal.fftconvolve(corr_db, np.ones(int(0.8 * SR)) / int(0.8 * SR), mode="same")
    taper = np.minimum(1.0, np.minimum(np.arange(Nout), Nout - 1 - np.arange(Nout)) / max(1, borde)).astype(np.float32)
    mezcla = limitar_mezcla(mezcla * (10 ** (corr_db * taper / 20)).astype(np.float32)[None, :])

    avance(0.93, "Preparando visualización…")
    viz_b, viz_g, viz_n = _viz_natural(ba, bb, gan, nombres, lin_a, lin_b, ca, cb, Nout)
    avance(1.0, "Listo")
    return Fusion(audio=mezcla, a_inicio=a_inicio, b_continua=b_continua, plan=plan, gan_a=lin_a, gan_b=lin_b,
                  viz_bandas=viz_b, viz_gan=viz_g, viz_nivel=viz_n, t_swap=float(sum(ventana_bajo(plan, A, ajustes)) / 2.0 / N), compas_seg=float(4 * per.mean()), congelado=_meta_congelado(plan, A, ajustes),
                  carriles=CARRILES_NATURAL, siglas=SIGLAS_NATURAL, motor=motor)



# ------------------------------------------------------------------------------------------- puente (temas que no combinan)
def _barrido(x: np.ndarray, cortes: list, tipo: str, idx: np.ndarray) -> np.ndarray:
    """Filtro que se abre o se cierra con el tiempo: mezcla versiones filtradas a distintos cortes según `idx` (0 = primera, len-1 = última).
    El corte `None` es el audio sin filtrar."""
    vers = []
    for fc in cortes:
        if fc is None:
            vers.append(x)
        else:
            sos = signal.butter(2, fc / (SR / 2), btype=tipo, output="sos")
            vers.append(signal.sosfilt(sos, x, axis=1).astype(np.float32))
    sal = np.zeros_like(x)
    for k, v in enumerate(vers):
        w = np.clip(1.0 - np.abs(idx - k), 0.0, 1.0).astype(np.float32)
        sal += v * w[None, :]
    return sal


def necesita_puente(plan: Plan, A: Analisis, B: Analisis) -> bool:
    """Los temas no combinan para fusionarlos pegados: tempos muy distintos, o armonía y voces que chocan a la vez."""
    from .postura import rasgos
    r = rasgos(plan, A, B)
    return r["dif_tempo"] > 0.12 or (r["armonia"] < 0.30 and r["choque_voces"] > 0.30)


def _renderizar_puente(A: Analisis, audio_a, B: Analisis, audio_b, plan: Plan, ajustes, avance) -> Fusion:
    """En vez de forzar una fusión: A sigue a su tempo mientras un filtro le va cerrando los agudos y le crece un eco; en una línea de
    compás A se corta y el eco se apaga solo, y B entra a su propio tempo (su «uno» cae en esa línea) con un filtro que se abre."""
    N = 4 * plan.L
    ia0 = int(A.compases[plan.a0])
    ib0 = int(B.compases[plan.b0])
    bA = A.beats[ia0:ia0 + N + 1]
    if len(bA) < N + 1:
        raise ValueError("No hay beats suficientes para el puente.")
    avance(0.1, "Armando el puente…")
    a_ini = int(round(bA[0] * SR))
    Nout = int(round((bA[N] - bA[0]) * SR))
    t_in_b = int(np.clip(4 * round(0.62 * N / 4), 8, N - 8))
    ti = int(round((bA[t_in_b] - bA[0]) * SR))
    rem = Nout - ti
    sB = int(round(B.beats[ib0] * SR))
    xa = _recorte(audio_a, a_ini, a_ini + Nout)
    xb_seg = _recorte(audio_b, sB, sB + rem)
    n = np.arange(Nout, dtype=np.float64)
    # --- A: se le cierra el filtro (8 s antes del corte, en ~4 compases) y crece el eco; se corta en la línea de compás
    ini_f = max(0, ti - int(4 * 4 * 60.0 / A.bpm * SR))
    sA = _suavizado((n - ini_f) / max(1.0, ti - ini_f))                       # 0 = abierto, 1 = cerrado
    idx_a = sA * 4.0
    xa_f = _barrido(xa, [None, 6000.0, 2500.0, 1000.0, 400.0], "low", idx_a)
    fa = max(8, int(0.03 * SR))
    g_a = np.where(n < ti - fa, 1.0, np.clip((ti - n) / fa, 0.0, 1.0)).astype(np.float32)
    d = int(round(0.75 * 60.0 / A.bpm * SR))                                  # eco de tres cuartos de pulso: cae a contratiempo
    entra_eco = _suavizado((n - (ti - 0.5 * (ti - ini_f))) / max(1.0, 0.5 * (ti - ini_f))).astype(np.float32)
    fuente = xa_f * g_a[None, :] * entra_eco[None, :]
    eco = np.zeros_like(xa)
    for k in range(1, 7):
        o = k * d
        if o >= Nout:
            break
        eco[:, o:] += (0.55 ** k) * fuente[:, :Nout - o]
    ca = (xa_f * g_a[None, :] + 0.9 * eco) * (10 ** (A.ganancia_db / 20))
    # --- B: entra en la línea de compás con un filtro que se abre durante dos compases
    xb = np.zeros_like(xa)
    xb[:, ti:] = xb_seg[:, :rem]
    dur_ap = int(2 * 4 * 60.0 / B.bpm * SR)
    sB_ = 1.0 - _suavizado((n - ti) / max(1.0, dur_ap))                       # 1 = cerrado, 0 = abierto
    sB_ = np.where(n < ti, 1.0, sB_)
    xb_f = _barrido(xb, [None, 250.0, 700.0, 1500.0], "high", (sB_ * 3.0))
    fb = max(8, int(0.008 * SR))
    g_b = np.clip((n - ti) / fb, 0.0, 1.0).astype(np.float32)
    cb = xb_f * g_b[None, :] * (10 ** (B.ganancia_db / 20))
    mezcla = limitar_mezcla(ca + cb)
    # --- visualización
    avance(0.8, "Preparando visualización…")
    lin_a, lin_b = 10 ** (A.ganancia_db / 20), 10 ** (B.ganancia_db / 20)
    ba, bb = _bandas(xa), _bandas(xb)
    sa, sb = sA.astype(np.float32), sB_.astype(np.float32)
    gan = {("A", "graves"): g_a, ("A", "cuerpo"): g_a * (1 - 0.3 * sa), ("A", "presencia"): g_a * (1 - 0.7 * sa), ("A", "brillo"): g_a * (1 - sa),
           ("B", "graves"): g_b * (1 - sb), ("B", "cuerpo"): g_b * (1 - 0.7 * sb), ("B", "presencia"): g_b * (1 - 0.3 * sb), ("B", "brillo"): g_b}
    nombres = ("graves", "cuerpo", "presencia", "brillo")
    viz_b, viz_g, viz_n = _viz_natural(ba, bb, gan, nombres, lin_a, lin_b, ca, cb, Nout)
    plan = replace(plan, razon=f"{plan.razon} · puente (filtro y eco)")
    avance(1.0, "Listo")
    return Fusion(audio=mezcla, a_inicio=a_ini, b_continua=sB + rem, plan=plan, gan_a=lin_a, gan_b=lin_b,
                  viz_bandas=viz_b, viz_gan=viz_g, viz_nivel=viz_n, t_swap=float(ti / Nout), compas_seg=float(4 * 60.0 / A.bpm),
                  carriles=CARRILES_NATURAL, siglas=SIGLAS_NATURAL, motor="puente")

# ------------------------------------------------------------------------------------------- motor en vivo (receta de AutoMix)
UMBRAL_VIVO = 0.15           # hasta esta diferencia de velocidad (fracción) se mezcla con el motor en vivo
BLK_VIVO = 1024


def _rejilla_suave(b: np.ndarray, vent: int = 8) -> np.ndarray:
    """Rejilla de beats regular: ajuste ROBUSTO en una ventana móvil de 2*vent+1 beats (siempre completa, también en los bordes).
    Pendiente = mediana de las pendientes entre pares (Theil-Sen), así unos pocos beats mal detectados no la arrastran; luego dos
    pasadas que descartan los beats que se alejan de la recta (como SmartSync de AutoMix). Quita el 'temblor' del detector sin perder
    la deriva real del tempo."""
    b = np.asarray(b, np.float64)
    n = len(b)
    if n < 6:
        return b.copy()
    w = min(n, 2 * vent + 1)
    k = np.arange(n, dtype=np.float64)
    ii, jj = np.triu_indices(w, 1)
    sal = np.empty(n)
    for i in range(n):
        lo = int(min(max(0, i - vent), n - w))
        kk, tt = k[lo:lo + w], b[lo:lo + w]
        p = float(np.median((tt[jj] - tt[ii]) / (kk[jj] - kk[ii])))
        q = float(np.median(tt - p * kk))
        for tol in (0.05, 0.02):
            ok = np.abs(tt - (q + p * kk)) < tol
            if ok.sum() < max(5, w // 2):
                break
            p, q = np.polyfit(kk[ok], tt[ok], 1)
        sal[i] = q + p * i
    return sal


def _env_graves(mono: np.ndarray, fc: float = 150.0, hop: int = 220, banda: tuple | None = None) -> np.ndarray:
    """Onsets de graves (bombo), un valor cada 5 ms. Con `banda=(lo, hi)` mide esa banda en vez de los graves (golpes de caja, guitarra, platillos)."""
    if banda:
        sos = signal.butter(2, [banda[0], banda[1]], btype="band", fs=SR, output="sos")
    else:
        sos = signal.butter(4, fc, btype="low", fs=SR, output="sos")
    e = np.abs(signal.sosfiltfilt(sos, mono))
    m = hop * 4
    e = np.convolve(e, np.ones(m) / m, mode="same")[::hop]
    return np.maximum(0.0, np.diff(e, prepend=e[0]))


def _lag_bombo(ea: np.ndarray, eb: np.ndarray, maxlag: int) -> tuple[int, float]:
    """Desfase (pasos de 5 ms) que mejor empareja los bombos de A y B; >0 = B va adelantado y hay que retrasarlo. Y qué tan claro es (0..1)."""
    n = min(len(ea), len(eb))
    ea, eb = ea[:n] - ea[:n].mean(), eb[:n] - eb[:n].mean()
    den = float(np.sqrt((ea ** 2).sum() * (eb ** 2).sum())) + 1e-12
    mejor, vmax = 0, -1e30
    for lag in range(-maxlag, maxlag + 1):
        v = float(np.dot(ea[lag:], eb[:n - lag]) if lag >= 0 else np.dot(ea[:n + lag], eb[-lag:]))
        if v > vmax:
            vmax, mejor = v, lag
    return mejor, vmax / den


def _warp_wsola(audio: np.ndarray, lo: int, src: np.ndarray) -> np.ndarray:
    """Lee `audio` (con origen en la muestra global `lo`) siguiendo el mapa de posiciones `src` (una por muestra de salida) con WSOLA:
    cambia la velocidad SIN cambiar el tono y sin manchar los golpes. Un control suave corrige la deriva respecto al mapa."""
    x = np.ascontiguousarray(audio.T, dtype=np.float32)
    w = WsolaFlujo()
    w.reset(float(src[0] - lo))
    n = len(src)
    sal = np.zeros((n, 2), np.float32)
    i = 0
    while i < n:
        m = min(BLK_VIVO, n - i)
        j = min(i + m, n - 1)
        g = (src[j] - src[i]) / max(1, j - i)
        corr = float(np.clip((src[i] - lo - w.head) / (4.0 * BLK_VIVO), -0.02, 0.02))
        blk, _fin = w.leer(x, m, g + corr)
        sal[i:i + m] = blk
        i += m
    return np.ascontiguousarray(sal.T)


def _warp_wsola_tono(audio: np.ndarray, lo: int, src: np.ndarray, semi: np.ndarray) -> np.ndarray:
    """Como `_warp_wsola`, pero además sube o baja el tono `semi` semitonos (uno por muestra de salida) SIN cambiar la duración.
    Se lee el audio con WSOLA a la velocidad que corresponde y después se remuestrea por el factor de tono: el tiempo de salida
    sigue siendo el del mapa `src`, así los golpes caen donde tienen que caer."""
    from scipy.ndimage import map_coordinates
    n = len(src)
    r = 2.0 ** (np.asarray(semi, np.float64) / 12.0)
    tau = np.concatenate([[0.0], np.cumsum(r)[:-1]])                  # posición en el audio "pre-remuestreo" de cada muestra de salida
    npre = int(np.ceil(tau[-1])) + 4
    t_de_tau = np.interp(np.arange(npre, dtype=np.float64), tau, np.arange(n, dtype=np.float64))
    src_pre = np.interp(t_de_tau, np.arange(n, dtype=np.float64), src)
    w = _warp_wsola(audio, lo, src_pre)
    sal = np.empty((2, n), np.float32)
    for c in range(2):
        sal[c] = map_coordinates(w[c], [tau], order=3, mode="nearest").astype(np.float32)
    return sal


def _renderizar_vivo(A, audio_a, B, audio_b, plan, ajustes, bA, bB, N, avance):
    """Fusión con el audio ORIGINAL de los dos temas y la receta del cruce Auto de AutoMix:
      · rejilla de compases regular (regresión robusta) y tempo común que se desliza de A a B mientras suenan juntos,
      · estiramiento WSOLA (sin cambiar el tono), afinado fino de la fase por correlación del bombo,
      · medios y agudos: B entra a nivel pleno y A cae con coseno^2,5 siguiendo el fader (con pausa y congelado),
      · graves: de uno solo a la vez, relevo seco sobre la línea de compás dentro de la pausa.
    Devuelve None si la diferencia de tempo es demasiado grande (entonces sigue el flujo de siempre)."""
    bA, bB = _rejilla_suave(bA), _rejilla_suave(bB)
    perA, perB = np.diff(bA), np.diff(bB)
    if perA.min() < 0.12 or perB.min() < 0.12:
        return None
    n_b = len(bA)
    b_idx = np.arange(n_b - 1, dtype=np.float64)
    w = _suavizado((b_idx - 0.18 * N) / (0.50 * N))                  # 0 = tempo de A, 1 = tempo de B
    per = np.exp((1 - w) * np.log(perA) + w * np.log(perB))
    t_out = np.concatenate([[0.0], np.cumsum(per)])
    Nout = int(round(t_out[N] * SR))
    n_axis = np.arange(Nout, dtype=np.float64)
    src_a = PchipInterpolator(t_out, bA * SR)(n_axis / SR)
    src_b = PchipInterpolator(t_out, bB * SR)(n_axis / SR)
    desv = max(float(np.abs(np.gradient(src_a) - 1.0).max()), float(np.abs(np.gradient(src_b) - 1.0).max()))
    if desv > UMBRAL_VIVO:
        return None
    src_a += round(src_a[0]) - src_a[0]
    beat_a_n = PchipInterpolator(np.arange(len(t_out), dtype=np.float64), t_out * SR)
    a_inicio = int(round(src_a[0]))

    # progreso del fader (pausa y congelado incluidos): igual que el resto de motores
    P, Q, ri_, R_ = disposicion(plan, A, ajustes)
    idx_b = np.minimum(np.arange(n_b, dtype=np.float64), N)
    nodos = list(zip([0, max(0.0, P - ri_), P, Q, Q + 0.6 * R_, Q + R_, N], [0.0, 0.0, 0.38, 0.55, 0.92, 1.0, 1.0]))
    h_c, fb_c, k_c = parametros_congelado(plan, A, ajustes)
    if fb_c > 0:
        nodos = _congelar(nodos, h_c, fb_c, k_c, N)
    u_b = _pchip_mono([x for x, _ in nodos], [y for _, y in nodos])(idx_b)
    u = np.clip(PchipInterpolator(t_out * SR, u_b)(n_axis), 0.0, 1.0).astype(np.float32)

    # --- afinado fino: el bombo de B cae sobre el de A (primeros ~24 beats)
    avance(0.08, "Alineando los temas a un mismo pulso…")
    marg = 4096 + int(0.30 * float(perA.mean()) * SR)
    lo_a, hi_a = int(src_a.min()) - marg, int(src_a.max()) + marg
    lo_b, hi_b = int(src_b.min()) - marg, int(src_b.max()) + marg
    xa = _recorte(audio_a, lo_a, hi_a)
    xb = _recorte(audio_b, lo_b, hi_b)
    nl = min(Nout, int(t_out[min(24, N)] * SR))
    ma = _leer_fraccionado(xa, src_a[:nl], lo_a).mean(axis=0)
    mb = _leer_fraccionado(xb, src_b[:nl], lo_b).mean(axis=0)
    maxlag = int(0.25 * float(perA.mean()) / (220 / SR))
    lag, claro = _lag_bombo(_env_graves(ma), _env_graves(mb), maxlag)
    if claro < 0.10:                                                  # bombo flojo: se prueba con el cuerpo (caja, guitarra, platillos)
        lag2, claro2 = _lag_bombo(_env_graves(ma, banda=(150.0, 5000.0)), _env_graves(mb, banda=(150.0, 5000.0)), maxlag)
        if claro2 > claro:
            lag, claro = lag2, claro2
    d = int(round(lag * 220)) if claro >= 0.10 else 0
    if d:
        pos = n_axis - d                                              # B retrasado d muestras: lee donde leía d muestras antes
        s2 = np.interp(pos, n_axis, src_b)
        r0 = float(src_b[1] - src_b[0])
        r1 = float(src_b[-1] - src_b[-2])
        s2 = np.where(pos < 0, src_b[0] + pos * r0, s2)
        s2 = np.where(pos > Nout - 1, src_b[-1] + (pos - (Nout - 1)) * r1, s2)
        src_b = s2
    c = src_b[-1] - (Nout - 1)
    src_b += round(c) - c                                             # el final de B cae en una muestra entera
    b_continua = int(round(src_b[-1])) + 1

    # --- estirado WSOLA de cada tema
    avance(0.30, "Estirando el tema que sale…")
    ya = _warp_wsola(xa, lo_a, src_a)
    avance(0.55, "Estirando el tema que entra…")
    if float((ajustes or {}).get("i_armonia", 0.0)) >= 0.5 and abs(plan.shift) >= 0.3:
        # armonía activa: B suena afinado con A durante la pausa y vuelve a su tono real después del cambio de bajo (igual que en «capas»)
        semi_b = plan.shift * (1.0 - _suavizado((idx_b - (Q - 2)) / max(1.0, (Q + R_) - (Q - 2))))
        semi = PchipInterpolator(t_out * SR, semi_b)(n_axis)
        yb = _warp_wsola_tono(xb, lo_b, src_b, semi)
    else:
        yb = _warp_wsola(xb, lo_b, src_b)
    # empalme exacto con la reproducción normal: el final de B es el audio original, con un cruce corto desde el estirado
    X = min(int(0.06 * SR), Nout // 4)
    exacto = _leer_fraccionado(xb, src_b[-X:], lo_b)
    rampa = np.linspace(0.0, 1.0, X, dtype=np.float32)[None, :]
    yb[:, -X:] = yb[:, -X:] * (1 - rampa) + exacto * rampa
    del xa, xb

    # --- curvas: graves del relevo de siempre; medios y agudos con la ley de AutoMix (sobre el fader u, que ya trae pausa y congelado)
    rec = receta_natural(plan, A, B, ajustes)
    ent_b = float((ajustes or {}).get("_ent_b", 0.20))               # cuándo entra B en el fader (lo fija la postura)
    gam_a = float((ajustes or {}).get("_gamma_a", 2.5))              # qué tan pronto se va A
    vs = np.maximum(0.0, np.cos(np.clip(u, 0, 1) * np.pi / 2)) ** gam_a
    ve = np.where(u <= ent_b, 0.0, np.sin(np.clip((u - ent_b) / 0.14, 0, 1) * np.pi / 2))
    tb = np.arange(0.0, N + 0.01, 0.5)
    nb = np.clip(np.round(beat_a_n(tb)).astype(int), 0, Nout - 1)
    for nombre in ("cuerpo", "presencia", "brillo"):
        rec[("A", nombre)] = [(float(t), float(vs[n])) for t, n in zip(tb, nb)]
        rec[("B", nombre)] = [(float(t), float(ve[n])) for t, n in zip(tb, nb)]
    _separar_voces(rec, plan, A, B, ajustes, N, Q, banda="presencia", fuerza=0.9)        # las voces de A y B no suenan juntas
    _separar_voces(rec, plan, A, B, ajustes, N, Q, banda="cuerpo", fuerza=0.55)
    return _renderizar_natural(A, audio_a, B, audio_b, plan, ajustes, src_a, src_b, a_inicio, b_continua, beat_a_n, u, per, Nout, N,
                               avance, xa=ya, xb=yb, rec=rec, motor="vivo")


def _viz_natural(ba, bb, gan, nombres, lin_a, lin_b, ca, cb, Nout):
    paso = SR // FPS_VIZ
    nf = Nout // paso
    if nf < 2:
        z = np.zeros((1, BANDAS_VIZ, 2), np.float16)
        return z, np.zeros((1, 4, 2), np.float16), np.zeros((1, 4, 2), np.float16)
    n_fft = 2048
    win = np.hanning(n_fft).astype(np.float32)
    edges = np.geomspace(30.0, 16000.0, BANDAS_VIZ + 1)
    fr = np.fft.rfftfreq(n_fft, 1 / SR)
    asign = np.clip(np.searchsorted(edges, fr) - 1, -1, BANDAS_VIZ)
    sel = [asign == b for b in range(BANDAS_VIZ)]
    bandas = np.zeros((nf, BANDAS_VIZ, 2), np.float32)
    for k, x in enumerate((ca.mean(axis=0), cb.mean(axis=0))):
        xx = np.pad(x, (n_fft, n_fft))
        for f in range(nf):
            c = f * paso + n_fft // 2 + n_fft
            m = np.abs(np.fft.rfft(xx[c - n_fft // 2: c + n_fft // 2] * win)) ** 2
            for b in range(BANDAS_VIZ):
                if sel[b].any():
                    bandas[f, b, k] = m[sel[b]].sum()
    bandas = np.sqrt(bandas)
    bandas = np.log1p(bandas * 40.0) / np.log1p(40.0 * max(1e-6, bandas.max()))
    gv = np.zeros((nf, 4, 2), np.float32)
    nv = np.zeros((nf, 4, 2), np.float32)
    for i, nombre in enumerate(nombres):
        for k, (bnd, key, lin) in enumerate(((ba, "A", lin_a), (bb, "B", lin_b))):
            g = gan[(key, nombre)]
            x = bnd[i].mean(axis=0) * g * lin
            nv[:, i, k] = np.sqrt((x[:nf * paso].reshape(nf, paso) ** 2).mean(axis=1))
            gv[:, i, k] = g[:nf * paso].reshape(nf, paso).mean(axis=1)
    nv = np.clip(nv / (np.percentile(nv, 98) + 1e-9), 0, 1.2)
    return bandas.astype(np.float16), gv.astype(np.float16), nv.astype(np.float16)


def fusion_simple(audio_a: np.ndarray, gan_a: float, audio_b: np.ndarray, gan_b: float, seg: float = 6.0, bpm: float = 100.0) -> Fusion:
    """Cruce de emergencia (sin análisis de beats): el final de A y el comienzo de B se solapan `seg` segundos con un cruce
    de potencia constante y el cambio de graves en el medio, para que nunca haya un corte seco ni un silencio entre temas."""
    n = int(min(seg * SR, audio_a.shape[1] // 2, audio_b.shape[1] // 2))
    xa = np.ascontiguousarray(audio_a[:, audio_a.shape[1] - n:])
    xb = np.ascontiguousarray(audio_b[:, :n])
    ba, bb = _bandas(xa), _bandas(xb)
    t = np.linspace(0.0, 1.0, n)
    s = _suavizado(t)
    g_a, g_b = np.cos(0.5 * np.pi * s), np.sin(0.5 * np.pi * s)                       # potencia constante
    sw = _suavizado((t - 0.5) / 0.25 + 0.5)                                            # cambio de graves en el centro (1,5 s)
    gan = {("A", "graves"): (1 - sw).astype(np.float32), ("B", "graves"): sw.astype(np.float32)}
    for nom in ("cuerpo", "presencia", "brillo"):
        gan[("A", nom)], gan[("B", nom)] = g_a.astype(np.float32), g_b.astype(np.float32)
    nombres = ("graves", "cuerpo", "presencia", "brillo")
    ca = sum(ba[i] * gan[("A", nm)] for i, nm in enumerate(nombres)) * gan_a
    cb = sum(bb[i] * gan[("B", nm)] for i, nm in enumerate(nombres)) * gan_b
    mezcla = limitar_mezcla(ca + cb)
    viz_b, viz_g, viz_n = _viz_natural(ba, bb, gan, nombres, gan_a, gan_b, ca, cb, n)
    L = max(1, int(round(seg / (4 * 60.0 / bpm))))
    plan = Plan(a0=0, b0=0, L=L, k=1.0, shift=0.0, modo="limpia", puntaje=0.0, armonia=0.0, razon="cruce simple (la fusión no estaba lista)")
    return Fusion(audio=mezcla, a_inicio=int(audio_a.shape[1] - n), b_continua=n, plan=plan, gan_a=gan_a, gan_b=gan_b,
                  viz_bandas=viz_b, viz_gan=viz_g, viz_nivel=viz_n, t_swap=0.5, compas_seg=4 * 60.0 / bpm,
                  carriles=CARRILES_NATURAL, siglas=SIGLAS_NATURAL, motor="simple")


def _viz(ren_a, ren_b, gan, lin_a, lin_b, ra, rb, dr, Nout):
    """Energía por banda de A y de B tras el procesado, y ganancia/nivel de cada capa, por cuadro de vídeo."""
    paso = SR // FPS_VIZ
    nf = Nout // paso
    if nf < 2:
        z = np.zeros((1, BANDAS_VIZ, 2), np.float16)
        return z, np.zeros((1, 4, 2), np.float16), np.zeros((1, 4, 2), np.float16)
    n_fft = 2048
    win = np.hanning(n_fft).astype(np.float32)
    edges = np.geomspace(30.0, 16000.0, BANDAS_VIZ + 1)
    fr = np.fft.rfftfreq(n_fft, 1 / SR)
    asign = np.clip(np.searchsorted(edges, fr) - 1, -1, BANDAS_VIZ)
    bandas = np.zeros((nf, BANDAS_VIZ, 2), np.float32)
    for k, x in enumerate((ra.mean(axis=0) + 0.5 * dr.mean(axis=0) * 0, rb.mean(axis=0))):
        xx = np.pad(x, (n_fft, n_fft))
        for f in range(nf):
            c = f * paso + n_fft // 2 + n_fft
            seg = xx[c - n_fft // 2: c + n_fft // 2] * win
            m = np.abs(np.fft.rfft(seg)) ** 2
            for bnd in range(BANDAS_VIZ):
                sel = asign == bnd
                if sel.any():
                    bandas[f, bnd, k] = m[sel].sum()
    # la batería se reparte según de quién sea en ese momento
    d_a = ren_a["bateria"].mean(axis=0) * gan[("A", "bateria")] * lin_a
    d_b = ren_b["bateria"].mean(axis=0) * gan[("B", "bateria")] * lin_b
    for k, x in enumerate((d_a, d_b)):
        xx = np.pad(x, (n_fft, n_fft))
        for f in range(nf):
            c = f * paso + n_fft // 2 + n_fft
            m = np.abs(np.fft.rfft(xx[c - n_fft // 2: c + n_fft // 2] * win)) ** 2
            for bnd in range(BANDAS_VIZ):
                sel = asign == bnd
                if sel.any():
                    bandas[f, bnd, k] += m[sel].sum()
    bandas = np.sqrt(bandas)
    bandas = np.log1p(bandas * 40.0) / np.log1p(40.0 * max(1e-6, bandas.max()))
    gv = np.zeros((nf, 4, 2), np.float32)
    nv = np.zeros((nf, 4, 2), np.float32)
    for i, nombre in enumerate(CAPAS):
        for k, (ren, key, lin) in enumerate(((ren_a, "A", lin_a), (ren_b, "B", lin_b))):
            g = gan[(key, nombre)]
            x = ren[nombre].mean(axis=0) * g * lin
            e = np.sqrt((x[:nf * paso].reshape(nf, paso) ** 2).mean(axis=1))
            nv[:, i, k] = e
            gv[:, i, k] = g[:nf * paso].reshape(nf, paso).mean(axis=1)
    nv = np.clip(nv / (np.percentile(nv, 98) + 1e-9), 0, 1.2)
    return bandas.astype(np.float16), gv.astype(np.float16), nv.astype(np.float16)


# ------------------------------------------------------------------------------------------- guardado en la base de datos
VERSION_FUSION = 7                         # súbela si cambia el algoritmo de fusión (las guardadas con otra versión se ignoran)


def empaquetar(f: Fusion) -> bytes:
    """Convierte la fusión en bytes comprimidos (audio en 16 bits) para guardarla en la base de datos."""
    import io
    import json
    import zlib
    from dataclasses import asdict

    def jd(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o).__name__)

    meta = {"a_inicio": int(f.a_inicio), "b_continua": int(f.b_continua), "gan_a": float(f.gan_a), "gan_b": float(f.gan_b),
            "t_swap": float(f.t_swap), "carriles": list(f.carriles), "siglas": list(f.siglas), "motor": f.motor,
            "compas_seg": float(f.compas_seg), "congelado": list(f.congelado), "plan": asdict(f.plan)}
    pcm = np.clip(np.rint(f.audio * 32767.0), -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    np.savez(buf, audio=pcm, viz_bandas=f.viz_bandas, viz_gan=f.viz_gan, viz_nivel=f.viz_nivel,
             meta=np.frombuffer(json.dumps(meta, default=jd).encode("utf-8"), np.uint8))
    return zlib.compress(buf.getvalue(), 1)


def desempaquetar(blob: bytes) -> Fusion:
    """Inversa de `empaquetar`. Lanza ValueError si los datos están dañados."""
    import io
    import json
    import zlib
    try:
        with np.load(io.BytesIO(zlib.decompress(blob)), allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"]).decode("utf-8"))
            audio = (z["audio"].astype(np.float32) / 32767.0)
            vb, vg, vn = z["viz_bandas"], z["viz_gan"], z["viz_nivel"]
        plan = Plan(**meta["plan"])
        return Fusion(audio=np.ascontiguousarray(audio), a_inicio=meta["a_inicio"], b_continua=meta["b_continua"], plan=plan,
                      gan_a=meta["gan_a"], gan_b=meta["gan_b"], viz_bandas=vb, viz_gan=vg, viz_nivel=vn, t_swap=meta["t_swap"],
                      carriles=tuple(meta["carriles"]), siglas=tuple(meta["siglas"]), motor=meta["motor"], compas_seg=meta["compas_seg"],
                      congelado=tuple(meta.get("congelado", (0.0, 1.0))))
    except (zlib.error, KeyError, TypeError, OSError, EOFError) as e:
        raise ValueError(f"fusión guardada dañada: {e}") from e
