"""Postura inteligente de la mezcla: NEXUS decide, para CADA par de temas, cómo mezclarlos.

Mide lo que importa de la pareja (diferencia de tempo, armonía, choque de voces, presencia de bajo, tiempo disponible) y adopta una
de cuatro posturas; cada una fija el largo de la pausa, el relevo de las voces, el cruce del bajo, cuándo entra B y qué tan pronto se va A.
Los ajustes manuales solo cuentan si el usuario elige «Manual» (o para lo que no depende del par, como el congelado de los faders).
"""
from __future__ import annotations

import numpy as np

# nombre: (pausa como fracción de la mezcla, relevo de voz s, cruce del bajo s, punto del fader donde entra B, caída de A)
POSTURAS = {
    "larga y suave":       (0.30, 3.0, 1.5, 0.15, 2.8),     # todo combina: mucho tiempo juntos, relevos lentos
    "estándar":            (0.20, 2.0, 1.0, 0.20, 2.5),     # la mezcla de siempre de AutoMix
    "cuidando las voces":  (0.16, 1.5, 1.0, 0.25, 2.0),     # cantan a la vez: poco solape y relevo de voz firme
    "firme y corta":       (0.10, 1.0, 0.5, 0.30, 1.8),     # tempos o tonos que chocan: se pasa rápido y limpio
}


def _presencia(v) -> np.ndarray:
    return np.clip((np.asarray(v, np.float64) - 0.35) / 0.30, 0.0, 1.0)


def _media(x, a: int, n: int, defecto: float) -> float:
    x = np.asarray(x, np.float64)[max(0, a):max(0, a) + max(1, n)]
    return float(x.mean()) if len(x) else defecto


def rasgos(plan, A, B) -> dict:
    """Qué tiene de particular este par (todo en 0..1 salvo el tiempo)."""
    L, Lb = int(plan.L), int(plan.Lb)
    vA = np.asarray(A.voz, np.float64)[plan.a0:plan.a0 + L]
    vB = np.asarray(B.voz, np.float64)[plan.b0:plan.b0 + Lb]
    if len(vA) >= 2 and len(vB) >= 1:
        pos_b = np.clip(np.round(np.arange(len(vA)) / plan.k).astype(int), 0, len(vB) - 1)
        choque = float(np.mean(_presencia(vA) * _presencia(vB[pos_b])))
    else:
        choque = 0.0
    return dict(
        tmix=4.0 * L * 60.0 / max(40.0, float(A.bpm)),
        dif_tempo=abs(float(B.bpm) * float(plan.k) / max(1e-6, float(A.bpm)) - 1.0),
        armonia=float(plan.armonia),
        choque_voces=choque,
        bajo_a=_media(A.bajo, plan.a0, L, 0.5),
        bajo_b=_media(B.bajo, plan.b0, Lb, 0.5),
    )


def inestabilidad(A) -> float:
    """Cuánto se mueve el tempo de un tema (0 = firme como un reloj). Mide el rango del período local de los beats
    (mediana móvil de 17) respecto de su mediana: un popurrí o un tema en vivo con tempo cambiante da valores altos."""
    b = np.asarray(A.beats, np.float64)
    if len(b) < 40:
        return 0.0
    per = np.diff(b)
    per = per[(per > 0.2) & (per < 2.0)]
    if len(per) < 40:
        return 0.0
    k = 17
    pad = np.pad(per, (k // 2, k // 2), mode="edge")
    loc = np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1)
    return float((np.percentile(loc, 95) - np.percentile(loc, 5)) / max(1e-6, np.median(loc)))


UMBRAL_INESTABLE = 0.08


NOMBRES = tuple(POSTURAS)


def vector(r: dict) -> list[float]:
    """Resumen numérico del par (0..1 aprox.) con el que se comparan las mezclas que el usuario votó."""
    return [min(1.5, r["dif_tempo"] / 0.15), float(np.clip(r["armonia"], 0.0, 1.0)), float(np.clip(r["choque_voces"], 0.0, 1.0)),
            min(1.0, r["tmix"] / 40.0)]


def _por_gustos(v: list[float], regla: str, votos: list) -> str:
    """Corrige la postura de las reglas con lo que el usuario votó en pares parecidos (👍 suma, 👎 resta)."""
    if len(votos) < 3:
        return regla
    v = np.asarray(v, np.float64)
    pun = {n: 0.0 for n in NOMBRES}
    for x in votos:
        try:
            d2 = float(np.sum((v - np.asarray(x["r"], np.float64)) ** 2))
            if x["p"] in pun:
                pun[x["p"]] += float(np.exp(-d2 / 0.15)) * (1.0 if int(x["v"]) > 0 else -1.0)
        except (KeyError, TypeError, ValueError):
            continue
    mejor = max(pun, key=pun.get)
    if pun[regla] < -0.3:                                  # la postura de las reglas ya se votó mal en pares parecidos: se cambia
        otros = {n: s for n, s in pun.items() if n != regla}
        n2 = max(otros, key=otros.get)
        return n2 if otros[n2] > pun[regla] else regla
    if pun[mejor] > pun[regla] + 0.5:                      # otra postura gustó claramente más
        return mejor
    return regla


def elegir(plan, A, B, aj: dict | None) -> tuple[dict, str]:
    """Devuelve (ajustes efectivos, nombre de la postura). Si el usuario eligió «Manual», no toca nada (nombre vacío)."""
    aj = dict(aj or {})
    if float(aj.get("auto", 1.0)) < 0.5:
        return aj, ""
    r = rasgos(plan, A, B)
    inestable = float(aj.get("i_rejilla", 0.0)) >= 0.5 and max(inestabilidad(A), inestabilidad(B)) > UMBRAL_INESTABLE
    if inestable:
        nombre = "firme y corta"
    elif r["dif_tempo"] > 0.10 or (r["armonia"] < 0.45 and r["choque_voces"] > 0.25):
        nombre = "firme y corta"
    elif r["choque_voces"] >= 0.35:
        nombre = "cuidando las voces"
    elif r["armonia"] >= 0.72 and r["dif_tempo"] <= 0.05 and r["choque_voces"] < 0.25:
        nombre = "larga y suave"
    else:
        nombre = "estándar"
    if float(aj.get("i_gustos", 0.0)) >= 0.5 and not inestable:
        nombre = _por_gustos(vector(r), nombre, aj.get("_gustos") or [])
    try:                                                    # queda anotado para poder votar 👍/👎 esta mezcla
        plan.detalle["gusto"] = dict(r=vector(r), p=nombre)
    except AttributeError:
        pass
    f_pausa, voz, bajo, ent_b, gam = POSTURAS[nombre]
    tmix = r["tmix"]
    pausa = float(np.clip(f_pausa * tmix, 2.0, min(10.0, 0.4 * tmix)))
    if r["bajo_a"] < 0.25 or r["bajo_b"] < 0.25:          # si uno de los dos casi no tiene bajo, el relevo no puede enturbiar: se hace más suave
        bajo *= 1.5
    aj.update(pausa=pausa, rampa_in=0.0, rampa_out=0.0, voz_rampa=float(voz), bajo_rampa=float(bajo),
              _ent_b=float(ent_b), _gamma_a=float(gam))
    return aj, (nombre + " (rejilla inestable)" if inestable else nombre)
