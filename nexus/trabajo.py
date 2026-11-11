"""Trabajos pesados que corren en un PROCESO APARTE (así el audio y la interfaz nunca se traban).

Estas funciones deben estar a nivel de módulo para que Windows pueda enviarlas al proceso hijo.
"""
from __future__ import annotations

import os
import sys


def iniciar_hijo() -> None:
    """Baja la prioridad del proceso hijo para que no le robe CPU al audio."""
    try:
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)   # BELOW_NORMAL
        else:
            os.nice(8)
    except Exception:
        pass


def iniciar_hijo_normal() -> None:
    """Hijo con prioridad normal (para lo que el usuario está esperando, como el botón ⚡)."""


def ventana_par(A, B, plan, margen_s: float = 6.0) -> tuple[int, int, int, int]:
    """Muestras (lo_a, hi_a, lo_b, hi_b) de cada tema que la fusión realmente usa (con margen)."""
    from .config import SR
    from .fusion import _beats_b_efectivos
    n = 4 * plan.L + 1 + 12
    ia0, ib0 = int(A.compases[plan.a0]), int(B.compases[plan.b0])
    ba, bb = A.beats[ia0:ia0 + n], _beats_b_efectivos(B, ib0, n, plan.k)
    m = int(margen_s * SR)
    return (max(0, int(ba.min() * SR) - m), int(ba.max() * SR) + m, max(0, int(bb.min() * SR) - m), int(bb.max() * SR) + m)


def _limpiar() -> None:
    """Devuelve al sistema la memoria que ya no hace falta (en un proceso hijo que sigue vivo para el próximo trabajo)."""
    import gc
    gc.collect()


def analizar_ruta(ruta: str) -> dict:
    """Analiza un tema y devuelve el resultado como dict (serializable)."""
    try:
        return _analizar_ruta(ruta)
    except MemoryError as e:                       # mensaje claro (un MemoryError no trae texto) para que el motor lo reintente
        raise RuntimeError("sin memoria suficiente para analizar el tema") from e
    finally:
        _limpiar()


def _analizar_ruta(ruta: str) -> dict:
    from .analisis import analizar
    return analizar(ruta).a_dict()


def renderizar_par(ruta_a: str, dict_a: dict, ruta_b: str, dict_b: dict, plan_dict: dict, ajustes: dict | None = None,
                   ca: str = "", cb: str = "", firma: str = "", audio_a=None, audio_b=None):
    """Renderiza la fusión A→B y la GUARDA en la base de datos (así la próxima vez es solo buscarla).

    Devuelve None si quedó guardada; si no se pudo guardar devuelve la fusión para usarla igual."""
    from . import basedatos
    from .analisis import Analisis
    from .fusion import empaquetar, renderizar
    from .audio_io import cargar_ventana
    from .config import SR
    from .fusion import _beats_b_efectivos
    from .planificador import Plan
    A, B = Analisis.de_dict(dict_a), Analisis.de_dict(dict_b)
    plan = Plan(**plan_dict)
    # solo se decodifica el tramo que la fusión realmente usa (±6 s): no los dos temas completos
    # (si el motor ya tenía el audio en memoria, lo manda recortado y no se decodifica nada)
    import time
    t0 = time.time()
    lo_a, hi_a, lo_b, hi_b = ventana_par(A, B, plan)
    if audio_a is None:
        audio_a = cargar_ventana(ruta_a, lo_a, hi_a)
    if audio_b is None:
        audio_b = cargar_ventana(ruta_b, lo_b, hi_b)
    t_dec = time.time() - t0
    try:
        t1 = time.time()
        fus = renderizar(A, audio_a, B, audio_b, plan, ajustes=ajustes)
        fus.tiempos = {"decodificar": round(t_dec, 2), "calcular": round(time.time() - t1, 2), "motor": fus.motor}
    except MemoryError as e:
        raise RuntimeError("sin memoria suficiente para calcular la mezcla") from e
    finally:
        del audio_a, audio_b
        _limpiar()
    if ca and cb and firma and basedatos.guardar_fusion(ca, cb, firma, empaquetar(fus), A.nombre, B.nombre):
        return None
    return fus
