"""Rutas, constantes y configuración persistente de NEXUS."""
from __future__ import annotations

import json
import os
from pathlib import Path

SR = 44100                       # frecuencia de trabajo de todo el motor
EXTENSIONES = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aiff", ".aif", ".opus")

RAIZ = Path(__file__).resolve().parent.parent


def carpeta_datos() -> Path:
    """Carpeta donde se guardan los análisis y la configuración (se puede forzar con NEXUS_DATOS)."""
    base = os.environ.get("NEXUS_DATOS")
    ruta = Path(base) if base else RAIZ / "datos"
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def archivo_config() -> Path:
    return carpeta_datos() / "config_nexus.json"


def leer_config() -> dict:
    try:
        with open(archivo_config(), "r", encoding="utf-8") as f:
            datos = json.load(f)
        return datos if isinstance(datos, dict) else {}
    except (OSError, ValueError):
        return {}


def guardar_config(datos: dict) -> None:
    try:
        with open(archivo_config(), "w", encoding="utf-8") as f:
            json.dump(datos, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


# Ajustes del sistema de mezcla (se guardan en config_nexus.json bajo "mezcla")
AJUSTES_MEZCLA = {
    "auto": 1.0,           # 1 = postura automática (NEXUS elige cómo mezclar cada par; ignora pausa, rampas, voz y bajo) | 0 = manual
    "pausa": 5.0,          # segundos que el cruce se queda quieto con los dos temas sonando juntos (0 = sin pausa)
    "rampa_in": 0.0,       # segundos de la rampa de entrada hasta la pausa (0 = Auto: usa el tiempo disponible)
    "rampa_out": 0.0,      # segundos de la rampa de salida, después de la pausa (0 = Auto)
    "motor": "natural",    # "natural": mezcla el audio original de los dos temas | "capas": separa en batería/bajo/voz/ambiente (experimental)
    "voz_rampa": 2.0,      # segundos que dura el relevo de las voces (las voces de A y B nunca suenan juntas)
    "bajo_rampa": 1.0,     # segundos que dura el cruce del bajo (cae en la mitad de la pausa, sobre una línea de compás)
    "congelar": 5.0,       # segundos que los faders quedan quietos cuando el cursor toca la línea blanca punteada (0 = no congelar)
    "inicio_pct": 0.0,     # % del tema donde empieza la mezcla con el siguiente, en todos los temas (0 = Auto: NEXUS elige el mejor punto de cada par)
}
# Ideas de mezcla que el usuario puede ir sumando con un tilde (1 = activa). Las que dicen False todavía no están programadas.
IDEAS_MEZCLA = {
    "i_rejilla": ("Confianza de rejilla: si los beats de un tema no son fiables (en vivo, popurrí, tempo que se mueve), mezcla con la postura segura", True),
    "i_estructura": ("Mezclar por estructura: la fusión empieza y termina en cambios de sección y no pisa estribillo con estribillo", True),
    "i_armonia": ("Armonía activa: sube o baja el tono de B durante la fusión para que encaje con A y lo devuelve a su tono real al terminar", True),
    "i_voces": ("Voces por frases: el relevo de voz se hace en un hueco de voz de A y se prefieren fusiones con ese hueco", True),
    "i_gustos": ("Aprender de mis gustos: aparecen 👍/👎 junto a ⚡ para votar cada mezcla y NEXUS ajusta las posturas a mi estilo", True),
    "i_puentes": ("Puentes: si dos temas no combinan (tempos muy distintos o armonía y voces que chocan), usa filtro y eco en vez de forzar la fusión", True),
}
for _k in IDEAS_MEZCLA:
    AJUSTES_MEZCLA[_k] = 0.0
LIMITES_AJUSTES = {"pausa": (0.0, 20.0), "rampa_in": (0.0, 30.0), "rampa_out": (0.0, 30.0), "voz_rampa": (0.5, 10.0), "bajo_rampa": (0.5, 8.0), "congelar": (0.0, 10.0), "inicio_pct": (0.0, 100.0)}


def ajustes_mezcla(cfg: dict | None = None) -> dict:
    """Ajustes de mezcla válidos (con los valores por defecto donde falten o estén fuera de rango)."""
    guardado = (cfg if cfg is not None else leer_config()).get("mezcla", {})
    sal = dict(AJUSTES_MEZCLA)
    if isinstance(guardado, dict):
        for k, (lo, hi) in LIMITES_AJUSTES.items():
            try:
                sal[k] = min(hi, max(lo, float(guardado.get(k, sal[k]))))
            except (TypeError, ValueError):
                pass
        for k in IDEAS_MEZCLA:
            if k in guardado:
                try:
                    sal[k] = 1.0 if float(guardado[k]) >= 0.5 else 0.0
                except (TypeError, ValueError):
                    pass
        if "auto" in guardado:
            try:
                sal["auto"] = 1.0 if float(guardado["auto"]) >= 0.5 else 0.0
            except (TypeError, ValueError):
                pass
        if guardado.get("motor") in ("natural", "capas"):
            sal["motor"] = guardado["motor"]
    return sal
